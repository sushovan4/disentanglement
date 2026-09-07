"""ECP measurement of one (config, epoch): the workhorse behind E2-E5.

Per layer:
  * pairwise interaction-quotient matrix over all class pairs, with both
    one-sided permutation p-values and the cancellation guard (r*, fired?);
  * per-fold raw statistics (F disjoint folds) -- the paired-test inputs
    consumed by e3_dynamics.py;
  * baseline stats (linear probe, NC1, CKA to penult) at the final epoch;
  * for 'e2' configs at the final epoch: the two-stage triple scan at
    d0=TRIPLE_PCA_DIM (B=49 screen over all C(10,3) triples, B=999 on
    survivors).

Pair tests parallelize over a process pool sized by SLURM_CPUS_PER_TASK.
Fold sizes adapt to the smallest class (cifar100s has 500/class).

Usage:
  python -m experiments.measure <config_index> <epoch>
  python -m experiments.measure --manifest        # rows for the sbatch array
"""

import os
import sys
import zlib
from itertools import combinations
from multiprocessing import get_context

import numpy as np

from .common import (CONFIGS, FOLD_STAT, N_FOLDS, PCA_DIM, POINTS_PER_CLASS, RESULTS_DIR,
                     STAT, TRIPLE_PCA_DIM, checkpoint_epochs, dump_json)
from .ecp import (cancellation_guard, default_r_grid, mixup_ecp,
                  permutation_test, profile_stat)
from .extract import class_clouds, extract_checkpoint, load_features

B_SWEEP = int(os.environ.get("DISENTANGLE_B", "199"))
B_HEADLINE = 999
B_ESC = int(os.environ.get("DISENTANGLE_B_ESC", str(B_HEADLINE)))
FOLD_M_MAX = 100
N_JOBS = int(os.environ.get("SLURM_CPUS_PER_TASK", str(os.cpu_count() or 4)))

# Follow-up campaign knobs (see cluster/submit_followup.sh):
#   DISENTANGLE_TRIPLE_D0      PCA dim for the triple scan (default 4)
#   DISENTANGLE_TRIPLE_LAYERS  comma-list restricting scanned layers
#   DISENTANGLE_FORCE_TRIPLES  "1": run triples for ANY config/epoch
#   DISENTANGLE_OUT_SUFFIX     appended to the output filename (rerun cells
#                              without clobbering the originals)
TRIPLE_D0 = int(os.environ.get("DISENTANGLE_TRIPLE_D0", str(TRIPLE_PCA_DIM)))
TRIPLE_LAYERS = [s for s in
                 os.environ.get("DISENTANGLE_TRIPLE_LAYERS", "").split(",")
                 if s]
FORCE_TRIPLES = os.environ.get("DISENTANGLE_FORCE_TRIPLES") == "1"
SKIP_PAIRS = os.environ.get("DISENTANGLE_SKIP_PAIRS") == "1"
OUT_SUFFIX = os.environ.get("DISENTANGLE_OUT_SUFFIX", "")
TRIPLE_TAGS = ("e2", "ssl", "vitpre")  # tags that triple-scan by default


def measure_pair(clouds, B, rng):
    r_grid = default_r_grid(clouds)
    res = permutation_test(clouds, r_grid, B=B, stat=STAT, rng=rng)
    fired, rstar = cancellation_guard(res["dchi_obs"], r_grid, clouds)
    return {"quotient": res["quotient"], "T_obs": res["T_obs"],
            "p_up": res["p_up"], "p_down": res["p_down"],
            "null_mean": res["null_mean"], "rstar": rstar,
            "guard_fired": fired}


def _pair_job(args):
    """Top-level worker (picklable): one pair/triple test."""
    key, clouds, B, seed = args
    return key, measure_pair(clouds, B, np.random.default_rng(seed))


def _run_jobs(jobs):
    if len(jobs) <= 1 or N_JOBS <= 1:
        return dict(_pair_job(j) for j in jobs)
    with get_context("fork").Pool(min(N_JOBS, len(jobs))) as pool:
        return dict(pool.imap_unordered(_pair_job, jobs, chunksize=1))


def _seed(name, epoch, extra):
    return zlib.crc32(f"{name}:{epoch}:{extra}".encode())


def measure_config_epoch(cfg, epoch):
    extract_checkpoint(cfg, epoch)
    feats, labels = load_features(cfg, epoch)
    classes = sorted(np.unique(labels).tolist())
    counts = {c: int(np.sum(labels == c)) for c in classes}
    m_head = min(POINTS_PER_CLASS, min(counts.values()))
    fold_m = min(FOLD_M_MAX, min(counts.values()) // N_FOLDS)
    final = epoch == checkpoint_epochs(cfg)[-1]
    out = {"name": cfg.name, "epoch": epoch, "m_head": m_head,
           "fold_m": fold_m, "layers": {}}

    for layer, F in feats.items():
        rec = {"pairs": {}, "folds": {}, "folds_norm": {}}
        if SKIP_PAIRS:
            out["layers"][layer] = rec
            continue
        # ---- headline pairwise matrix (parallel over pairs) ---------------
        jobs = []
        for c1, c2 in combinations(classes, 2):
            clouds = class_clouds(F, labels, [c1, c2], m=m_head, d0=PCA_DIM)
            jobs.append((f"{c1},{c2}", clouds, B_SWEEP,
                         _seed(cfg.name, epoch, f"{layer}:{c1},{c2}")))
        rec["pairs"] = _run_jobs(jobs)
        # ---- fold statistics for paired comparisons (cheap, serial) -------
        for c1, c2 in combinations(classes, 2):
            Ts, Tn = [], []
            for fold in range(N_FOLDS):
                clouds = class_clouds(F, labels, [c1, c2], m=fold_m,
                                      d0=PCA_DIM, fold=fold)
                r_grid = default_r_grid(clouds)
                dchi = mixup_ecp(clouds, r_grid)
                Ts.append(profile_stat(dchi, r_grid, STAT))
                Tn.append(profile_stat(dchi, r_grid, FOLD_STAT))
            rec["folds"][f"{c1},{c2}"] = Ts
            rec["folds_norm"][f"{c1},{c2}"] = Tn
        # ---- baselines (final epoch only; raw features) --------------------
        if final:
            from .baselines import linear_cka, linear_probe_acc, nc1
            sub = np.random.default_rng(0).choice(
                len(labels), size=min(len(labels), 5000), replace=False)
            rec["probe_acc"] = linear_probe_acc(F[sub], labels[sub])
            rec["nc1"] = nc1(F[sub], labels[sub])
            rec["cka_to_penult"] = linear_cka(F[sub], feats["penult"][sub])
        out["layers"][layer] = rec
        print(f"[{cfg.name} ep{epoch}] layer {layer} done", flush=True)

    # ---- triple scan (final epoch of TRIPLE_TAGS, or forced) ----------------
    if FORCE_TRIPLES or (cfg.tag in TRIPLE_TAGS and final):
        out["triple_d0"] = TRIPLE_D0
        out["triples"] = triple_scan(cfg, epoch, feats, labels, classes,
                                     m_head)

    dump_json(out, os.path.join(RESULTS_DIR, "measure",
                                f"{cfg.name}_epoch{epoch}{OUT_SUFFIX}.json"))
    return out


def triple_scan(cfg, epoch, feats, labels, classes, m_head, top=10):
    """Two-stage scan for jointly entangled triples whose pairs are null."""
    scan = {}
    for layer, F in feats.items():
        if TRIPLE_LAYERS and layer not in TRIPLE_LAYERS:
            continue
        jobs = []
        for trip in combinations(classes, 3):
            clouds = class_clouds(F, labels, list(trip), m=m_head,
                                  d0=TRIPLE_D0)
            jobs.append((",".join(map(str, trip)), clouds, 49,
                         _seed(cfg.name, epoch, f"tri:{layer}:{trip}")))
        stage1 = _run_jobs(jobs)
        rows = [{"triple": k, **v} for k, v in stage1.items()]
        rows.sort(key=lambda r: -(r["quotient"] or 0))
        # escalate survivors to headline precision
        esc = []
        for r in rows[:top]:
            trip = list(map(int, r["triple"].split(",")))
            clouds = class_clouds(F, labels, trip, m=m_head,
                                  d0=TRIPLE_D0)
            esc.append((r["triple"], clouds, B_ESC,
                        _seed(cfg.name, epoch, f"esc:{layer}:{trip}")))
        stage2 = _run_jobs(esc)
        for r in rows[:top]:
            r.update({f"headline_{k}": v
                      for k, v in stage2[r["triple"]].items()})
        scan[layer] = rows
        print(f"[{cfg.name} ep{epoch}] triples {layer} done", flush=True)
    return scan


def manifest():
    rows = []
    for i, cfg in enumerate(CONFIGS):
        eps = checkpoint_epochs(cfg)
        if cfg.tag == "pop":  # E5 needs final-epoch features only
            eps = [eps[-1]]
        rows += [(i, e) for e in eps]
    return rows


if __name__ == "__main__":
    if sys.argv[1] == "--manifest":
        for i, e in manifest():
            print(i, e)
        sys.exit(0)
    cfg = CONFIGS[int(sys.argv[1])]
    measure_config_epoch(cfg, int(sys.argv[2]))

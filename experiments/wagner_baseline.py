"""G1: total-mixup baseline + wall-clock, via Wagner et al.'s own code.

For one (config, final-epoch) cell: per layer, per class pair, run the
vendored mixup-barcode pipeline (vendor/Mixup-SoCG26, see vendor/README.md)
on the SAME PCA-projected clouds the ECP measurement uses, in both inclusion
directions (the mixup barcode is asymmetric), degrees 0 and 1.  Records per
pair: total mixup, total persistence, total/mean mixup percentage per degree
and direction, plus wall-clock for (a) the full Wagner pipeline and (b) our
ECP statistic on identical clouds (profile only, no permutations -- the
apples-to-apples cost of one descriptor evaluation).

Outputs results/wagner/<name>_epoch<k>.json.  E5 consumes the per-layer
mean of the direction-averaged mean-mixup-percentage (degree 0+1) as the
"total mixup" baseline feature family; the E3 wall-clock table uses the
timing fields.

Usage:
  python -m experiments.wagner_baseline <config_index>
  python -m experiments.wagner_baseline --manifest   # rows for sbatch array
"""

import json
import os
import shutil
import sys
import tempfile
import time
import zlib
from itertools import combinations

import numpy as np

from .common import CONFIGS, PCA_DIM, POINTS_PER_CLASS, RESULTS_DIR, ROOT, dump_json
from .common import checkpoint_epochs
from .ecp import default_r_grid, mixup_ecp, profile_stat
from .extract import class_clouds, extract_checkpoint, load_features

VENDOR = os.path.join(ROOT, "vendor", "Mixup-SoCG26")
MAX_DIM = 1  # degrees 0 and 1, as in Wagner et al.'s disentanglement study


def _vendor_modules():
    if not os.path.isdir(VENDOR):
        sys.exit(f"vendor repo missing: {VENDOR} (see vendor/README.md)")
    if VENDOR not in sys.path:
        sys.path.insert(0, VENDOR)
    import matplotlib
    matplotlib.use("Agg")
    from analysis import pipeline_stats
    from analysis.barcode_utils import merge_standard_and_image_barcodes
    return pipeline_stats, merge_standard_and_image_barcodes


def _stats_to_dict(s):
    return {"total_mixup": float(s.total_mixup),
            "total_pers": float(s.total_pers),
            "total_mixup_perc": float(s.total_mixup_perc),
            "mean_mixup_perc": float(s.mean_mixup_perc)}


def wagner_pair(A, B, pipeline_stats, merge, seed):
    """One directed mixup-barcode evaluation A -> A u B, degrees 0..MAX_DIM."""
    np.random.seed(seed)  # their tie-breaking perturbation draws from global RNG
    try:
        res_A, res_im = pipeline_stats.compute_barcodes_from_ripser(A, B, MAX_DIM)
    except SystemExit as e:
        # their run_subprocess() calls bare sys.exit() on ripser failure,
        # which exits 0 and would let Slurm mark a dead task COMPLETED
        raise RuntimeError("vendored ripser invocation failed "
                           "(see their logged stderr above)") from e
    if len(res_A) < MAX_DIM + 1 or len(res_im) < MAX_DIM + 1:
        raise RuntimeError(f"ripser output parsed to {len(res_A)}/{len(res_im)} "
                           f"dims, expected {MAX_DIM + 1}: output format mismatch")
    out = {}
    for d in range(MAX_DIM + 1):
        matched = merge(res_A[d], res_im[d], d)
        s = pipeline_stats.get_persistence_stats(matched)
        if not np.isfinite([s.total_mixup, s.total_pers]).all():
            raise RuntimeError(f"non-finite mixup stats at dim {d}")
        out[f"dim{d}"] = _stats_to_dict(s)
    return out


def measure_cell(cfg, epoch):
    extract_checkpoint(cfg, epoch)
    feats, labels = load_features(cfg, epoch)
    classes = sorted(np.unique(labels).tolist())
    counts = {c: int(np.sum(labels == c)) for c in classes}
    m = min(POINTS_PER_CLASS, min(counts.values()))

    pipeline_stats, merge = _vendor_modules()

    # their pipeline is cwd-relative (./data, ./bin); isolate per task
    workdir = tempfile.mkdtemp(prefix="wagner_")
    os.symlink(os.path.join(VENDOR, "bin"), os.path.join(workdir, "bin"))
    home = os.getcwd()
    os.chdir(workdir)
    from analysis import library as _wlib
    _wlib.create_delete_data_folder()

    out = {"name": cfg.name, "epoch": epoch, "m": m, "d0": PCA_DIM,
           "layers": {}}
    try:
        for layer, F in feats.items():
            rec = {"pairs": {}}
            for c1, c2 in combinations(classes, 2):
                X, Y = class_clouds(F, labels, [c1, c2], m=m, d0=PCA_DIM)
                seed = zlib.crc32(f"wag:{cfg.name}:{epoch}:{layer}:{c1},{c2}".encode())

                t0 = time.perf_counter()
                fwd = wagner_pair(X, Y, pipeline_stats, merge, seed)
                bwd = wagner_pair(Y, X, pipeline_stats, merge, seed + 1)
                t_wagner = time.perf_counter() - t0

                t0 = time.perf_counter()
                r_grid = default_r_grid([X, Y])
                _ = profile_stat(mixup_ecp([X, Y], r_grid), r_grid, "int")
                t_ecp = time.perf_counter() - t0

                rec["pairs"][f"{c1},{c2}"] = {
                    "fwd": fwd, "bwd": bwd,
                    "t_wagner": t_wagner, "t_ecp": t_ecp,
                }
            sym = [np.mean([p["fwd"][f"dim{d}"]["mean_mixup_perc"],
                            p["bwd"][f"dim{d}"]["mean_mixup_perc"]])
                   for p in rec["pairs"].values() for d in range(MAX_DIM + 1)]
            rec["mean_mixup_perc_sym"] = float(np.mean(sym))
            rec["t_wagner_total"] = float(sum(p["t_wagner"] for p in rec["pairs"].values()))
            rec["t_ecp_total"] = float(sum(p["t_ecp"] for p in rec["pairs"].values()))
            out["layers"][layer] = rec
            print(f"[{cfg.name} ep{epoch}] wagner {layer} done "
                  f"(wagner {rec['t_wagner_total']:.1f}s vs ecp {rec['t_ecp_total']:.1f}s)",
                  flush=True)
    finally:
        os.chdir(home)
        shutil.rmtree(workdir, ignore_errors=True)

    dump_json(out, os.path.join(RESULTS_DIR, "wagner",
                                f"{cfg.name}_epoch{epoch}.json"))
    return out


def manifest():
    """Final-epoch cells: the E5 population plus the E2/E3 models (timing)."""
    rows = []
    for i, cfg in enumerate(CONFIGS):
        if cfg.tag in ("pop", "e2", "e3"):
            rows.append((i, checkpoint_epochs(cfg)[-1]))
    return rows


if __name__ == "__main__":
    if sys.argv[1] == "--manifest":
        for i, e in manifest():
            print(i, e)
        sys.exit(0)
    cfg = CONFIGS[int(sys.argv[1])]
    measure_cell(cfg, checkpoint_epochs(cfg)[-1])

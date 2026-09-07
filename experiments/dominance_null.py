"""What does the dominance ratio look like when there is NOTHING to find?

The dominance score compares one triple quotient against the MAX of three
pairwise quotients. Under exchangeable labels every quotient has mean 1 with
permutation noise, so the max of three is biased upward and the ratio is
biased below 1: a fully null configuration will register as "dominated" a
large fraction of the time. E10's 120/120 at initialization sits close to that
regime (pair quotients ~0.74), so the number that matters is how far the init
and trained ratio distributions sit from the null one.

Two nulls, same protocol constants as E10 (d0=5, B=49):
  gauss  -- four clouds from one isotropic Gaussian in R^5 (pure geometry)
  shuffle -- the init penultimate features with labels permuted (real
             density, no class signal)
m=100 per class keeps the fully-overlapping Alpha complexes tractable; the
bias under study depends on B, not m.
"""
import json
import os
import sys
from itertools import combinations
from multiprocessing import get_context

import numpy as np

from .common import RESULTS_DIR, dump_json
from .extract import class_clouds
from .init_dominance import FEAT_DIR, summarize

N_JOBS = int(os.environ.get("DISENTANGLE_JOBS", "24"))


def _job(args):
    from .measure import measure_pair
    key, clouds, B, seed = args
    return key, measure_pair(clouds, B, np.random.default_rng(seed))


def run(kind, seed, m=100, d0=5, B=49, k=4):
    rng = np.random.default_rng(1000 + seed)
    if kind == "gauss":
        F = rng.normal(size=(k * 400, d0)); labels = np.repeat(np.arange(k), 400)
    else:
        z = np.load(os.path.join(FEAT_DIR, "e3_resnet20w1_cifar10_wd0.0005_aug_s0_init.npz"))
        F = z["penult"]; labels = rng.permutation(z["labels"])
    classes = list(range(k))
    jobs = [(f"s{seed}:{a},{b}", class_clouds(F, labels, [a, b], m=m, d0=d0), B, seed * 100 + a * 10 + b)
            for a, b in combinations(classes, 2)]
    jobs += [(f"s{seed}:" + ",".join(map(str, t)), class_clouds(F, labels, list(t), m=m, d0=d0), B,
              seed * 1000 + t[0] * 100 + t[1] * 10 + t[2]) for t in combinations(classes, 3)]
    return jobs


if __name__ == "__main__":
    seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    out = {}
    for kind in ("gauss", "shuffle"):
        jobs = [j for s in range(seeds) for j in run(kind, s)]
        with get_context("fork").Pool(min(N_JOBS, len(jobs))) as pool:
            res = dict(pool.imap_unordered(_job, jobs, chunksize=1))
        stats = []
        for s in range(seeds):
            pre = f"s{s}:"
            pairs = {k[len(pre):]: v for k, v in res.items() if k.startswith(pre) and k.count(",") == 1}
            trips = {k[len(pre):]: v for k, v in res.items() if k.startswith(pre) and k.count(",") == 2}
            stats.append(summarize(pairs, trips))
        out[kind] = stats
        print(kind, [(x["dominated"], x["n"], round(x["median_ratio"], 3), round(x["max_ratio"], 3)) for x in stats], flush=True)
    dump_json(out, os.path.join(RESULTS_DIR, "dominance_null.json"))
    print("NULL DONE")

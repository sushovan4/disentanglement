# Vendored reference implementations

## Wagner et al. mixup barcodes (G1)

The E5 total-mixup baseline and the E3 wall-clock table use the AUTHORS'
reference implementation, not a re-reading of it (PLAN.md G1). It is GPL-3.0
licensed, so it is cloned here at setup time rather than committed into this
repository:

```bash
cd vendor
git clone https://github.com/hubwag/Mixup-SoCG26
cd Mixup-SoCG26
# portability patch 1: modern GCC requires the explicit include
sed -i '1i #include <cstdint>' bin/ripser_hacked.cpp
# portability patch 2: disable their debug dump of tst_matrix.raw into the
# repo root -- it memmap-writes one SHARED GPFS path from every Slurm task
# (SIGBUS / mmap-size races); it is only read by their Windows-only
# "alternative software" cross-check (USE_ALTERNATIVE_SOFTWARE = False)
sed -i 's|save_my_distance_matrix(D/2)|pass  # patched: GPFS-shared debug dump, races across Slurm tasks|' datagen/ripser_binary_io.py
cd bin
# static C++ runtime: `module load anaconda` puts an old libstdc++ on
# LD_LIBRARY_PATH at job runtime (GLIBCXX version mismatch otherwise)
c++ -std=c++11 ripser_hacked.cpp -o ripser_hacked -O3 -D NDEBUG \
    -static-libstdc++ -static-libgcc
```

(`ripser_hacked` is the image-persistence ripser variant with consistent
indexing that `analysis/library.py` invokes; the Makefile's `ripser-image`
target builds the unhacked variant and is not what the pipeline calls.)

Python deps beyond mixup-env: `matplotlib` (imported at module level by their
`analysis/library.py` even for non-plotting use):

```bash
~/.conda/envs/mixup-env/bin/python -m pip install matplotlib
```

The driver is `experiments/wagner_baseline.py`; it imports their
`analysis.pipeline_stats.compute_barcodes_from_ripser`,
`analysis.barcode_utils.merge_standard_and_image_barcodes`, and
`analysis.pipeline_stats.get_persistence_stats` unmodified, and runs each
measurement inside a throwaway working directory (their pipeline reads/writes
a cwd-relative `./data/` and `./bin/ripser_hacked`, which would collide across
Slurm array tasks otherwise).

License note: nothing from the GPL repo is copied into this repository or
linked into the paper's code; we shell out to their pipeline and record its
JSON-serializable outputs. Cite both the paper (wagner2024mixup) and the
repository URL in the acknowledgments/reproducibility statement.

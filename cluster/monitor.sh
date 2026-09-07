#!/bin/bash
# Quick progress summary for the disentangle suite.
set -uo pipefail
cd "$HOME/disentanglement"

echo "── queue ──────────────────────────────────────────────"
squeue -u "$USER" --format="%12i %14j %8T %10M %R" | head -30

echo
echo "── recent completions ────────────────────────────────"
sacct -X -S "$(date -d '2 days ago' +%F 2>/dev/null || date -v-2d +%F)" \
      --format=JobID%16,JobName%14,State%12,Elapsed,MaxRSS | tail -20

echo
echo "── results on disk ───────────────────────────────────"
for d in train measure; do
  n=$(ls results/$d/*.json 2>/dev/null | wc -l)
  echo "results/$d: $n files"
done
for f in e1_mnist e3_dynamics e4_calibration e5_predict; do
  [ -f "results/$f.json" ] && echo "results/$f.json: present"
done

echo
echo "── last errors (if any) ──────────────────────────────"
grep -l . cluster/logs/*.err 2>/dev/null | tail -5 | while read -r f; do
  echo "$f:"; tail -3 "$f"
done

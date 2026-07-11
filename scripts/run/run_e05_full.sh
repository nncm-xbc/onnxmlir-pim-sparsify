#!/usr/bin/env bash
# Full e05_stupidity_point rerun to 2160 steps (Task 2, 2026-06-12).
# The original partial run (1221 rows) is preserved in sparsified_partial_1221/.
set -u
cd /home/simon/repos/onnxmlir-pim-sparsify
PY=/home/simon/venv/general/bin/python3
LOG=artifacts/method_runs.log
A=artifacts/e05_stupidity_point
if [ -d "$A/sparsified" ] && [ ! -d "$A/sparsified_partial_1221" ]; then
  mv "$A/sparsified" "$A/sparsified_partial_1221"
fi
echo "[$(date '+%H:%M:%S')] START e05 full rerun (2160 steps)" >> "$LOG"
t0=$SECONDS
$PY -m sparsifier.sparsifier experiments/e05_stupidity_point.json > "$A/run_stdout.log" 2>&1
rc=$?
echo "[$(date '+%H:%M:%S')] END e05 rc=$rc elapsed=$((SECONDS-t0))s" >> "$LOG"

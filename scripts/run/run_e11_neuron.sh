#!/usr/bin/env bash
# driver4: after driver3 (e15/e16) completes, run e11_neuron_baseline (15 steps).
set -u
cd /home/simon/repos/onnxmlir-pim-sparsify
PY=/home/simon/venv/general/bin/python3
LOG=artifacts/method_runs.log
LOCK=artifacts/.sparsify_driver.lock
until grep -q 'DRIVER3 COMPLETE' "$LOG"; do sleep 30; done
while [ -e "$LOCK" ]; do sleep 5; done
while pgrep -f 'python3 -m sparsifier\.' > /dev/null; do sleep 10; done
trap 'rm -f "$LOCK"' EXIT
echo $$ > "$LOCK"
echo "[$(date '+%H:%M:%S')] START neuron_sparsifier e11_neuron_baseline (driver4)" >> "$LOG"
t0=$SECONDS
$PY -m sparsifier.neuron_sparsifier experiments/e11_neuron_baseline.json > artifacts/e11_neuron_baseline/run_stdout.log 2>&1
rc=$?
echo "[$(date '+%H:%M:%S')] END neuron_sparsifier rc=$rc elapsed=$((SECONDS-t0))s" >> "$LOG"
echo "[$(date '+%H:%M:%S')] DRIVER4 COMPLETE" >> "$LOG"

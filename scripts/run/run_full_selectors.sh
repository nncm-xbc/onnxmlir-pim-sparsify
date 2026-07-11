#!/usr/bin/env bash
# driver3: after driver2 (obd -> e05) completes, run e15_magnitude_full and
# e16_kwon_full sequentially (2160 steps each). Strictly serial — concurrent
# JAX processes crash with 'No BLAS support for stream'.
set -u
cd /home/simon/repos/onnxmlir-pim-sparsify
PY=/home/simon/venv/general/bin/python3
LOG=artifacts/method_runs.log
LOCK=artifacts/.sparsify_driver.lock

# wait for driver2 to finish and release the lock
until grep -q 'DRIVER2 COMPLETE' "$LOG"; do sleep 30; done
while [ -e "$LOCK" ]; do sleep 5; done
while pgrep -f 'python3 -m sparsifier\.' > /dev/null; do sleep 10; done
trap 'rm -f "$LOCK"' EXIT
echo $$ > "$LOCK"

run() {
  local mod=$1 cfg=$2
  echo "[$(date '+%H:%M:%S')] START $mod $cfg (driver3)" >> "$LOG"
  local t0=$SECONDS
  $PY -m sparsifier.$mod experiments/$cfg.json > artifacts/$cfg/run_stdout.log 2>&1
  local rc=$?
  echo "[$(date '+%H:%M:%S')] END $mod rc=$rc elapsed=$((SECONDS-t0))s" >> "$LOG"
}

run magnitude_sparsifier e15_magnitude_full
run kwon_sparsifier e16_kwon_full
echo "[$(date '+%H:%M:%S')] DRIVER3 COMPLETE" >> "$LOG"

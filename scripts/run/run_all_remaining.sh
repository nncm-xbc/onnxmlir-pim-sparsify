#!/usr/bin/env bash
# Sequential: remaining 4 method baselines, then full e05 rerun (2026-06-12).
# Detach-safe; guarded against concurrent sparsifier processes.
set -u
cd /home/simon/repos/onnxmlir-pim-sparsify
PY=/home/simon/venv/general/bin/python3
LOG=artifacts/method_runs.log
LOCK=artifacts/.sparsify_driver.lock
if [ -e "$LOCK" ]; then echo "[$(date '+%H:%M:%S')] DRIVER ABORT: lock exists" >> "$LOG"; exit 1; fi
trap 'rm -f "$LOCK"' EXIT
echo $$ > "$LOCK"

# wait for any foreign sparsifier process to finish
while pgrep -f 'python3 -m sparsifier\.' > /dev/null; do sleep 10; done

run() {
  local mod=$1 cfg=$2
  echo "[$(date '+%H:%M:%S')] START $mod $cfg (driver2)" >> "$LOG"
  local t0=$SECONDS
  $PY -m sparsifier.$mod experiments/$cfg.json > artifacts/$cfg/run_stdout.log 2>&1
  local rc=$?
  echo "[$(date '+%H:%M:%S')] END $mod rc=$rc elapsed=$((SECONDS-t0))s" >> "$LOG"
}

run kwon_sparsifier e12_kwon
run obs_sparsifier e14_obs
run lazarevich_sparsifier e08_omega_data
run obd_sparsifier e13_obd
echo "[$(date '+%H:%M:%S')] ALL METHOD RUNS DONE (driver2)" >> "$LOG"

# e05 full rerun; original partial already preserved at sparsified_partial_1221
run sparsifier e05_stupidity_point
echo "[$(date '+%H:%M:%S')] DRIVER2 COMPLETE" >> "$LOG"

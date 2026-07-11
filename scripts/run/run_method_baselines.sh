#!/usr/bin/env bash
# Sequential method-comparison baseline runs (Task 1, 2026-06-12)
set -u
cd /home/simon/repos/onnxmlir-pim-sparsify
PY=/home/simon/venv/general/bin/python3
LOG=artifacts/method_runs.log
run() {
  local mod=$1 cfg=$2
  echo "[$(date '+%H:%M:%S')] START $mod $cfg" >> "$LOG"
  local t0=$SECONDS
  $PY -m sparsifier.$mod experiments/$cfg.json > artifacts/$cfg/run_stdout.log 2>&1
  local rc=$?
  echo "[$(date '+%H:%M:%S')] END $mod rc=$rc elapsed=$((SECONDS-t0))s" >> "$LOG"
}
run magnitude_sparsifier e09_magnitude_search
run kwon_sparsifier e12_kwon
run obs_sparsifier e14_obs
run lazarevich_sparsifier e08_omega_data
run obd_sparsifier e13_obd
echo "[$(date '+%H:%M:%S')] ALL METHOD RUNS DONE" >> "$LOG"

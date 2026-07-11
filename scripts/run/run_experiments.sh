#!/usr/bin/env bash
# run_experiments.sh — run all pending experiments serially
#
# Skip logic:
#   - already trained  → artifacts/<name>/W_0.npy exists
#   - already sparsified → artifacts/<name>/sparsified/ exists
#
# Skipped entirely (need work before they can run):
#   e07_fullres_784    — requires data/X_train_full.csv (not generated yet)
#   e08_omega_data     — requires make_omega() to support omega_source='data'
#   e09_magnitude_search — requires prune() to support search='magnitude'
#   mnist_neuron       — config has no name set (skeleton only)

set -euo pipefail

REPO="$(cd "$(dirname "$0")" && pwd)"
LOG="$REPO/run_experiments.log"

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }
die() { log "FATAL: $*"; exit 1; }

cd "$REPO"
log "=== Experiment run started ==="
log "Repo: $REPO"
log "Log:  $LOG"

# ---------------------------------------------------------------------------
run_exp() {
    local cfg="$1"
    local name
    name=$(python3 -c "import json,sys; print(json.load(open('$cfg'))['name'])")
    [[ -z "$name" ]] && { log "SKIP $cfg — name is empty"; return; }

    log "--- $name ---"

    if [[ -f "artifacts/$name/W_0.npy" ]]; then
        log "  train: already done, skipping"
    else
        log "  train: starting"
        python3 scripts/train.py "$cfg" 2>&1 | tee -a "$LOG"
        log "  train: done"
    fi

    if [[ -d "artifacts/$name/sparsified" ]]; then
        log "  sparsify: already done, skipping"
    else
        log "  sparsify: starting"
        python3 sparsifier/sparsifier.py "$cfg" 2>&1 | tee -a "$LOG"
        log "  sparsify: done"
    fi
}
# ---------------------------------------------------------------------------

# E1 — adjust ablation (no adjust step)
run_exp experiments/e01_no_adjust.json

# E2 — omega sample size ablation
run_exp experiments/e02_omega_100.json
run_exp experiments/e02_omega_1000.json
run_exp experiments/e02_omega_100000.json

# E6 — activation function ablation
run_exp experiments/e06_act_tanh.json
run_exp experiments/e06_act_sigmoid.json

# E10 — seed variance (baseline=seed0 already done; seeds 1–4 here)
run_exp experiments/e10_seed_1.json
run_exp experiments/e10_seed_2.json
run_exp experiments/e10_seed_3.json
run_exp experiments/e10_seed_4.json

# Larger architecture
run_exp experiments/mnist_784_256_128_10.json

# Already completed — listed for reference, script will detect and skip:
#   baseline, e03_width_50, e03_width_200, e04_depth_2layer,
#   e04_depth_4layer, e05_stupidity_point

log "=== All done ==="

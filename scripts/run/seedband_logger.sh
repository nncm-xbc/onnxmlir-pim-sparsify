#!/usr/bin/env bash
# Appends a progress line per seed-band run to the experiment log every INTERVAL seconds,
# until the supervisor (queue_seedband) exits. Usage: seedband_logger.sh <log.md> [interval]
set -u
LOG=$1; INTERVAL=${2:-1800}
cd "$(dirname "$0")/../.."
line() {  # run csv
  [ -f "$2" ] || return
  tail -1 "$2" | awk -F, -v r="$1" '$1 ~ /^[0-9]+$/ {printf "| %s | %s | %.1f %% | %.3f | %s |\n", r, $1, $4*100, $5, $6}'
}
while :; do
  alive=$(pgrep -f 'supervise.py artifacts/seedband/queue_resume.txt' >/dev/null && echo 1 || echo 0)
  {
    echo ""
    echo "**$(date '+%Y-%m-%d %H:%M')** — supervisor $([ $alive = 1 ] && echo running || echo exited)"
    echo ""
    echo "| run | step | sparsity | acc | d_manifold |"
    echo "|---|---|---|---|---|"
    for s in 1 2 3 4; do line e25_seed_$s artifacts/e25_stupidity_seed_$s/sparsified/sparsification_log.csv; done
    for s in 1 2; do line e34_seed_$s artifacts/e34_obd_full_seed_$s/obd_sparsified/obd_sparsification_log.csv; done
    echo ""
    echo '```'
    grep -E "launch|DONE|EXIT|STALL|GAVE UP|QUEUE" artifacts/seedband/supervise.log 2>/dev/null | tail -4
    echo '```'
  } >> "$LOG"
  [ $alive = 1 ] || break
  sleep "$INTERVAL"
done

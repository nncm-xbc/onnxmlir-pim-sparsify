#!/usr/bin/env bash
# Appends a progress table (one row per queued run) to the experiment log every INTERVAL seconds
# while the supervisor in PIDFILE is alive, plus a final entry when it exits.
# Usage: seedband_logger.sh <log.md> <queue.txt> <pidfile> <supervise.log> [interval]
set -u
LOG=$1; QUEUE=$2; PIDFILE=$3; SUPLOG=$4; INTERVAL=${5:-1800}
cd "$(dirname "$0")/../.."
while :; do
  kill -0 "$(cat "$PIDFILE")" 2>/dev/null && alive=1 || alive=0
  {
    echo ""
    echo "**$(date '+%Y-%m-%d %H:%M')** — supervisor $([ $alive = 1 ] && echo running || echo exited)"
    echo ""
    echo "| run | step | sparsity | acc | d_manifold |"
    echo "|---|---|---|---|---|"
    for cfg in $(grep -v '^#' "$QUEUE" | grep -o 'experiments/[^ ]*\.json' | sort -u); do
      name=$(basename "$cfg" .json)
      csv=$(ls artifacts/$name/*sparsified/*sparsification_log.csv 2>/dev/null | head -1)
      [ -n "$csv" ] && tail -1 "$csv" | awk -F, -v r="$name" '$1 ~ /^[0-9]+$/ {printf "| %s | %s | %.1f %% | %.3f | %s |\n", r, $1, $4*100, $5, $6}'
    done
    echo ""
    echo '```'
    grep -E "launch|DONE|EXIT|STALL|GAVE UP|QUEUE" "$SUPLOG" 2>/dev/null | tail -3 | cut -c1-200
    echo '```'
  } >> "$LOG"
  [ $alive = 1 ] || break
  sleep "$INTERVAL"
done

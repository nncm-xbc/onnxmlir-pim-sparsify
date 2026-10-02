#!/usr/bin/env bash
# Idempotent (re)launch of a seed-band queue: skips train lines whose dense net already exists
# (retraining mid-run would replace the reference network), starts the supervisor (runs resume
# from their latest checkpoint) and the progress logger. Liveness is tracked by a pidfile, not by
# pgrep patterns (a pattern also matches any shell whose command line contains it).
# Usage: seedband_launch.sh <queue.txt> [log.md]     Safe to call at @reboot.
set -u
cd "$(dirname "$0")/../.."
QUEUE=$1; LOG=${2:-docs/experiments/2026-09-seedband.md}
TAG=$(basename "$QUEUE" .txt); DIR=artifacts/seedband; PID=$DIR/$TAG.pid
mkdir -p $DIR
[ -f $PID ] && kill -0 "$(cat $PID)" 2>/dev/null && exit 0
: > $DIR/$TAG.resume.txt
while IFS= read -r ln; do
  case "$ln" in
    ''|'#'*) continue ;;
    *scripts/train.py*)
      name=$(python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['name'])" "${ln##* }")
      [ -f "artifacts/$name/W_0.npy" ] && continue ;;
    *zero_cost_fraction*)
      name=$(python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['name'])" "${ln##* }")
      [ -f "artifacts/$name/zero_cost.json" ] && continue ;;
    *sparsifier*)  # skip runs whose log (for THIS selector's subdir) already holds every step
      cfg=${ln##* }
      read -r name steps < <(python3 -c "import json,sys;c=json.load(open(sys.argv[1]));print(c['name'],c['sparsify']['steps'])" "$cfg")
      mod=$(echo "$ln" | grep -o 'sparsifier\.[a-z_]*' | head -1)
      case "$mod" in sparsifier.sparsifier) sub=sparsified ;; *) sub=${mod#sparsifier.}; sub=${sub%_sparsifier}_sparsified ;; esac
      csv=$(ls artifacts/$name/$sub/*sparsification_log.csv 2>/dev/null | head -1)
      [ -n "$csv" ] && [ $(( $(wc -l < "$csv") - 1 )) -ge "$steps" ] && continue ;;
  esac
  echo "$ln" >> $DIR/$TAG.resume.txt
done < "$QUEUE"
printf '\n**%s** — (re)launch of `%s`: %s commands queued\n' "$(date '+%Y-%m-%d %H:%M')" "$QUEUE" "$(wc -l < $DIR/$TAG.resume.txt)" >> "$LOG"
export JAX_PLATFORMS=cuda RESTART_SLEEP=60 MAX_RESTARTS=10
setsid nohup /home/simon/venv/general/bin/python3 scripts/run/supervise.py $DIR/$TAG.resume.txt \
  >> $DIR/$TAG.supervise.log 2>&1 < /dev/null &
echo $! > $PID
# GPU telemetry every 10 s, to diagnose the unexplained host crashes (see experiment log).
setsid nohup nvidia-smi --query-gpu=timestamp,temperature.gpu,power.draw,utilization.gpu,clocks.sm,clocks_throttle_reasons.active \
  --format=csv,noheader -l 10 >> $DIR/gpu_telemetry.csv 2>&1 < /dev/null &
BPID=$DIR/board_telemetry.pid
if ! { [ -f $BPID ] && kill -0 "$(cat $BPID)" 2>/dev/null; }; then
  setsid nohup scripts/run/board_telemetry.sh $DIR/board_telemetry.csv > /dev/null 2>&1 < /dev/null &
  echo $! > $BPID
fi
setsid nohup scripts/run/seedband_logger.sh "$LOG" $DIR/$TAG.resume.txt $PID $DIR/$TAG.supervise.log 1800 > /dev/null 2>&1 < /dev/null &

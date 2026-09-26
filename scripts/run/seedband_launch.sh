#!/usr/bin/env bash
# Idempotent (re)launch of the D2 seed-band queue: skips train lines whose dense net already
# exists (retraining mid-run would replace the reference network), then starts the supervisor
# (runs resume from their latest checkpoint) and the progress logger. Safe to call at @reboot.
set -u
cd "$(dirname "$0")/../.."
pgrep -f 'supervise.py artifacts/seedband/queue_resume.txt' >/dev/null && exit 0
mkdir -p artifacts/seedband
: > artifacts/seedband/queue_resume.txt
while IFS= read -r ln; do
  case "$ln" in
    ''|'#'*) continue ;;
    *scripts/train.py*)
      name=$(python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['name'])" "${ln##* }")
      [ -f "artifacts/$name/W_0.npy" ] && continue ;;
  esac
  echo "$ln" >> artifacts/seedband/queue_resume.txt
done < scripts/run/queue_seedband.txt
LOG=docs/experiments/2026-09-seedband.md
printf '\n**%s** — (re)launch: %s commands queued\n' "$(date '+%Y-%m-%d %H:%M')" "$(wc -l < artifacts/seedband/queue_resume.txt)" >> $LOG
export JAX_PLATFORMS=cuda
setsid nohup /home/simon/venv/general/bin/python3 scripts/run/supervise.py artifacts/seedband/queue_resume.txt \
  >> artifacts/seedband/supervise.log 2>&1 < /dev/null &
sleep 5
sed -i "s|supervise.py scripts/run/queue_seedband.txt|supervise.py artifacts/seedband/queue_resume.txt|" scripts/run/seedband_logger.sh
setsid nohup scripts/run/seedband_logger.sh $LOG 1800 > /dev/null 2>&1 < /dev/null &

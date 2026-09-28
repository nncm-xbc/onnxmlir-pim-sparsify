#!/usr/bin/env bash
# Every 10 s: timestamp, +12V / +5V rails, CPU core V, CPU/VRM/chipset temp, VRM current (ASUS WMI).
# Written to diagnose unexplained hard crashes under GPU load (+12V read 10.08 V at idle, 2026-09-28).
OUT=${1:-artifacts/seedband/board_telemetry.csv}
cd "$(dirname "$0")/../.."
[ -s "$OUT" ] || echo "time,v12,v5,vcore,cpu_c,vrm_c,chipset_c,vrm_a" > "$OUT"
while :; do
  sensors -u asus_wmi_sensors-virtual-0 2>/dev/null | awk -v t="$(date '+%F %T')" '
    /^\+12V Voltage:/{k="v12"} /^\+5V Voltage:/{k="v5"} /^CPU Core Voltage:/{if(!("vc" in s))k="vc"}
    /^CPU Temperature:/{k="ct"} /^CPU VRM Temperature:/{k="vt"} /^Chipset Temperature:/{k="cs"} /^CPU VRM Output Current:/{k="va"}
    /_input:/{ if(k!="" && !(k in s)) s[k]=$2; k="" }
    END{printf "%s,%s,%s,%s,%s,%s,%s,%s\n", t, s["v12"], s["v5"], s["vc"], s["ct"], s["vt"], s["cs"], s["va"]}' >> "$OUT"
  sync "$OUT" 2>/dev/null
  sleep 10
done

#!/usr/bin/env bash
# Confirm on REAL Pi hardware the relaxed/p-min configs the x86 sweep predicted. Cool-start before each
# bench (pre-throttle numbers; user has a tower cooler). Reconciles the prior session's 13.5 claim.
set -uo pipefail
PI=mklab@192.168.0.6
cd /home/mkt2126/cpullm
pitemp(){ timeout 8 ssh $PI 'cat /sys/class/thermal/thermal_zone0/temp' 2>/dev/null | awk '{printf "%.1f",$1/1000}'; }
bench(){ HOST=192.168.0.6 python3 scripts/pi_bench.py 8080 --suite general --n 96 2>/dev/null | grep -E 'general'; }
cool(){ for i in $(seq 1 40); do t=$(pitemp); awk "BEGIN{exit !($t<66)}" 2>/dev/null && { echo "  [cool ${t}C]"; return; }; echo "  [cooling ${t}C]"; sleep 8; done; }
launch_cfg(){ # pratio pmin
  local pr=$1 pm=$2
  timeout 15 ssh $PI 'pkill -x llama-server 2>/dev/null; sleep 2' >/dev/null 2>&1
  timeout 20 ssh $PI "cd ~/cpullm && setsid env RELAX_PRATIO=$pr RELAX_TOPK=8 PMIN=$pm bash scripts/run_pi5.sh launch </dev/null >/tmp/pi_srv.log 2>&1 & echo launched" >/dev/null 2>&1
  for i in $(seq 1 90); do timeout 6 ssh $PI 'curl -sf http://127.0.0.1:8080/health >/dev/null 2>&1' && return 0; sleep 2; done
  echo "  !! health timeout"; return 1
}
# config list:  pratio:pmin   (lossless, old-13.5-claim, best-predicted, mid)
for cfg in "0:0.5" "0.3:0.5" "0.15:0.5" "0.08:0.2" "0.08:0.1"; do
  pr=${cfg%%:*}; pm=${cfg##*:}
  echo "=== pratio=$pr pmin=$pm ==="
  cool
  if launch_cfg "$pr" "$pm"; then
    echo -n "  "; bench
    echo "  post-temp=$(pitemp)C"
  fi
done
echo "=== done ==="

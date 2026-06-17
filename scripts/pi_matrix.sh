#!/usr/bin/env bash
# Full Pi workload matrix: {lossless, light-relaxed, max-relaxed, max-relaxed+ngram} x {general,repetition,literal}.
# Determines whether an honestly-labeled structured/code mode can reach 15 tok/s where general cannot.
set -uo pipefail
PI=mklab@192.168.0.6
cd /home/mkt2126/cpullm
pitemp(){ timeout 8 ssh $PI 'cat /sys/class/thermal/thermal_zone0/temp' 2>/dev/null | awk '{printf "%.1f",$1/1000}'; }
cool(){ for i in $(seq 1 50); do t=$(pitemp); awk "BEGIN{exit !($t<66)}" 2>/dev/null && { echo "  [cool ${t}C]"; return; }; sleep 8; done; }
launch_cfg(){ # pratio pmin ngram
  local pr=$1 pm=$2 ng=$3
  timeout 15 ssh $PI 'pkill -x llama-server 2>/dev/null; sleep 2' >/dev/null 2>&1
  timeout 20 ssh $PI "cd ~/cpullm && setsid env RELAX_PRATIO=$pr RELAX_TOPK=8 PMIN=$pm NGRAM=$ng NGRAM_N=8 bash scripts/run_pi5.sh launch </dev/null >/tmp/pi_srv.log 2>&1 & echo launched" >/dev/null 2>&1
  for i in $(seq 1 90); do timeout 6 ssh $PI 'curl -sf http://127.0.0.1:8080/health >/dev/null 2>&1' && return 0; sleep 2; done
  echo "  !! health timeout"; return 1
}
#       label              pratio pmin ngram
for cfg in "lossless:0:0.5:0" "relax15:0.15:0.5:0" "relaxmax:0.08:0.1:0" "relaxmax+ngram:0.08:0.1:1"; do
  IFS=: read lbl pr pm ng <<<"$cfg"
  echo "=== $lbl (pratio=$pr pmin=$pm ngram=$ng) ==="
  cool
  if launch_cfg "$pr" "$pm" "$ng"; then
    HOST=192.168.0.6 python3 scripts/pi_bench.py 8080 --suite all --n 96 2>/dev/null | grep -E 'general|repetition|literal'
    echo "  post-temp=$(pitemp)C"
  fi
done
echo "=== done ==="

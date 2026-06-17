#!/usr/bin/env bash
# Sweep threadpool --poll (busy-wait vs futex-block between ops). LOSSLESS (poll never changes output).
# The profile showed heavy __futex_wait between many small MatFormer ops; full polling (100) may cut that.
set -uo pipefail
PI=mklab@192.168.0.6
cd /home/mkt2126/cpullm
pitemp(){ timeout 8 ssh $PI 'cat /sys/class/thermal/thermal_zone0/temp' 2>/dev/null | awk '{printf "%.1f",$1/1000}'; }
cool(){ for i in $(seq 1 50); do t=$(pitemp); awk "BEGIN{exit !($t<66)}" 2>/dev/null && { echo "  [cool ${t}C]"; return; }; sleep 8; done; }
launch(){ # poll
  local pl=$1
  timeout 15 ssh $PI 'pkill -x llama-server 2>/dev/null; sleep 2' >/dev/null 2>&1
  timeout 22 ssh $PI "cd ~/cpullm && export LD_LIBRARY_PATH=\$HOME/cpullm/bin && setsid taskset -c 0-3 bin/llama-server \
    -m models/gemma-4-E2B-it-Q4_0.gguf -md models/mtp-gemma-4-E2B-it.gguf --spec-type draft-mtp \
    --spec-draft-n-max 3 --spec-draft-p-min 0.5 -t 4 -tb 4 --poll $pl -c 4096 --mlock --no-warmup \
    --host 0.0.0.0 --port 8080 </dev/null >/tmp/pi_srv.log 2>&1 & echo ok" >/dev/null 2>&1
  for i in $(seq 1 90); do timeout 6 ssh $PI 'curl -sf http://127.0.0.1:8080/health >/dev/null 2>&1' && return 0; sleep 2; done
  echo "  !! health timeout"; return 1
}
for pl in 50 100 0; do
  echo "=== --poll $pl ==="
  cool
  if launch "$pl"; then
    HOST=192.168.0.6 python3 scripts/pi_bench.py 8080 --suite general --n 96 2>/dev/null | grep general
  fi
done
echo "=== done ==="

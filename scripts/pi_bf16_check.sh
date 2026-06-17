#!/usr/bin/env bash
# Restart new binary (bf16 NEON kernel) in quality mode, cool-start bench general (tok/s gain), then capture
# outputs for the old-vs-new distribution-exactness diff.
set -uo pipefail
PI=mklab@192.168.0.6
cd /home/mkt2126/cpullm
pitemp(){ timeout 8 ssh $PI 'cat /sys/class/thermal/thermal_zone0/temp' 2>/dev/null | awk '{printf "%.1f",$1/1000}'; }
echo "=== restart new binary, quality mode ==="
timeout 15 ssh $PI 'pkill -x llama-server 2>/dev/null; sleep 2' >/dev/null 2>&1
timeout 22 ssh $PI 'cd ~/cpullm && setsid bash scripts/run_pi5.sh launch </dev/null >/tmp/pi_srv.log 2>&1 & echo ok' >/dev/null 2>&1
for i in $(seq 1 90); do timeout 6 ssh $PI 'curl -sf http://127.0.0.1:8080/health >/dev/null 2>&1' && break; sleep 2; done
echo "  banner: $(timeout 6 ssh $PI 'grep -h "\[launch\]" /tmp/pi_srv.log | tail -1')"
echo "=== cool to <66C ==="
for i in $(seq 1 50); do t=$(pitemp); awk "BEGIN{exit !($t<66)}" 2>/dev/null && { echo "  [cool ${t}C]"; break; }; sleep 8; done
echo "=== NEW binary tok/s (cool start) ==="
HOST=192.168.0.6 python3 scripts/pi_bench.py 8080 --suite general --n 96 2>/dev/null | grep general
echo "=== capture NEW outputs + diff vs OLD ==="
python3 scripts/cmp_quality.py capture 192.168.0.6 8080 /tmp/q_new.json >/dev/null
python3 scripts/cmp_quality.py diff /tmp/q_old.json /tmp/q_new.json
echo "=== done ==="

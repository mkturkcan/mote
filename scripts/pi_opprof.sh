#!/usr/bin/env bash
# Deep per-op-type profile of a quality-mode decode using the built-in GGML_OP_PROFILE instrumentation.
# Launches with profiling on, runs a sustained decode, then SIGINTs the server (graceful -> atexit dump).
set -uo pipefail
PI=mklab@192.168.0.6
timeout 20 ssh $PI 'pkill -x llama-server 2>/dev/null; sleep 2; fuser -k 8080/tcp 2>/dev/null; sleep 2; pgrep -x llama-server && pkill -9 -x llama-server; sleep 1; echo "port free: $(ss -ltn | grep -c :8080)"' 2>&1
echo "=== launch with GGML_OP_PROFILE=1 (skip 40 warmup graph-computes) ==="
timeout 22 ssh $PI 'cd ~/cpullm && setsid env GGML_OP_PROFILE=1 GGML_OP_PROFILE_SKIP=40 bash scripts/run_pi5.sh launch </dev/null >/tmp/pi_opprof.log 2>&1 & echo ok' >/dev/null 2>&1
for i in $(seq 1 90); do timeout 6 ssh $PI 'curl -sf http://127.0.0.1:8080/health >/dev/null 2>&1' && break; sleep 2; done
echo "=== sustained decode (n_predict=256) ==="
timeout 90 ssh $PI 'curl -s -m 80 http://127.0.0.1:8080/completion -H "Content-Type: application/json" -d "{\"prompt\":\"Explain in detail how a modern CPU pipeline works, including fetch, decode, execute, and retire stages, branch prediction, and out-of-order execution.\",\"n_predict\":256,\"temperature\":0,\"cache_prompt\":false}" >/dev/null 2>&1; echo "decode done"'
echo "=== SIGINT server -> atexit dumps op profile ==="
timeout 12 ssh $PI 'kill -INT $(pgrep -x llama-server) 2>/dev/null; sleep 4; echo signaled'
echo "=== OP PROFILE ==="
timeout 10 ssh $PI 'sed -n "/GGML OP PROFILE/,/^$/p" /tmp/pi_opprof.log 2>/dev/null | head -40'

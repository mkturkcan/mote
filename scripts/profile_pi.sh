#!/usr/bin/env bash
# Poor-man's sampling profiler for the running llama-server (no perf on the Pi). Kicks a sustained decode,
# then repeatedly attaches gdb and records every thread's top frame. The histogram of top-of-stack functions
# approximates where wall-clock goes across the 4 compute threads. Frames in the q4_0 matmul = BW-bound floor;
# everything else = lossless-addressable overhead.
set -uo pipefail
PID=$(pgrep -x llama-server | head -1)
N=${1:-120}
[ -z "$PID" ] && { echo "no llama-server"; exit 1; }
echo "profiling pid=$PID, $N samples"
# sustained load: long generation to localhost (detached so it keeps the server busy through sampling)
setsid bash -c 'curl -s -m 240 http://127.0.0.1:8080/completion -H "Content-Type: application/json" \
  -d "{\"prompt\":\"Write a long detailed essay about the history and culture of the Roman Empire, covering its founding, expansion, key emperors, daily life, engineering, and eventual decline.\",\"n_predict\":4096,\"temperature\":0,\"cache_prompt\":false}" >/tmp/profload.json 2>/dev/null' </dev/null >/dev/null 2>&1 &
sleep 3   # let generation reach steady state
for i in $(seq 1 "$N"); do
  gdb -p "$PID" -batch -ex 'set pagination off' -ex 'thread apply all bt 1' 2>/dev/null \
    | grep -E '^#0' | sed -E 's/^#0 +0x[0-9a-f]+ in +//; s/ *\(.*$//'
done | sort | uniq -c | sort -rn | head -35

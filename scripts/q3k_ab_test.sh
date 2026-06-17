#!/usr/bin/env bash
# A/B losslessness test for the q3_K repacked kernel (non-interactive llama-completion).
# ON  = repack active (new gemv/gemm).  OFF = GGML_NO_Q3K_REPACK=1 -> standard ggml_vec_dot_q3_K.
# Greedy (temp 0, fixed seed) => identical token text proves the kernel makes bit-identical decisions.
set -uo pipefail
cd /home/mkt2126/cpullm
BIN=third_party/llama.cpp/build/bin/llama-completion
M=models/gemma-4-E2B-it-Q3_K_M.gguf
PROMPT="Explain why the sky is blue, then list three prime numbers."
N=40
OUT=/tmp/q3k_ab
mkdir -p $OUT

run() { # $1=label  $2=env
    env $2 taskset -c 0-7 $BIN -m "$M" -p "$PROMPT" -n $N -t 8 --temp 0 -s 1 \
        -no-cnv --no-warmup 2>"$OUT/$1.err" 1>"$OUT/$1.out"
    echo "[$1] exit=$?"
}

echo "=== ON (repack) ==="  ; run on  ""
echo "=== OFF (no repack) ==="; run off "GGML_NO_Q3K_REPACK=1"

echo "=== generated text ON ===" ; cat "$OUT/on.out"
echo "=== generated text OFF ==="; cat "$OUT/off.out"
echo "=== DIFF (empty = bit-identical decisions = LOSSLESS) ==="
if diff -q "$OUT/on.out" "$OUT/off.out" >/dev/null; then
    echo "RESULT: IDENTICAL — q3_K repack is lossless vs standard vec_dot"
else
    echo "RESULT: DIFFERENT:"; diff "$OUT/on.out" "$OUT/off.out"
fi
echo "=== decode tok/s (ON vs OFF) ==="
grep -iE 'eval time|tokens per second|t/s' "$OUT/on.err"  | tail -2 | sed 's/^/ ON  /'
grep -iE 'eval time|tokens per second|t/s' "$OUT/off.err" | tail -2 | sed 's/^/ OFF /'

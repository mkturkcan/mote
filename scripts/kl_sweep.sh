#!/bin/bash
set -u
WIKI=/tmp/wikitext-2-raw/wiki.test.raw
B=/home/mkt2126/cpullm/third_party/llama.cpp/build/bin/llama-perplexity
cd /home/mkt2126/cpullm
BASE=/tmp/kl_base_q6k.bin
CH=12
echo "=== [1] generating reference logits from Q6_K (chunks=$CH) ==="
timeout 1800 $B -m models/gemma-4-E2B-it-Q6_K.gguf -f $WIKI -t 12 -c 512 --chunks $CH \
   --kl-divergence-base $BASE 2>&1 | grep -iE "save|logits|estimate|tokens" | tail -3
echo "base size: $(du -h $BASE 2>/dev/null | cut -f1)"
for q in Q4_0 Q3_K_M UD-IQ2_M; do
  echo "===== KL: $q vs Q6_K ====="
  timeout 1800 $B -m models/gemma-4-E2B-it-$q.gguf -f $WIKI -t 12 -c 512 --chunks $CH \
     --kl-divergence --kl-divergence-base $BASE 2>&1 \
     | grep -iE "Mean KL|Maximum KL|99.0%|Median KL|Mean Top|same top|RMS|token probability" | head -12
done
echo "KLDONE"

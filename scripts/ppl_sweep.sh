#!/bin/bash
WIKI=/tmp/wikitext-2-raw/wiki.test.raw
B=/home/mkt2126/cpullm/third_party/llama.cpp/build/bin/llama-perplexity
cd /home/mkt2126/cpullm
for q in Q6_K Q4_0 Q3_K_M UD-IQ2_M; do
  M=models/gemma-4-E2B-it-$q.gguf
  echo "===== $q ====="
  timeout 1200 $B -m $M -f $WIKI -t 8 -c 512 --chunks 40 2>&1 | grep -iE "^\[|Final estimate|PPL|perplexity:" | tail -3
done
echo "PPLDONE"

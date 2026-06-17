#!/usr/bin/env bash
# x86 (dev-box) launcher for ONE relaxed/lossless MTP server — used to explore the relaxed-acceptance
# grid OFF the Pi. tok/pass (acceptance) and similarity-to-greedy are HARDWARE-INDEPENDENT (pure model
# math at temp=0), so the whole quality+multiplier sweep runs here on the 32-core box; only wall-clock
# tok/s must be confirmed on the Pi. Mirrors run_pi5.sh launch() flags minus the Pi-only bits
# (no taskset/--mlock, host 127.0.0.1).
#   x86_serve.sh <port> <pratio> <topk> [nmax] [pmin]   (pratio=0 -> lossless reference)
set -euo pipefail
PORT="$1"; PRATIO="${2:-0}"; TOPK="${3:-8}"; NMAX="${4:-3}"; PMIN="${5:-0.5}"
ROOT=/home/mkt2126/cpullm
BIN="$ROOT/third_party/llama.cpp/build/bin/llama-server"
MODELS="$ROOT/models"
TARGET="$MODELS/gemma-4-E2B-it-Q4_0.gguf"
DRAFT="$MODELS/mtp-gemma-4-E2B-it.gguf"
export LD_LIBRARY_PATH="$(dirname "$BIN")${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
if awk "BEGIN{exit !($PRATIO>0)}"; then
  export LLAMA_SPEC_ACCEPT_PRATIO="$PRATIO" LLAMA_SPEC_ACCEPT_TOPK="$TOPK"
fi
exec "$BIN" -m "$TARGET" -md "$DRAFT" --spec-type draft-mtp \
  --spec-draft-n-max "$NMAX" --spec-draft-p-min "$PMIN" \
  -t 4 -tb 4 -c 4096 --no-warmup \
  --host 127.0.0.1 --port "$PORT" >/tmp/x86srv_$PORT.log 2>&1

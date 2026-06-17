#!/usr/bin/env bash
# Build the tuned A76 server and package a SELF-CONTAINED release tarball (binary + .so + launcher + gemma +
# bench). Attach the result to a GitHub release; a fresh Pi then installs in seconds (no native build) via
#   curl -fsSL <install.sh url> | CPULLM_PKG_URL=<this tarball url> bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="$ROOT/third_party/llama.cpp/build-a76/bin"
STAGE="$ROOT/dist/gemma-pi5"
OUT="$ROOT/dist/gemma-pi5.tar.gz"

echo "[release] building the A76 binary (KleidiAI + LTO) ..."
bash "$ROOT/scripts/pi_build.sh"
[ -x "$BIN/llama-server" ] || { echo "ERROR: build produced no llama-server"; exit 1; }

echo "[release] staging self-contained package ..."
rm -rf "$STAGE"; mkdir -p "$STAGE/bin" "$STAGE/scripts"
cp "$BIN/llama-server" "$BIN"/*.so* "$STAGE/bin/"
cp "$ROOT/scripts/run_pi5.sh" "$ROOT/scripts/gemma" "$ROOT/scripts/pi_bench.py" "$STAGE/scripts/"
chmod +x "$STAGE/scripts/gemma"

echo "[release] compressing ..."
tar -czf "$OUT" -C "$ROOT/dist" gemma-pi5

echo "[release] DONE -> $OUT  ($(du -h "$OUT" | cut -f1))"
echo "  1. create a GitHub release and attach this file as 'gemma-pi5.tar.gz'"
echo "  2. also attach scripts/install.sh so the one-liner can fetch it"
echo "  3. fresh-Pi install:"
echo "     curl -fsSL https://github.com/mkturkcan/mote/releases/latest/download/install.sh \\"
echo "       | CPULLM_PKG_URL=https://github.com/mkturkcan/mote/releases/latest/download/gemma-pi5.tar.gz bash"

#!/usr/bin/env bash
# ============================================================================
# One-command install for Gemma-4-E2B on a Raspberry Pi 5 (Raspberry Pi OS bookworm, 64-bit).
# Builds the tuned llama.cpp server (ARM KleidiAI matmul + LTO + the A76 kernels), downloads the model,
# installs the `gemma` command, and tunes the system. Safe to re-run (skips finished steps).
#
#   bash scripts/install.sh
#
# Optional: set MOTE_PKG_URL=<url-to-prebuilt-tarball> to skip the ~15 min native build.
# ============================================================================
set -euo pipefail

DEST="${MOTE_HOME:-$HOME/mote}"
JOBS="$(nproc)"
HF="https://huggingface.co/unsloth/gemma-4-E2B-it-GGUF/resolve/main"
TARGET_GGUF="gemma-4-E2B-it-Q4_0.gguf"        # 2.9 GB target
DRAFT_GGUF="mtp-gemma-4-E2B-it.gguf"          #  94 MB MTP draft head
# repo root, if this script is being run from a checkout (not needed for the prebuilt MOTE_PKG_URL path)
SRC="$(cd "$(dirname "$(readlink -f "$0" 2>/dev/null || echo /nonexistent)")/.." 2>/dev/null && pwd || echo /nonexistent)"

say(){ printf "\n\033[1;34m[install]\033[0m %s\n" "$*"; }
have(){ command -v "$1" >/dev/null 2>&1; }

mkdir -p "$DEST/bin" "$DEST/scripts" "$DEST/models"

say "1/5  dependencies (sudo apt) ..."
sudo apt-get update -qq
sudo apt-get install -y -qq build-essential cmake git curl libgomp1

say "2/5  the tuned server ..."
if [ -n "${MOTE_PKG_URL:-}" ]; then
  # prebuilt path — fully standalone (no repo needed): the tarball carries bin/ + scripts/
  echo "  downloading prebuilt package: $MOTE_PKG_URL"
  tmp="$(mktemp -d)"; curl -fL --retry 3 "$MOTE_PKG_URL" | tar -xz -C "$tmp"
  cp "$tmp"/*/bin/llama-server "$tmp"/*/bin/*.so* "$DEST/bin/"
  cp "$tmp"/*/scripts/* "$DEST/scripts/" 2>/dev/null || true
  rm -rf "$tmp"
else
  # source path — native build from the repo
  [ -d "$SRC/third_party/llama.cpp" ] || { echo "ERROR: run from the mote repo, or set MOTE_PKG_URL=<prebuilt tarball>"; exit 1; }
  echo "  building natively (KleidiAI + LTO, ~10-20 min on a Pi 5) ..."
  cmake -S "$SRC/third_party/llama.cpp" -B "$SRC/build-pi" \
    -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON \
    -DGGML_CPU_KLEIDIAI=ON -DGGML_LTO=ON \
    -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF >/dev/null
  cmake --build "$SRC/build-pi" --target llama-server -j"$JOBS"
  cp "$SRC/build-pi/bin/llama-server" "$SRC"/build-pi/bin/*.so* "$DEST/bin/"
  cp "$SRC/scripts/run_pi5.sh" "$SRC/scripts/pi_bench.py" "$SRC/scripts/gemma" "$DEST/scripts/"
fi
chmod +x "$DEST/scripts/gemma"

say "3/5  the model (~3 GB, resumable; skips if present) ..."
for f in "$TARGET_GGUF" "$DRAFT_GGUF"; do
  if [ -s "$DEST/models/$f" ]; then echo "  have $f"; else
    echo "  downloading $f ..."; curl -fL --retry 5 -C - -o "$DEST/models/$f" "$HF/$f"
  fi
done

say "4/5  the 'gemma' command ..."
mkdir -p "$HOME/.local/bin"
ln -sf "$DEST/scripts/gemma" "$HOME/.local/bin/gemma"
case ":$PATH:" in *":$HOME/.local/bin:"*) :;; *)
  grep -q '.local/bin' "$HOME/.bashrc" 2>/dev/null || echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$HOME/.bashrc"
  export PATH="$HOME/.local/bin:$PATH";; esac

say "5/5  system tuning (governor, hugepages) ..."
bash "$DEST/scripts/run_pi5.sh" tune || true

IP="$(hostname -I 2>/dev/null | awk '{print $1}')"
printf "\n\033[1;32m✓ done.\033[0m  Gemma-4-E2B is installed at %s\n\n" "$DEST"
cat <<EOF
  start it:    gemma start            # lossless ~11 tok/s   (also: gemma start fast | turbo)
  chat:        gemma chat             # in this terminal
  or browser:  http://$IP:8080        # built-in web chat UI
  speed test:  gemma bench

  (if 'gemma' isn't found yet:  source ~/.bashrc )
EOF

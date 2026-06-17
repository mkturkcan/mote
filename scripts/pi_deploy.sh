#!/usr/bin/env bash
# ============================================================================
# Deploy the cross-built Pi package + launcher + benchmark scripts to a Raspberry Pi 5
# over SSH. Run scripts/pi_build.sh first (to produce the aarch64 binary).
#
# Usage:
#   PI_HOST=pi@192.168.1.50 ./scripts/pi_deploy.sh            # binary + scripts only (fast, ~19 MB)
#   PI_HOST=pi@192.168.1.50 ./scripts/pi_deploy.sh --models   # also push *.gguf (slow, several GB; do once)
#   PI_HOST=... PI_DIR=/home/pi/cpullm ./scripts/pi_deploy.sh # override remote dir (default ~/cpullm)
#
# On-Pi layout it creates:   ~/cpullm/{bin,scripts,models}
# After deploy, on the Pi:   cd ~/cpullm/scripts && ./run_pi5.sh launch
# ============================================================================
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="$ROOT/third_party/llama.cpp/build-a76/bin"
PI_HOST="${PI_HOST:-${1:-}}"
PI_DIR="${PI_DIR:-cpullm}"   # relative to the Pi user's home
WITH_MODELS=0
for a in "$@"; do [ "$a" = "--models" ] && WITH_MODELS=1; done

[ -n "$PI_HOST" ] || { echo "ERROR: set PI_HOST=user@ip (e.g. pi@192.168.1.50)"; exit 1; }
[ -x "$BIN/llama-server" ] || { echo "ERROR: no aarch64 binary; run scripts/pi_build.sh first"; exit 1; }

echo "[deploy] target: $PI_HOST:~/$PI_DIR"
ssh "$PI_HOST" "mkdir -p ~/$PI_DIR/bin ~/$PI_DIR/scripts ~/$PI_DIR/models"

echo "[deploy] (1/3) binary + project .so  ($(du -shc "$BIN"/llama-server "$BIN"/*.so* | tail -1 | cut -f1))"
rsync -av --copy-links --info=progress2 \
  "$BIN/llama-server" "$BIN"/*.so* "$PI_HOST:$PI_DIR/bin/"

echo "[deploy] (2/3) launcher + benchmark scripts"
rsync -av \
  "$ROOT/scripts/run_pi5.sh" "$ROOT/scripts/pi_bench.py" \
  "$ROOT/scripts/lossless_gate.py" "$ROOT/scripts/tokens_per_pass.py" \
  "$PI_HOST:$PI_DIR/scripts/"
ssh "$PI_HOST" "chmod +x ~/$PI_DIR/scripts/run_pi5.sh"

if [ "$WITH_MODELS" = "1" ]; then
  echo "[deploy] (3/3) models (*.gguf) — large, resumable"
  rsync -av --partial --info=progress2 "$ROOT"/models/*.gguf "$PI_HOST:$PI_DIR/models/"
else
  echo "[deploy] (3/3) SKIP models (pass --models to push them, or scp the needed *.gguf once)"
fi

echo "[deploy] DONE. On the Pi:"
echo "    cd ~/$PI_DIR/scripts"
echo "    ./run_pi5.sh launch                 # MTP baseline"
echo "    TREE_K=2 ./run_pi5.sh launch        # + tree-draft"
echo "    NGRAM=1  ./run_pi5.sh launch        # + n-gram (repetition workloads)"
echo "  Then from THIS machine, A/B against the Pi:"
echo "    HOST=<pi-ip> python3 scripts/pi_bench.py 8080 --suite all --runs 3"

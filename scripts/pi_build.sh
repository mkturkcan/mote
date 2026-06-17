#!/usr/bin/env bash
# ============================================================================
# Cross-compile the (tree-draft + n-gram patched) llama-server for Raspberry Pi 5
# (Cortex-A76, aarch64) on THIS x86 host. Fast iteration: builds here, deploy with
# scripts/pi_deploy.sh. Produces a self-contained ~19 MB package (binary + project .so).
#
# Requires: aarch64-linux-gnu-g++/gcc (apt install g++-aarch64-linux-gnu), cmake.
# ABI: built with GCC 11 -> needs glibc<=2.36 / GLIBCXX<=3.4.30 == Raspberry Pi OS *bookworm*.
#      On an OLDER Pi OS (bullseye), build natively on the Pi instead.
# ============================================================================
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LLAMA="$ROOT/third_party/llama.cpp"
BUILD="$LLAMA/build-a76"
TOOLCHAIN="$ROOT/scripts/aarch64-a76.toolchain.cmake"
BIN="$BUILD/bin"

command -v aarch64-linux-gnu-g++ >/dev/null || { echo "ERROR: install g++-aarch64-linux-gnu"; exit 1; }

# (re)configure if the cache is missing, points at a different toolchain, or predates the KleidiAI+LTO options.
# A toolchain CONTENT change (we switched -mcpu->-march for the KleidiAI dotprod gate) is invisible to cmake's
# path check AND CMAKE_*_FLAGS_INIT only applies on a FRESH configure -- so wipe the dir to force it.
if [ ! -f "$BUILD/CMakeCache.txt" ] \
   || ! grep -q "$TOOLCHAIN" "$BUILD/CMakeCache.txt" 2>/dev/null \
   || ! grep -q "GGML_CPU_KLEIDIAI:BOOL=ON" "$BUILD/CMakeCache.txt" 2>/dev/null; then
  echo "[pi_build] configuring cross build (Cortex-A76, NATIVE=OFF, KleidiAI+LTO)..."
  rm -rf "$BUILD/CMakeCache.txt" "$BUILD/CMakeFiles"
  cmake -S "$LLAMA" -B "$BUILD" \
    -DCMAKE_TOOLCHAIN_FILE="$TOOLCHAIN" \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_NATIVE=OFF -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF \
    -DGGML_CPU_KLEIDIAI=ON -DGGML_LTO=ON >/dev/null
fi

echo "[pi_build] building llama-server (aarch64)..."
cmake --build "$BUILD" --target llama-server -j"$(nproc)"

[ -x "$BIN/llama-server" ] || { echo "ERROR: build produced no llama-server"; exit 1; }

echo "[pi_build] sanity: aarch64 ELF + patched flags (via qemu, if present)"
file "$BIN/llama-server" | grep -q aarch64 || { echo "ERROR: not an aarch64 binary"; exit 1; }
if command -v qemu-aarch64-static >/dev/null; then
  # decouple from qemu's exit code (--help exits nonzero; pipefail would mis-report)
  help_out="$(qemu-aarch64-static -L /usr/aarch64-linux-gnu -E LD_LIBRARY_PATH="$BIN" "$BIN/llama-server" --help 2>/dev/null || true)"
  if printf '%s' "$help_out" | grep -qE -- '--spec-tree-k'; then
    echo "  ok: tree-draft + n-gram flags present"
  else
    echo "  WARN: could not verify flags under qemu (binary still built)"
  fi
fi

echo "[pi_build] ABI check (must be satisfiable by the target Pi OS):"
MAXGLIBC=$(aarch64-linux-gnu-readelf -V "$BIN"/llama-server "$BIN"/*.so* 2>/dev/null | grep -oE 'GLIBC_[0-9.]+'   | sort -V | tail -1)
MAXCXX=$(  aarch64-linux-gnu-readelf -V "$BIN"/llama-server "$BIN"/*.so* 2>/dev/null | grep -oE 'GLIBCXX_[0-9.]+' | sort -V | tail -1)
echo "  needs <= $MAXGLIBC and <= $MAXCXX   (bookworm provides GLIBC_2.36 / GLIBCXX_3.4.30)"

echo "[pi_build] DONE. Package ($(du -shc "$BIN"/llama-server "$BIN"/*.so* 2>/dev/null | tail -1 | cut -f1)):"
echo "  $BIN/{llama-server, *.so*}"
echo "  Deploy it with:  PI_HOST=pi@<ip> $ROOT/scripts/pi_deploy.sh"

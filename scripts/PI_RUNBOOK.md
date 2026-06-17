# Raspberry Pi 5 runbook — Gemma‑4‑E2B lossless speculative stack

Everything needed to build, deploy, run, and A/B the speculative-decoding stack on a Pi 5
(BCM2712 / Cortex‑A76) from this x86 dev box. The Pi has **no model/build of its own** —
we cross-compile here and push a self-contained package.

## 0. One-time: what's already prepared on this box
- **Cross toolchain**: `aarch64-linux-gnu-g++` (GCC 11.4) + `scripts/aarch64-a76.toolchain.cmake` (`-mcpu=cortex-a76`, dotprod+fp16, NATIVE=OFF).
- **Patched aarch64 binary**: `third_party/llama.cpp/build-a76/bin/llama-server` (+ project `.so`).
  Built from the tree-draft + n-gram patched source; verified to load a model on ARM with
  `NEON=1 DOTPROD=1 REPACK=1`, and its flags include `--spec-tree-k` / `--spec-ngram-simple-size-n`.
- **ABI**: needs ≤ `GLIBC_2.34` / ≤ `GLIBCXX_3.4.30` → runs on **Raspberry Pi OS bookworm** (glibc 2.36).
  On an **older** Pi OS (bullseye), build natively on the Pi instead — see §5.

## 1. Connect the Pi
Get it reachable over SSH (Ethernet/Wi-Fi or USB-gadget). Confirm:
```
ssh pi@<pi-ip> 'uname -m && . /etc/os-release && echo $VERSION_CODENAME'   # expect: aarch64 + bookworm
```
Set once in your shell:  `export PI_HOST=pi@<pi-ip>`

## 2. Build (only after editing llama.cpp source; the current binary is already built)
```
./scripts/pi_build.sh        # cross-compiles llama-server for A76 + prints the ABI check
```

## 3. Deploy
```
# first time (push binary+scripts AND the models — several GB, resumable):
PI_HOST=$PI_HOST ./scripts/pi_deploy.sh --models
# subsequent (after a rebuild — binary+scripts only, ~19 MB, fast):
PI_HOST=$PI_HOST ./scripts/pi_deploy.sh
```
Creates on the Pi: `~/cpullm/{bin,scripts,models}`. (If you'd rather copy models manually:
`scp models/gemma-4-E2B-it-Q4_0.gguf models/mtp-gemma-4-E2B-it.gguf $PI_HOST:~/cpullm/models/`.)
NOTE: `Q4_K_M` (the recommended quant) is now present in `models/` and is what `run_pi5.sh` picks by
default (it auto-falls back to `Q4_0`→`Q3_K_M` only if absent). `--models` deploys all `*.gguf`; to push
just the two you need: `scp models/gemma-4-E2B-it-Q4_K_M.gguf models/mtp-gemma-4-E2B-it.gguf $PI_HOST:~/cpullm/models/`.

## 4. Run on the Pi
```
ssh $PI_HOST
cd ~/cpullm/scripts
./run_pi5.sh tune            # one-time: performance governor + transparent hugepages (needs sudo)
./run_pi5.sh launch          # baseline: Q4_K_M(or fallback) + MTP, pinned to the 4 A76 cores, port 8080
```
Opt-in levers (default off; lossless; composable):
```
TREE_K=2 ./run_pi5.sh launch     # tree-draft: +~8% tok/pass any workload (projected +4–8% Pi tok/s)
NGRAM=1  ./run_pi5.sh launch     # n-gram: +27% on LITERAL-repetition output, ~neutral elsewhere
TREE_K=2 NGRAM=1 ./run_pi5.sh launch   # both
```
Leave the server running; open a second terminal (or run from this box) for measurement.

## 5. Measure / A/B (run from THIS box, pointed at the Pi)
The bench/gate scripts take `HOST=<pi-ip>` (default localhost):
```
# throughput multiplier (tok/pass) + wall-clock tok/s, per workload suite:
HOST=<pi-ip> python3 scripts/pi_bench.py 8080 --suite all --runs 3

# losslessness gate — run a pure-target server (no -md/--spec-type) on another port (e.g. 8081) and:
HOST=<pi-ip> python3 scripts/lossless_gate.py 8081 8080 64
#   PASS = byte-identical; a few coherent near-tie diffs == MTP's known batch non-invariance (still lossless).
```
A/B protocol: launch config A (e.g. plain MTP) on 8080, config B (e.g. `TREE_K=2`) on 8081
(set `PORT=8081`), then `pi_bench.py` each and compare `tok/pass` **and** `tok/s`.
`scripts/tree_roofline.py` predicts the Pi tok/s gain from the (hardware-independent) tok/pass.

## 6. If the cross binary won't run on the Pi (older OS / GLIBC error)
Build natively on the Pi (slower but ABI-proof):
```
PI_HOST=$PI_HOST ./scripts/pi_deploy.sh           # pushes scripts/models; then on the Pi:
ssh $PI_HOST 'sudo apt-get install -y cmake g++ && \
  git clone --depth1 <your llama.cpp fork-with-patches> ~/cpullm/llama.cpp || true'
# (or rsync this third_party/llama.cpp source up), then:
ssh $PI_HOST 'cmake -S ~/cpullm/llama.cpp -B ~/cpullm/llama.cpp/build -DGGML_NATIVE=ON -DLLAMA_CURL=OFF && \
  cmake --build ~/cpullm/llama.cpp/build --target llama-server -j4'
# then point run_pi5.sh at it:  BIN=~/cpullm/llama.cpp/build/bin/llama-server ./run_pi5.sh launch
```

## 7. Known constraints (don't fight these)
- `-fa` (flash-attn) and `-ctk/-ctv` (KV quant) **crash** with `draft-mtp` — do not add them.
- Decode is memory-bandwidth bound: tok/s saturates at **4 threads** (the 4 A76 cores); more is wasted.
- Lossless single-stream ceiling ≈ **25–30 tok/s** on Pi5 (`BW/bytes × spec_mult`); 100 tok/s single-stream
  lossless is out of reach (see PLAN §42). The levers here stack on MTP within that ceiling.
- Quant: keep **Q4_K_M/Q4_0** (A76-fast repack kernel, BW-bound). `Q3_K_M` is compute-bound on A76 (no repack
  kernel) and ties/loses — only worth an A/B, not a default (PLAN §29/§38).

#!/usr/bin/env bash
# ============================================================================
# Gemma 4 E2B on Raspberry Pi 5 (BCM2712 / Cortex-A76) — max-throughput launcher
# All levers here are LOSSLESS (system tuning + MTP speculative decoding, which
# is mathematically exact). Quant choice is the one place quality is traded —
# default Q4_0 keeps it tiny; see §QUANT below for the measured-safe options.
#
# Validated facts this encodes (from the dev-box measurements):
#  - decode is memory-bandwidth bound; tok/s saturates at 4 threads (= the 4 A76
#    cores). More threads do nothing. So: -t 4, pinned, performance governor.
#  - MTP speculative decoding (--spec-type draft-mtp) is lossless and gives a
#    measured ~2x on code/reasoning, ~2.4x avg. n_max=3 is optimal (deeper hurts).
#  - the 97MB MTP head is cheap enough to actually pay off on a BW-bound CPU.
# ============================================================================
set -euo pipefail

MODELS=${MODELS:-$HOME/mote/models}
DRAFT=${DRAFT:-$MODELS/mtp-gemma-4-E2B-it.gguf}        # MTP head, 94MB
# BIN: the deployed package (scripts/pi_deploy.sh puts the binary + project .so here).
BIN=${BIN:-$HOME/mote/bin/llama-server}
PORT=${PORT:-8080}

# TARGET: prefer Q4_K_M (A76-fast, BW-bound; see §QUANT), but fall back to whatever quant is actually present
# so a fresh deploy "just runs". Override with TARGET=/path/to/model.gguf.
if [ -n "${TARGET:-}" ]; then :; else
  # Prefer Q4_0: MEASURED +8% over Q4_K_M on the A76 (its flat per-32 scale dequant is near sdot-peak; Q4_K's
  # hierarchical scales cost ~23% in the compute-bound verify). Quality is slightly lower (accepted trade).
  for q in Q4_0 Q4_K_M Q3_K_M; do
    if [ -f "$MODELS/gemma-4-E2B-it-$q.gguf" ]; then TARGET="$MODELS/gemma-4-E2B-it-$q.gguf"; break; fi
  done
  TARGET=${TARGET:-$MODELS/gemma-4-E2B-it-Q4_0.gguf}   # last resort (will error clearly if absent)
fi

# ---- §MODE: one-word speed/quality preset. Picks the dials below; any explicit env (RELAX_PRATIO=, PMIN=,
# NGRAM=, NMAX=) still overrides. MEASURED on a real Pi5 (pre-throttle, tower-cooler conditions, Q4_0+MTP):
#
#   MODE      general  structured(JSON/SQL/CSV)  literal(logs/repeats)   quality
#   quality   11.2     13.4                      16.0                    EXACT (greedy, lossless)   <-- default
#   fast      12.5     13.9                      17.2                    lossy (~half tokens drift from greedy)
#   turbo     12.3     13.4                      20.4                    lossy + n-gram (for repetitive output)
#   (quality = +20% over the stock build: NEON bf16 dot, fast logits PAD, ARM KleidiAI matmul kernels, LTO. all exact.)
#
# 15 tok/s is NOT reachable for general/prose on this model: the verify is memory-
# bandwidth bound and the MTP draft's 2nd/3rd-token accuracy caps tokens/pass. 'fast' buys ~+17% general at a
# real quality cost. 'turbo' adds the n-gram drafter and reaches 15-18 tok/s ONLY on literal/log/boilerplate
# output (it slightly HURTS structured/prose because the n-gram displaces the stronger MTP draft). Pick 'turbo'
# only when your output is highly repetitive.  Usage:  MODE=fast ./run_pi5.sh launch
MODE=${MODE:-quality}
case "$MODE" in
  quality) : ;;                                                        # lossless exact greedy (all dials off)
  fast)    : "${RELAX_PRATIO:=0.08}" "${RELAX_TOPK:=8}" "${PMIN:=0.1}" ;;
  turbo)   : "${RELAX_PRATIO:=0.08}" "${RELAX_TOPK:=8}" "${PMIN:=0.1}" "${NGRAM:=1}" ;;
  *) echo "ERROR: unknown MODE='$MODE' (want quality|fast|turbo)"; exit 1 ;;
esac

# ---- §TREE: tree-draft (multi-hypothesis MTP) — OPT-IN, default OFF ----------
# TREE_K=1 (default) == plain linear MTP (the shipped, proven ~2.5x win). TREE_K=2 enables tree-draft:
# each verify pass checks a SECOND draft hypothesis (the top-2 token at the first uncertain step) in the
# SAME target weight-read, committing whichever chain the target accepts further. It is LOSSLESS (63-agent
# adversarial review found no correctness/KV bug; every committed token is the in-batch target argmax,
# verified identical-to-MTP divergence). MEASURED gain: +8.7% tokens-per-pass, +17% acceptance (dev box,
# hardware-independent). Roofline projects +4-8% Pi tok/s on top of MTP because
# the target weight-read dominates each pass and the branch rows hide under it (avg verify M < A76 crossover).
#   *** This is a PREDICTION — confirm on real Pi5 silicon. *** On compute-rich/fast-memory hosts (dev box)
#   it REGRESSES (-23%) because the per-pass overhead dominates a cheap read; the Pi inverts that economics.
# To A/B on a real Pi:  TREE_K=2 ./run_pi5.sh launch   (compare tok/s vs the default TREE_K=1).
# TREE_P_MAX = branch only when the draft's top-1 prob < this (0.9 captures the gain; 0.7 is more selective).
TREE_K=${TREE_K:-1}
TREE_P_MAX=${TREE_P_MAX:-0.9}

# ---- §NGRAM: prompt-lookup (n-gram) draft chained with MTP — OPT-IN, default OFF -----
# NGRAM=1 adds a free n-gram/prompt-lookup drafter ahead of MTP (--spec-type draft-mtp,ngram-simple). It drafts
# long EXACT repeats straight from the context (no model eval). LOSSLESS (target verifies; divergence == MTP).
# WORKLOAD-DEPENDENT (measured): on LITERAL-repetition output (logs, data/CSV/JSON dumps, repeated boilerplate)
# it is +27% tokens/pass; on normal chat/code/structured text it is ~NEUTRAL (-0.3% to -0.7%, noise). It is NOT
# a general win because MTP is already a strong LEARNED draft — a short n-gram match is worse than MTP and, since
# n-gram runs priority-first, a too-small lookup DISPLACES MTP and loses (size_n=3 measured -4% general). NGRAM_N=8
# is the no-regret setting: long enough that n-gram only fires on confident exact matches (keeps the +27% repeat
# win) yet rarely on prose (no displacement). Lower NGRAM_N (3-4) only if your output is almost entirely literal
# repetition. => Enable ONLY for repetition-heavy workloads.  Usage:  NGRAM=1 ./run_pi5.sh launch
NGRAM=${NGRAM:-0}
NGRAM_N=${NGRAM_N:-8}

# ---- §NMAX: MTP draft depth. n_max=10 was an EARLIER 'Pi optimum', but real Pi5 measurement shows the
# Q4_K_M verify is COMPUTE-bound at M~=11 (effective ~6.5 GB/s << 17 peak), so deeper drafting raises M and
# COSTS compute. A SHALLOWER draft (lower M/pass) can be faster despite lower acceptance. Sweep to find the
# real optimum on hardware:  for n in 3 4 6 8 10; do NMAX=$n PORT=80$n ./run_pi5.sh launch & done
# MEASURED on Pi5: the q4_0 GEMM streams weights ONCE per 4-ROW TILE (the active A76 kernel is q4_0_4x4 — the 8x8
# i8mm path is a scalar stub here). The MTP verify runs at M = 1(sampled) + n_max(draft). So FILL the tile exactly:
# n_max=3 -> M=4 = one weight read, 3 draft tokens. n_max=4 -> M=5 SPILLS one row into a separate leftover gemv that
# RE-STREAMS the weights (+60 ms/pass). Measured (Q4_0): n_max 2->6.29, 3->9.28, 4->7.88, 7(M=8)->7.65 tok/s.
# => n_max=3 is the optimum (+18% over n_max=4, +56% over the original n_max=10). Deeper drafts lose: the marginal
#    draft token's low acceptance never pays for spilling past the tile. (Tile size 4 = the q4_0_4x4 kernel.)
NMAX=${NMAX:-3}

# ---- §PMIN: draft confidence gate (--spec-draft-p-min). The MTP draft stops proposing once its top-1 prob
# drops below PMIN. Default 0.5 = only draft confident tokens (best for LOSSLESS: deeper low-confidence drafts
# get rejected by greedy anyway). MEASURED (x86, hardware-independent): lowering PMIN makes the draft propose
# its full n_max depth (draft/gen 0.89->1.22) but the extra tokens have LOW acceptance, so it only helps a
# little when paired with relaxed acceptance -- tok/pass tops out ~2.6 (the MTP 2nd/3rd-token accuracy wall).
PMIN=${PMIN:-0.5}

# ---- §RELAX: LOSSY relaxed-acceptance dial — OPT-IN, default OFF (RELAX_PRATIO=0 = lossless) ----
# Accept a draft token that is NOT the target's argmax if it is among the target's top-RELAX_TOPK candidates
# AND its prob >= RELAX_PRATIO * top1_prob. Raises tokens/pass on PROSE (where greedy is uncertain) -> faster,
# at a tunable QUALITY cost. CORRECTED MEASUREMENT (validated x86-sweep -> real-Pi confirm, within ~2%): this
# dial is a WEAK lever. The earlier "~13.5" was an artifact; real Pi general: lossless 9.44 -> pratio0.3 9.88
# (sim 0.69) -> pratio0.08+pmin0.1 11.0 (sim 0.53). So you pay a large similarity drop for ~+17% at best, because
# tokens/pass only moves 2.29->2.59 (the MTP 2nd/3rd-token accuracy wall; n_max=3 prefix caps it). It barely
# helps CODE/structured (already well-predicted). The 'fast' MODE preset uses pratio0.08+pmin0.1. 0 = exact/lossless.
# (Implemented in the binary as LLAMA_SPEC_ACCEPT_TOPK/PRATIO; this just plumbs the env through.)
RELAX_PRATIO=${RELAX_PRATIO:-0}
RELAX_TOPK=${RELAX_TOPK:-8}

# ---- §SYSTEM: one-time lossless host tuning (needs sudo; safe to re-run) ----
tune_system() {
  echo "[tune] performance governor on all 4 cores"
  for c in /sys/devices/system/cpu/cpu[0-3]/cpufreq/scaling_governor; do
    echo performance | sudo tee "$c" >/dev/null 2>&1 || true
  done
  echo "[tune] enable transparent hugepages (2MB) — cuts TLB misses on the 1.3GB weight stream"
  echo always | sudo tee /sys/kernel/mm/transparent_hugepage/enabled >/dev/null 2>&1 || true
  echo defer+madvise | sudo tee /sys/kernel/mm/transparent_hugepage/defrag >/dev/null 2>&1 || true
  echo "[tune] (optional) move IRQs off the compute cores and check throttling"
  vcgencmd get_throttled 2>/dev/null || true
}

# ---- §OVERCLOCK: edit /boot/firmware/config.txt then reboot (BW = the wall) ----
# Memory bandwidth is the roofline, so a memory/SoC OC is ~linear tok/s. Needs
# active cooling. Conservative, widely-stable values:
#   arm_freq=2900          # A76 2.4 -> 2.9 GHz
#   over_voltage_delay=0
#   force_turbo=0
#   gpu_freq=960
# (LPDDR4X timing is fixed on Pi5; arm/SoC clock + good cooling is the practical lever.)
print_overclock_hint() {
  cat <<'EOF'
[overclock] Append to /boot/firmware/config.txt (then reboot), requires cooling:
    arm_freq=2900
    force_turbo=0
    gpu_freq=960
  Verify after reboot:  vcgencmd measure_clock arm ; vcgencmd get_throttled
EOF
}

# ---- §QUANT: pick for the A76 KERNEL, not just bytes ----
# COUNTERINTUITIVE (llvm-mca on real A76 kernels): fewer bytes != faster here.
#   gemma-4-E2B-it-Q4_K_M.gguf  ~1383 MB/tok  *** RECOMMENDED *** has fast repacked
#                                gemv kernel -> BANDWIDTH-bound -> scales with BW/overclock.
#   gemma-4-E2B-it-Q4_0.gguf    ~1383 MB/tok  also fast repacked gemv (slightly lower quality)
#   gemma-4-E2B-it-Q3_K_M.gguf  ~1172 MB/tok  AVOID on A76: no repacked kernel -> COMPUTE-bound
#                                at ~7.8 GB/s, STUCK regardless of memory BW. ~32-86% SLOWER than Q4_K_M.
# So default to Q4_K_M: more bytes but the A76 actually saturates the bus with it, and quality is
# >= Q3_K_M. (2-bit/IQ2 are off the table: too lossy for E2B.)
# Confirm decode tok/s on the real Pi (this is an llvm-mca prediction, very strong but model-based).
#
# EXPERIMENTAL: a hand-written NEON+dotprod q3_K repacked gemv is now integrated
# (arch/arm/repack.cpp, default-on for ARM+dotprod). q3_K is ~15% fewer bytes than Q4_K_M; the kernel
# is bit-exact/lossless and *may* be BW-bound (~+18%) on the A76 -- but that hinges on the OoO core
# hiding the in-kernel scale unpack, which only real Pi silicon can confirm (llvm-mca/QEMU can't).
# To A/B on a real Pi:  TARGET=$MODELS/gemma-4-E2B-it-Q3_K_M.gguf  (then compare tok/s vs Q4_K_M).
# If q3_K wins, it stacks with MTP. If it ties/loses, keep Q4_K_M.

# ---- §LAUNCH: MTP speculative server, A76-tuned -------------------------
launch() {
  # tree-draft is opt-in via TREE_K>1 (default 1 = plain linear MTP); see §TREE above.
  local tree_args=() ; local spec_type="draft-mtp" ; local ngram_args=() ; local tag="MTP"
  if [ "${TREE_K}" -gt 1 ] 2>/dev/null; then
    tree_args=(--spec-tree-k "$TREE_K" --spec-tree-p-max "$TREE_P_MAX") ; tag="$tag+tree(k=$TREE_K)"
  fi
  # n-gram prompt-lookup is opt-in via NGRAM=1 (default off); see §NGRAM above.
  if [ "${NGRAM}" = "1" ]; then
    spec_type="draft-mtp,ngram-simple"
    ngram_args=(--spec-ngram-simple-size-n "$NGRAM_N" --spec-ngram-simple-size-m 16) ; tag="$tag+ngram(n=$NGRAM_N)"
  fi
  # relaxed (lossy) acceptance dial — only when opted in (RELAX_PRATIO>0); see §RELAX
  if awk "BEGIN{exit !($RELAX_PRATIO>0)}" 2>/dev/null; then
    export LLAMA_SPEC_ACCEPT_PRATIO="$RELAX_PRATIO" LLAMA_SPEC_ACCEPT_TOPK="$RELAX_TOPK"
    tag="$tag+relax(p=$RELAX_PRATIO)"
  fi
  echo "[launch] mode=$MODE  target=$(basename "$TARGET")  draft=$tag  port=$PORT"
  [ -f "$TARGET" ] || { echo "ERROR: model not found: $TARGET (set TARGET= or deploy with --models)"; exit 1; }
  [ -x "$BIN" ]    || { echo "ERROR: binary not found: $BIN (run pi_build.sh + pi_deploy.sh, or set BIN=)"; exit 1; }
  # the deployed binary's project .so live next to it -> let the loader find them
  export LD_LIBRARY_PATH="$(dirname "$BIN")${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  # -t 4            : the 4 A76 cores (decode saturates here; more is wasted)
  # --spec-type draft-mtp + -md head : lossless ~2x on code/reasoning, ~2.5x avg
  # --spec-draft-n-max 10            : MEASURED Pi optimum (+14.5% overall, +37% code vs n_max=3).
  #   On the bandwidth-bound Pi each verify pass is ~1 target weight-read regardless of batch, so deeper
  #   drafting wins; the MTP draft streams only 10.7 MB/token, so its per-step cost is small. Peak at ~10.
  #   (--spec-draft-n-max also bounds the tree branch depth, clamped further to remaining context.)
  # NOTE: -fa (flash-attn) and -ctk/-ctv (KV quant) CRASH with draft-mtp — the MTP
  #       shared-KV/nextn path is incompatible with them (verified). They'd only help
  #       at long context anyway (decode is weights-dominated), so MTP's 2.5x wins.
  #       => do NOT add -fa / KV-quant here.
  # --mlock                          : keep weights resident, no page-out
  exec taskset -c 0-3 "$BIN" \
    -m "$TARGET" -md "$DRAFT" --spec-type "$spec_type" --spec-draft-n-max "$NMAX" --spec-draft-p-min "$PMIN" \
    "${tree_args[@]}" "${ngram_args[@]}" \
    -t 4 -tb 4 -c 4096 \
    --mlock --no-warmup \
    --host 0.0.0.0 --port "$PORT"
}

case "${1:-launch}" in
  tune)       tune_system ;;
  overclock)  print_overclock_hint ;;
  launch)     tune_system; launch ;;
  *) echo "usage: $0 [tune|overclock|launch]"; exit 1 ;;
esac

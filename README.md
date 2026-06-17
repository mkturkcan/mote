# Mote

*An open community project for running capable language models on small, common hardware: the Intelligence of Things. The Raspberry Pi 5 is the first target; the techniques carry to other Arm CPUs. Mote is independent and not affiliated with the developers of Gemma.*

Run Google's **Gemma‑4‑E2B**, an effective 2.3B‑parameter model, on a **Raspberry Pi 5** (BCM2712 /
Cortex‑A76, 4 cores) at **9–18 tokens/sec** depending on workload and the quality mode you pick. The engine is
`llama.cpp` driven by **MTP speculative decoding** (Gemma 4's first‑party draft head), tuned for the A76's
memory system.

Gemma‑4‑E2B keeps **2.3B effective** parameters and **5.1B** in total. The extra mass is a per‑layer embedding
table read as a lookup rather than multiplied, so it streams from storage instead of occupying working memory.
The text‑only GGUF here loads as about **4.65B** parameters: it still carries that table, with the vision and
audio encoders stripped.

All numbers below are **measured on real Pi 5 silicon** (Q4_0 weights, 4 threads @ 2.4 GHz, pre‑throttle /
active‑cooled). They are reproducible with `scripts/pi_bench.py`.

## TL;DR, measured tok/s by mode and workload

| `MODE=` | general (incl. prose) | structured (JSON/SQL/CSV/code) | literal (logs / repeated lines) | quality |
|---|:---:|:---:|:---:|---|
| **`quality`** (default) | **11.2** | 13.4 | 16.0 | **exact**, distribution‑faithful to the target |
| **`fast`** | **12.5** | 13.9 | 17.2 | lossy, ~half the tokens drift from greedy |
| **`turbo`** | 12.3 | 13.4 | **20.4** | lossy + n‑gram, built for repetitive output |

> **vs just running the GGUF.** The *same* Q4_0 model in a stock GGUF library (`llama-cpp-python`, single‑stream,
> no speculative decoding, what most people actually run) does **6.6 tok/s** on this Pi. This project's MTP
> speculative decoding + A76 kernels + system tuning make it **1.7× faster on general text (11.2), 2.0× on
> structured (13.4), and up to 2.4× on literal output (16.0)**, and `quality` mode stays distribution‑faithful,
> so that speedup is *free of quality loss*. (The newest kernel work, `bf16` dot, `PAD`, KleidiAI, LTO, is the
> last +20% of that climb, 9.4 → 11.2.) The naive PyTorch/`transformers` path is slower still, its fp16 weights
> (~9 GB) don't even fit in the Pi's 8 GB RAM.

## Install & run (Raspberry Pi OS, 64‑bit)

From a clone of this repo, one command does everything, builds the tuned server, downloads the model (~3 GB),
installs the `gemma` command, and tunes the system:

```bash
bash scripts/install.sh
```

Or, once a prebuilt package is attached to a GitHub release, a fresh Pi installs in seconds with no build (see
[`docs/RELEASE.md`](docs/RELEASE.md)):

```bash
curl -fsSL https://github.com/mkturkcan/mote/releases/latest/download/install.sh \
  | CPULLM_PKG_URL=https://github.com/mkturkcan/mote/releases/latest/download/gemma-pi5.tar.gz bash
```

Then it's one command to run and play with:

```bash
gemma start            # launch the server (lossless; also: gemma start fast | turbo)
gemma chat             # chat in the terminal
#  …or open  http://<pi-ip>:8080  in any browser for the built‑in web chat UI
gemma bench            # quick speed test
gemma stop
```

Under the hood it's an OpenAI‑compatible server on `:8080` (`/completion`, `/v1/chat/completions`, web UI at `/`).
Power users can drive the launcher directly: `MODE=fast bash scripts/run_pi5.sh launch`.

### Which mode?

- **`quality`** is the default. Output is **distribution‑exact**: every emitted token is a true argmax
  of the full target model, so you get the model's real answer with **no quality loss**, just faster. Use this
  for anything where the answer matters.
- **`fast`**, trades quality for ~+17% on general text by accepting draft tokens that are *near* the target's
  top choice (not exactly it). On prose roughly half the tokens diverge from the greedy path, so it reads
  slightly differently and can drift on facts. Fine for drafts, chat, brainstorming.
- **`turbo`**, `fast` plus an n‑gram drafter that replays long exact repeats straight from the context. Only
  worth it when your output is **highly repetitive** (logs, CSV/JSON dumps, repeated boilerplate), where it hits
  **15–18 tok/s**. It *slightly hurts* prose/structured output (the n‑gram displaces the stronger MTP draft), so
  don't use it as a general default.

## Why these numbers, and why 15 tok/s general is *not* reachable here

CPU decode is **memory‑bandwidth bound**: each token streams the model's weights through RAM once, so
`tok/s ≈ sustained_BW / bytes_per_token`. Speculative decoding is the one big lever, it **amortizes a single
weight read over several emitted tokens** by drafting ahead with the tiny (94 MB) MTP head and verifying a batch
against the full model.

Two hard walls cap general throughput on this device, and we measured both:

1. **The verify read is bandwidth‑bound.** Even an arm overclock doesn't help, the Pi 5's LPDDR4X timing is
   fixed, and the weight read, not compute, dominates each pass. This sets the **pass rate** (~4.2 passes/s).
2. **The draft can only predict so far.** Tokens/pass is `pass_rate × accepted_tokens`. The MTP draft's
   2nd/3rd‑token accuracy is the ceiling: even accepting *every* near‑miss (heavy quality loss), tokens/pass
   tops out at ~2.6 of a theoretical 4.0. That's ~11 tok/s general, **not 15.**

Reaching 15 tok/s general would need a structurally better draft or a smaller draft‑compatible target (the MTP
head is Gemma‑4‑E2B‑specific, so you can't just swap in a smaller model). 15+ *is* reachable, but only on
**literal/repetitive** output, which is what `turbo` targets, and the README reports those figures separately from the general number.

### The single biggest tuning insight: fill the 4‑row tile

The active A76 GEMM kernel (`q4_0_4x4`) streams the weights **once per 4‑row tile**. The MTP verify batch is
`M = 1 sampled + n_max drafted`. So **`n_max=3` → M=4 exactly fills one tile** = one weight read for 3 draft
tokens. `n_max=4` → M=5 spills one row into a second gemv that **re‑streams the whole weight matrix** (+60 ms).
Measured: n_max 2→6.3, **3→9.4**, 4→7.9, 7→7.7 tok/s. This single setting is +56% over the naive `n_max=10`.
(`Q4_0` over `Q4_K_M` is a further +8%: its flat per‑32 scale dequant runs near the A76's sdot peak.)

### Kernel work (A76-specific, all distribution-exact)

Profiling a `quality`-mode pass (244 ms = 204 ms M=4 verify + ~40 ms for 3 MTP draft steps; verify is
bandwidth-bound) found the engine was already near-optimal **except** where Gemma's MatFormer scaffolding hit
*generic, unvectorized* kernels on the A76. The wins:

- **ARM-NEON `bf16` dot product** (`ggml_vec_dot_bf16`). `llama.cpp` had AVX-512/AVX2/POWER paths but **no NEON
  one**, and the A76 has no BF16 dot extension, so the 27.5 MB `per_layer_model_proj` (read every pass) ran a
  fully *scalar* loop. The NEON path mirrors the AVX2 arithmetic exactly (`vshll_n_u16` for the exact bf16→f32
  upconvert, multiply-then-add, f32 accumulate). **Result: ~16 ms/pass saved → +5% general, +9% structured, +19%
  literal.** Distribution-exact (a 3-lens adversarial review + aarch64/qemu checks confirmed bit-exact upconvert
  and no UB; outputs are identical-or-equally-valid-greedy, the same standard the AVX2 path already meets).
- **Fast `PAD`** (`ggml_compute_forward_pad_f32`). A deep per-op profile found the 256K-vocab **logits pad**, run
  every forward, was a *scalar, per-element, 4-way-bounds-checked* copy (upstream `// TODO: optimize`). Replaced
  with a `memcpy`/`memset` fast path parallelized over the vocab axis. **Bit-identical** (KL=0); the PAD slice
  dropped 1.74% → 0.43% (4.2× faster), for **+1.5–2%** end-to-end across all modes.
- **ARM-NEON `tanh`** (`ggml_vec_tanh_f32`, for the 256K-vocab final-logit softcap). Implemented and verified
  *argmax-exact* (softcap is monotonic → identical tokens, confirmed sim 1.0), but **throughput-neutral**, the
  softcap is only ~1% of a pass. Kept (correct, free, helps sampling/long-context) but **not** a decode-speed win.

- **KleidiAI** (`-DGGML_CPU_KLEIDIAI=ON`), **ARM's own hand-tuned matmul microkernels**, and the biggest single
  win. The decode is 81.7% matmul; a 15-agent audit confirmed the in-tree `q4_0_4x4` GEMM is near-optimal (hand-
  written A76 assembly, ~16 SDOT accumulators, 97% of SDOT peak, software prefetch, Q4_K dequant, F32 re-reads and
  gemv tails were all ruled out). But ARM's KleidiAI `neon_dotprod` kernel, which handles exactly our Q4_0 weights
  and Q8_0 draft head, **beat it by 8.4%** (weight-matmul time 21.3M→19.5M µs; decode rate 4.48→4.86 passes/s).
  Weights are bit-exact (repacked Q4_0); it re-quantizes activations with slightly different rounding, so it's
  distribution-exact (coherent valid greedy paths) but drifts a bit more from the baseline than the other kernels.
  *Cross-compile note:* KleidiAI's cmake gates its dotprod kernels on a literal `+dotprod` in the arch flags, so the
  toolchain uses `-march=armv8.2-a+dotprod+fp16` (not `-mcpu=cortex-a76`, which lacks the literal → link failure).
- **LTO** (`-DGGML_LTO=ON`), link-time optimization, **+1%, bit-identical** (sim 1.0). Free; stacks with the above.

**Net of all kernel work: general 9.4 → 11.3 tok/s (+20%), distribution-exact.** The remaining bottleneck is the
bandwidth wall, which only an overclock (firmware-locked) or requantization (quality loss) can move.

## "Lossless" means distribution‑exact (an important subtlety)

Speculative decoding is **not byte‑identical** to single‑stream decode: transformer inference isn't
batch‑invariant, so the multi‑token verify batch rounds floating‑point differently than batch‑1 decode and can
flip a near‑tie argmax onto a *different but equally valid* greedy path. The rigorous definition of "no quality
loss" is therefore **distribution‑exact (KL≈0)**, every emitted token is a true argmax of the target, which
`quality` mode guarantees by construction. `scripts/lossless_gate.py` is the regression harness for it.

## Build & deploy to a Pi

The Pi runs a **cross‑compiled, self‑contained package** (binary + project `.so` + GGUFs); it builds nothing
itself. Full steps in [`scripts/PI_RUNBOOK.md`](scripts/PI_RUNBOOK.md). Short version, from this x86 dev box:

```bash
export PI_HOST=mklab@<pi-ip>
./scripts/pi_build.sh                       # cross-compile llama-server for Cortex-A76 (+ ABI check)
PI_HOST=$PI_HOST ./scripts/pi_deploy.sh --models   # push binary, scripts, and the GGUFs (resumable)
ssh $PI_HOST 'cd cpullm && MODE=quality bash scripts/run_pi5.sh launch'
```

Requires Raspberry Pi OS **bookworm** (glibc ≥ 2.36). The model files: target `gemma-4-E2B-it-Q4_0.gguf`
(2.9 GB) + draft `mtp-gemma-4-E2B-it.gguf` (94 MB); `scripts/dl_quants.py` fetches them.

## Benchmark / verify

```bash
# throughput per workload (run on the Pi, or from x86 with HOST=<pi-ip>):
HOST=<pi-ip> python3 scripts/pi_bench.py 8080 --suite all --n 96

# prove quality mode is distribution-exact vs the pure target:
python3 scripts/lossless_gate.py REF MTP

# quality of a lossy mode vs greedy (run OFF the Pi, quality is hardware-independent):
python3 scripts/relaxed_quality.py <ref_port> <cand_port>
```

## Dials (all optional; the `MODE` presets set sensible defaults)

`run_pi5.sh` reads these env vars, see the `§`‑sections in the script for the measured rationale behind each:

| var | default | what it does |
|---|---|---|
| `MODE` | `quality` | preset: `quality` \| `fast` \| `turbo` (sets the dials below) |
| `NMAX` | `3` | MTP draft depth. **3 = fills the A76 4‑row tile** (the optimum); deeper spills and costs. |
| `TARGET` | `Q4_0` | weight file; `Q4_0` is fastest on A76. Falls back to whatever quant is present. |
| `RELAX_PRATIO` | `0` | lossy accept: take a non‑argmax draft token if `prob ≥ pratio × top1`. `0` = exact. |
| `PMIN` | `0.5` | draft confidence gate; lower lets the draft propose deeper (only helps with `RELAX`). |
| `NGRAM` | `0` | add the n‑gram drafter (helps **only** literal‑repeat output). |
| `TREE_K` | `1` | tree‑draft (multi‑hypothesis MTP); opt‑in, regresses on this Pi (see `§TREE`). |

## Repository layout

- [`scripts/run_pi5.sh`](scripts/run_pi5.sh), turnkey Pi 5 launcher with the three modes + all dials.
- [`scripts/PI_RUNBOOK.md`](scripts/PI_RUNBOOK.md), build / deploy / run / A‑B on a real Pi.
- `scripts/pi_bench.py`, per‑workload throughput (the hardware‑independent tokens/pass + wall‑clock tok/s).
- `scripts/lossless_gate.py`, distribution‑equivalence gate for `quality` mode.
- `scripts/relax_sweep.py`, `relax_sweep2.py`, `x86_serve.sh`, the off‑Pi (pratio × p‑min) exploration harness.
- `scripts/pi_build.sh`, `pi_deploy.sh`, cross‑build and deploy.
- `PLAN.md`, full engineering log (physics, every measurement, the dead ends).
- `third_party/llama.cpp`, patched engine (MTP + n‑gram + relaxed‑accept knobs + parallelized M=1 GEGLU +
  ARM‑NEON `bf16` dot + fast `PAD`; built with ARM **KleidiAI** matmul kernels + **LTO**; see *Kernel work*).

## Dead ends (measured, don't re‑try)

- **Q3_K_M / 2‑bit**, fewer bytes but *slower* on A76: no repacked kernel → compute‑bound at ~7.8 GB/s, and
  the quality drop isn't worth it for E2B.
- **n_max > 3**, spills the 4‑row tile, re‑streams weights, nets slower.
- **tree‑draft on the Pi**, adds verify rows that spill the tile (−14%); it only wins on fast‑memory hosts.
- **relaxed acceptance for general text**, a weak lever (+17% max for a large quality drop). The earlier
  "~13.5 tok/s" claim for it was a measurement artifact; real is ~11. It's in `fast`/`turbo` for those who want it.
- **arm overclock for general decode**, the verify is bandwidth‑bound, so it barely moves general tok/s.

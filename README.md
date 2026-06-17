# Mote

*An open community project for running capable language models on small, common hardware: the Intelligence
of Things. The Raspberry Pi 5 is the first target; the techniques carry to other Arm CPUs. Mote is independent
and not affiliated with the developers of Gemma.*

Mote runs Google's **Gemma-4-E2B**, an effective 2.3B-parameter model, on a **Raspberry Pi 5** at **11 tokens
per second** on general text and up to 20 on repetitive output, on four Arm Cortex-A76 cores with 8 GB of RAM
and no accelerator. The engine is a tuned `llama.cpp` driven by Gemma's own multi-token-prediction head for
speculative decoding, with hand-written A76 kernels and Arm KleidiAI underneath.

## Quick start

On a Raspberry Pi 5 running 64-bit Raspberry Pi OS, one command installs a prebuilt server, downloads the
model, and adds a `gemma` command. Nothing builds on the Pi.

```bash
curl -fsSL https://github.com/mkturkcan/mote/releases/latest/download/install.sh \
  | MOTE_PKG_URL=https://github.com/mkturkcan/mote/releases/latest/download/gemma-pi5.tar.gz bash
```

Then:

```bash
gemma start            # start the server (quality mode; also: gemma start fast | turbo)
gemma chat             # chat in the terminal
gemma bench            # measure tokens per second
gemma stop
```

Or open `http://<pi-ip>:8080` in any browser for the built-in chat UI. The server speaks the OpenAI API on
port 8080. If `gemma` is not found right after install, run `source ~/.bashrc`.

## Installation

### Requirements

- Raspberry Pi 5 (BCM2712, Cortex-A76), 8 GB recommended.
- 64-bit Raspberry Pi OS, bookworm or newer (glibc 2.36 or later).
- About 4 GB of free disk for the model files.

### Prebuilt release (recommended)

The Quick start command is the whole install. It fetches `install.sh` and a self-contained package from the
latest GitHub release (the A76 `llama-server` binary, its shared libraries, and the launcher scripts),
downloads the Q4_0 model and the draft head from Hugging Face, links a `gemma` command into `~/.local/bin`,
and applies the CPU-governor and hugepage tuning. It is safe to re-run; finished steps are skipped. Everything
lands in `~/mote`.

### Build from source

To compile the server on the Pi instead of using the prebuilt binary, clone the repo and run the installer
with no package URL:

```bash
git clone https://github.com/mkturkcan/mote && cd mote
bash scripts/install.sh
```

This builds `llama.cpp` with Arm KleidiAI, link-time optimization, and the A76 kernels in
[`third_party/llama.cpp`](third_party/llama.cpp), which takes roughly 10 to 20 minutes on a Pi 5, then
downloads the model and installs `gemma` as above.

### Choosing a mode

`gemma start` takes a mode:

- **quality** (default). Every emitted token is a true argmax of the full model, so the output is the model's
  own, just produced faster. Use it whenever the answer matters.
- **fast**. Accepts draft tokens that are near the model's top choice rather than exactly it. About 17% faster
  on general text at a real quality cost: on prose roughly half the tokens diverge from the greedy path. Good
  for drafts and chat.
- **turbo**. `fast` plus an n-gram drafter that replays long exact repeats from the context. Worth it only for
  highly repetitive output such as logs or CSV, where it reaches the high teens. It slightly hurts prose and
  structured output, so it is not a general default.

### The gemma command

```
gemma start [quality|fast|turbo]   start the server
gemma chat                         chat in the terminal
gemma ui                           print the browser chat URL
gemma bench                        measure throughput
gemma status | stop
```

Power users can drive the launcher directly: `MODE=fast bash scripts/run_pi5.sh launch`.

## Results

Measured on a Raspberry Pi 5 with the Q4_0 weights and four threads at 2.4 GHz, active-cooled. Throughput is
tokens per second; reproduce with `scripts/pi_bench.py`.

| mode | general | structured | boilerplate |
| --- | :---: | :---: | :---: |
| stock GGUF, single-stream | 6.6 | 6.6 | 6.6 |
| **quality** (default) | **11.2** | 13.4 | 16.0 |
| fast | 12.5 | 13.9 | 17.2 |
| turbo | 12.3 | 13.4 | 20.4 |

The baseline is the same Q4_0 model served by a stock GGUF library (`llama-cpp-python`, single-stream, no
speculative decoding), which is what most people run. Mote's speculative decoding, A76 kernels, and system
tuning make quality mode 1.7x faster on general text, 2.0x on structured output such as JSON and code, and 2.4x
on repetitive output, and quality mode stays distribution-exact, so that speedup costs no quality. The naive
PyTorch path is slower still; its fp16 weights do not fit in the Pi's 8 GB.

Workloads: **general** includes prose, **structured** is JSON, SQL, CSV, and code, and **boilerplate** is logs
and repeated lines.

## Configuration

`run_pi5.sh` reads these environment variables; the `MODE` presets set them for you.

| variable | default | effect |
|---|---|---|
| `MODE` | `quality` | preset: `quality`, `fast`, or `turbo` |
| `NMAX` | `3` | MTP draft depth; 3 fills the A76 four-row tile and is the optimum |
| `TARGET` | `Q4_0` | weight file; Q4_0 is fastest on the A76 |
| `RELAX_PRATIO` | `0` | accept a non-argmax draft token if its probability is at least this fraction of the top token; 0 is exact |
| `PMIN` | `0.5` | draft confidence gate; lower lets the draft propose deeper |
| `NGRAM` | `0` | add the n-gram drafter, which helps only on repetitive output |
| `TREE_K` | `1` | tree-draft (multi-hypothesis); opt-in, and slower on this Pi |

## The model

Mote runs Gemma-4-E2B in Q4_0 GGUF form, with the model's multi-token-prediction head as the draft. It keeps
2.3B effective parameters and 5.1B in total; the extra mass is a per-layer embedding table read as a lookup
rather than multiplied, so it streams from storage instead of occupying working memory. The text-only GGUF
loads as about 4.65B parameters, since it carries that table but drops the vision and audio encoders. The files
are at [unsloth/gemma-4-E2B-it-GGUF](https://huggingface.co/unsloth/gemma-4-E2B-it-GGUF): the Q4_0 target is
2.9 GB and the draft head is 94 MB, and the installer fetches both. Use of the model is governed by the
[Gemma Terms of Use](https://ai.google.dev/gemma/terms); Mote's own code is Apache 2.0.

---

The rest of this document is for the curious: how the speedup is built, what was tried and abandoned, and how
to build and release it yourself.

## How it works

CPU decode is bound by memory bandwidth. Each token streams the entire weight set through RAM once, so
`tokens/sec ≈ sustained_bandwidth / bytes_per_token`. Everything here either amortizes that read or removes a
stall the stock engine leaves on the A76.

**Speculative decoding** is the main lever. Gemma-4-E2B ships a 94 MB multi-token-prediction head that drafts
tokens ahead; the full model verifies the whole draft in one batched pass and keeps the tokens that match its
own greedy output. One weight read then yields several tokens instead of one.

**Fill the four-row tile.** The A76 int8 matmul reads the weights once per group of four output rows. The
verify batch is one sampled token plus `n_max` drafts, so `n_max = 3` makes a batch of four that lands on
exactly one tile: a single pass over the weights returns four positions. A fourth draft widens the batch to
five and spills a row into a second full pass over the matrix, which costs more than the token it buys.
Measured draft depth: `n_max` 2 gives 6.3, 3 gives 9.4, 4 gives 7.9 tokens per second. Choosing Q4_0 over
Q4_K_M adds a little more, because its flat per-32 dequant runs near the A76's SDOT peak.

**Cortex-A76 kernels.** Profiling a quality-mode pass found the engine near-optimal except where Gemma's
MatFormer scaffolding hit generic, unvectorized code on the A76:

- A NEON `bf16` dot product (`ggml_vec_dot_bf16`). The A76 has no bf16 dot instruction, so the per-layer
  projection weights, read every pass, ran a scalar loop; upstream had AVX and POWER paths but none for NEON.
  The NEON path mirrors the AVX2 arithmetic exactly and saves about 16 ms per pass.
- A fast logits `PAD` (`ggml_compute_forward_pad_f32`). The per-step pad over the 256K-token vocabulary was a
  scalar, per-element, bounds-checked copy; it is now a row-wise memcpy parallelized across the vocabulary,
  bit-identical to the original.

**Arm KleidiAI.** Decode is more than 80% weight matmul. The in-tree `q4_0_4x4` GEMM is hand-written A76
assembly already near the SDOT peak, but Arm's KleidiAI `neon_dotprod` microkernel, which handles exactly these
Q4_0 weights, beats it by about 8%. Enabling it (`-DGGML_CPU_KLEIDIAI=ON`) requires `-march=armv8.2-a+dotprod+fp16`
in the toolchain, because KleidiAI's build gates the dot-product kernels on a literal `+dotprod` in the arch
flags.

**Link-time optimization** (`-DGGML_LTO=ON`) inlines the dequant and dot-product inner loops across file
boundaries and adds about 1%.

Together the kernel and build work takes general decode from 9.4 to 11.3 tokens per second, all
distribution-exact. The remaining bottleneck is the bandwidth wall, which only an overclock (firmware-locked)
or requantization (quality loss) can move.

**"Lossless" means distribution-exact.** Speculative decoding is not byte-identical to single-stream decode:
transformer inference is not batch-invariant, so the multi-token verify batch rounds floating point differently
than a batch of one and can flip a near-tie argmax onto a different but equally valid greedy path. The precise
guarantee is therefore distribution-exact, KL near zero: every emitted token is a true argmax of the target.
Quality mode holds to that by construction, and `scripts/lossless_gate.py` is its regression test.

**Why 15 tokens per second is not reachable for general text here.** Two walls cap it. The verify read is
bandwidth-bound, which fixes the pass rate at about 4.2 per second; an Arm overclock does not move it because
the Pi 5's LPDDR4X timing is fixed. And the draft can only predict so far: even accepting every near-miss, the
MTP head's second- and third-token accuracy caps accepted tokens per pass near 2.6 of a possible 4. That
product is about 11, not 15. Reaching 15 would need a structurally better draft or a smaller draft-compatible
target, neither of which exists for this model. The high teens are reachable only on repetitive output, which
is what turbo targets.

## Dead ends (measured, do not retry)

- **Q3_K_M and 2-bit.** Fewer bytes but slower on the A76: with no repacked kernel it goes compute-bound, and
  the quality drop is not worth it for E2B.
- **n_max above 3.** Spills the four-row tile, re-streams the weights, and nets slower.
- **Tree-draft on the Pi.** Adds verify rows that spill the tile and costs about 14%; it only wins on
  faster-memory hosts.
- **Relaxed acceptance for general text.** A weak lever, 17% at most for a large quality drop, so it lives in
  `fast` and `turbo` for those who want it.
- **Arm overclock for general decode.** The verify is bandwidth-bound, so it barely moves general throughput.

## For maintainers

The Pi runs a cross-compiled, self-contained package and builds nothing itself. From an x86 machine:

```bash
export PI_HOST=user@<pi-ip>
./scripts/pi_build.sh                              # cross-compile llama-server for Cortex-A76
PI_HOST=$PI_HOST ./scripts/pi_deploy.sh --models   # push the binary, scripts, and GGUFs
```

To cut a release, build the self-contained tarball, then attach it and `install.sh` to a GitHub release:

```bash
bash scripts/package_release.sh        # builds the A76 binary and writes dist/gemma-pi5.tar.gz
gh release create v1.0 dist/gemma-pi5.tar.gz scripts/install.sh -t v1.0 -n "Prebuilt Pi 5 package"
```

The Quick start one-liner then works for anyone. Full notes are in [`docs/RELEASE.md`](docs/RELEASE.md). The
fork point and the changes Mote makes to the engine are documented in
[`third_party/llama.cpp/MOTE.md`](third_party/llama.cpp/MOTE.md), with the isolated diff against upstream in
[`patches/mote-llama.cpp.patch`](patches/mote-llama.cpp.patch).

## Repository layout

- [`scripts/`](scripts/), the installer, the `gemma` CLI, the `run_pi5.sh` launcher, the cross-build and
  deploy scripts, the benchmark, and the release packager.
- [`third_party/llama.cpp/`](third_party/llama.cpp/), the patched engine; see
  [`MOTE.md`](third_party/llama.cpp/MOTE.md).
- [`patches/`](patches/), the isolated diff against upstream llama.cpp.
- [`web/`](web/), the project page.

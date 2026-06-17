# Mote's llama.cpp fork

This directory is a vendored, patched copy of llama.cpp. The full source is committed here as plain
files. It is not a git submodule and not a reference to an external repository, so building Mote builds
these sources directly, with the changes below already in place.

## Fork point

Forked from [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) at commit
`5f04dc7ac3f40271f60105574be0f617887195bb`.

Upstream llama.cpp is MIT licensed. Mote's changes are released under the same terms.

## What Mote changed

999 insertions across 16 files. The isolated diff against the fork point is kept at
[`patches/mote-llama.cpp.patch`](../../patches/mote-llama.cpp.patch), both for review and for rebasing
onto a newer upstream later.

**Cortex-A76 CPU kernels** (`ggml/src/ggml-cpu/`)
- `vec.cpp`, `vec.h`: a NEON bf16 dot product (`ggml_vec_dot_bf16`) for the model's bf16 projection
  weights, which upstream runs through a scalar loop on Arm, plus a NEON path for the final-logit softcap.
- `ops.cpp`: a fast path for the per-step logits pad (`ggml_compute_forward_pad_f32`) over the
  256K-token vocabulary, replacing a scalar per-element bounds-checked copy with a row-wise memcpy
  parallelized across the vocabulary.
- `repack.cpp`, `repack.h`, `arch/arm/repack.cpp`, `arch-fallback.h`: the q4_0_4x4 tile repack path for
  the A76's int8 dot-product matmul.
- `ggml-cpu.c`: a built-in per-op profiler, enabled with `GGML_OP_PROFILE=1`, that dumps a time
  breakdown by op at exit.

**MTP speculative decoding** (`common/`)
- `speculative.cpp`, `speculative.h`: draft-depth control tuned so the verify batch fills exactly one
  four-row weight-read tile, plus the verify batching that goes with it.
- `sampling.cpp`, `arg.cpp`, `common.cpp`, `common.h`: the quality, fast, and turbo acceptance settings
  and the command-line knobs that drive them.

**Server** (`tools/server/`)
- `server-context.cpp`: wires the MTP draft head and the three modes into the server's decode loop.

`examples/ffn-sparsity/` is an exploratory FFN-sparsity probe wired through `examples/CMakeLists.txt`. It
is not part of the shipped decode path.

KleidiAI and link-time optimization are build-flag changes (`-DGGML_CPU_KLEIDIAI=ON`, `-DGGML_LTO=ON`)
set in the Mote build scripts, not source edits, so they do not appear in the patch.

## Updating upstream later

To move to a newer llama.cpp: clone upstream at the new commit, apply `patches/mote-llama.cpp.patch`,
resolve conflicts, replace this directory with the result, and record the new base commit here.

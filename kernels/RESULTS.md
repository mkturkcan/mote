# 2-bit FFN weight packing + UDOT cvt-fold matvec — results

Validation of the bandwidth-saving kernel idea for **Gemma 4 E2B** FFN matrices,
targeting Raspberry Pi 5 (Cortex-A76, Armv8.2-A: NEON + FEAT_DotProd UDOT/SDOT,
**no i8mm, no SVE**).

## What was built

- `dump_weights.py` — dequantizes 3 real FFN tensors from the Q4_0 GGUF to fp32
  raw blobs (`data/*.f32` + `data/manifest.txt`). Tensors used:
  - `blk.20.ffn_gate.weight` — out=12288, in=1536
  - `blk.20.ffn_up.weight`   — out=12288, in=1536
  - `blk.20.ffn_down.weight` — out=1536,  in=12288
- `bench.c` — self-contained microbenchmark (no deps but libm). Packs each tensor
  in 4 formats, runs `y = W @ x`, and reports recon error, output error,
  bytes/weight, and NEON-vs-scalar correctness.
- `Makefile` — builds `bench_x86` (native gcc, scalar path) and `bench_arm`
  (cross `aarch64-linux-gnu-gcc -mcpu=cortex-a76`, static), runs the latter under
  `qemu-aarch64-static`.

## Formats

| Tag | Description | Dequant | Hot loop |
|-----|-------------|---------|----------|
| **REF** | fp32 W @ fp32 x (ground truth) | — | fp32 |
| **Q40** | Q4_0-style: 4-bit symmetric, per-32 fp16 scale | (nib−8)·s | UDOT int8 |
| **U2**  | uniform 2-bit symmetric, per-32 fp16 scale, levels {−1.5,−.5,.5,1.5}·s | affine "cvt-fold" | UDOT int8 |
| **CB2** | per-group 4-level codebook (Lloyd-Max MSE fit), 4 fp16 levels/group | table | bucket-sum fp |
| **CBT** | **per-tensor** shared 4-level codebook + per-32 fp16 scale | level·s | UDOT int8 |

The **cvt-fold** structure: the int kernels quantize `x` once to int8 (per-group
scale), accumulate integer UDOT dot products group-by-group, and the entire
weight dequant collapses into **one fp multiply per group** (`isum * w_scale *
x_scale`) — off the hot path. CB2 uses arbitrary per-group fp levels so it cannot
use UDOT; it instead buckets x by 2-bit index (4 adds) then dots with 4 levels.
CBT quantizes its 4 shared levels to int8 once, so it *is* a UDOT kernel.

## Measured numbers (identical on x86 scalar and aarch64 UDOT)

Bytes/weight is purely structural (the bandwidth metric; fp32 = 4.0):

| Format | bytes/weight | compression vs fp32 |
|--------|-------------:|--------------------:|
| Q40 | 0.5625 | 7.1× |
| U2  | 0.3125 | 12.8× |
| CB2 | 0.5000 | 8.0× |
| CBT | 0.3125 | 12.8× |

Per-tensor errors (W recon RMSE relative to the tensor's own RMS, and end-to-end
output relative error of `y` averaged over 4 x-vectors, mixing Gaussian-ish and
spiky/heavy-tailed activations):

### gate [12288×1536], W_rms=0.0332
| Format | W RMSE/W_rms | out rel-err |
|--------|-------------:|------------:|
| Q40 | 0.0549 | **0.0573** |
| U2  | 0.4803 | 0.5045 |
| CB2 | 0.2931 | 0.2959 |
| CBT | 0.4103 | 0.4179 |

### up [12288×1536], W_rms=0.0335
| Format | W RMSE/W_rms | out rel-err |
|--------|-------------:|------------:|
| Q40 | 0.0543 | **0.0545** |
| U2  | 0.4744 | 0.4652 |
| CB2 | 0.2924 | 0.2918 |
| CBT | 0.4060 | 0.4050 |

### down [1536×12288], W_rms=0.0239
| Format | W RMSE/W_rms | out rel-err |
|--------|-------------:|------------:|
| Q40 | 0.0557 | **0.0561** |
| U2  | 0.4901 | 0.4943 |
| CB2 | 0.2952 | 0.2938 |
| CBT | 0.4131 | 0.4106 |

### NEON UDOT vs scalar correctness (aarch64 under qemu)
Max abs diff between the NEON `vdotq_s32` path and a scalar recompute, for all
three integer kernels, on every tensor:

```
NEON-vs-scalar max abs diff:  Q40=0.000e+00  U2=0.000e+00  CBT=0.000e+00  [PASS]
```

Bit-exact (the int32 dot-product accumulation is integer, so NEON and scalar
agree exactly; the single per-group fp rescale is identical). The fp16 path,
packing, and matvec all run correctly on the real Armv8.2-A ISA.

> QEMU gives **correctness only**, not representative Pi timing — no timings are
> reported as Pi performance.

## Key findings — does 2-bit work for FFN?

1. **4-bit (Q4_0) is excellent**: ~5.5% output relative error at 0.56 B/w. The
   weights are well-behaved (max|w|≈0.6, RMS≈0.03), so 4-bit symmetric loses
   almost nothing.

2. **Uniform 2-bit is not usable as-is**: ~46–50% output relative error. A flat
   4-level grid simply cannot represent the bell-shaped weight distribution — the
   error is ~9× worse than 4-bit. Cutting bytes from 0.56→0.31 costs an order of
   magnitude in accuracy.

3. **A learned codebook materially beats uniform 2-bit at the same bandwidth.**
   At identical 0.3125 B/w, CBT (per-tensor 4-level Lloyd-Max shape) gives
   ~0.40–0.42 vs U2's ~0.46–0.50 — roughly a **15–20% relative-error reduction
   for free** (same bytes, same UDOT kernel). The fitted shape is consistently
   asymmetric/non-uniform, e.g. `[-0.87, -0.21, 0.36, 1.00]`, reflecting the
   weight distribution. So **yes, the codebook helps**, confirming the VQ
   direction — but it does not rescue 2-bit on its own.

4. **Per-group codebooks (CB2) are the strongest 2-bit recon (~29% out-err)** but
   at 0.50 B/w they barely beat Q4_0's 0.56 B/w while being ~5× *worse* in error —
   storing 4 fp16 levels per 32 weights destroys the bandwidth advantage. Not a
   good operating point.

5. **Bottom line for FFN**: plain/codebook 2-bit at ~0.31 B/w still introduces
   **40–50% output error per matvec** — far too lossy to use directly for FFN
   gate/up/down, whereas 4-bit is ~5%. The codebook idea is real and worth ~15–20%
   at equal bytes, but to make 2-bit viable you'd need more than a static 4-level
   shape: e.g. mixed precision (keep a few salient channels/columns at higher
   bits), per-group codebooks with a *shared* (amortized) codebook table rather
   than per-group fp16 levels, larger effective codebooks (e.g. 2-bit indices into
   a per-channel 4-entry table chosen from a shared set), or activation-aware
   (GPTQ/AWQ-style) calibration that minimizes output error rather than weight
   MSE. The cvt-fold UDOT kernel mechanics are proven correct on the A76 ISA and
   are format-agnostic, so any of those refinements drop straight into the same
   matvec.

## Reproduce

```sh
cd /home/mkt2126/cpullm/kernels
python3 dump_weights.py        # dumps real FFN weights -> data/
make bench_x86 && ./bench_x86 data                       # native scalar
make bench_arm && qemu-aarch64-static ./bench_arm data   # aarch64 NEON UDOT
# or: make run
```

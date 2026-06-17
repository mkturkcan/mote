#!/usr/bin/env python3
"""RPi5 (Cortex-A76 / BCM2712) roofline projector for Gemma 4 E2B CPU decode.

Grounded in measured tensor data from gemma-4-E2B-it-Q4_0.gguf:
  streamed/token ≈ 1383 MB  (ffn 933, lm_head/token_embd 226, attn 167, other 55, norm ~1)
  PLE/aux lookup ≈ 1642 MB  (NOT streamed per token — looked up by token id)
Decode is bandwidth-bound: tok/s ≈ sustained_BW / streamed_bytes_per_token.
Speculative decoding amortizes one target weight-read over `accept` emitted tokens.
"""

# ---- measured per-token streamed components (MB), E2B Q4_0 ----
COMP = {"ffn": 933.5, "lm_head": 226.5, "attn": 167.5, "other": 55.1, "norm": 0.7}
BASE = sum(COMP.values())  # 1383 MB

# ---- RPi5 bandwidth scenarios (GB/s sustained, single stream) ----
BW = {"stock 2.4GHz (~12)": 12.0, "tuned+hugepages (~13.5)": 13.5, "mem OC ~15%": 15.5}

def ceiling(stream_mb, bw):  # single forward pass tok/s
    return bw * 1000.0 / stream_mb

def scenario(name, ffn_bits=4.5, lmhead_bits=4.5, accept=1.0, bw=12.0):
    """ffn/lmhead_bits: effective bits/weight after requant (Q4_0≈4.5). accept: spec tokens/pass."""
    ffn = COMP["ffn"] * ffn_bits / 4.5
    lm  = COMP["lm_head"] * lmhead_bits / 4.5
    stream = ffn + lm + COMP["attn"] + COMP["other"] + COMP["norm"]
    single = ceiling(stream, bw)
    eff = single * accept
    print(f"  {name:46s} stream={stream:6.0f}MB  single={single:5.1f}  x{accept:.1f} -> {eff:6.1f} tok/s")
    return eff

if __name__ == "__main__":
    print(f"Measured baseline streamed/token: {BASE:.0f} MB  (PLE lookup 1642 MB excluded)\n")
    print("Single-pass ceilings (no speculation):")
    for n, bw in BW.items():
        print(f"  {n:28s} {ceiling(BASE, bw):5.1f} tok/s")
    print("\nStacked scenarios (E2B quality, lossless spec decoding):")
    scenario("baseline Q4_0, stock", 4.5, 4.5, 1.0, 12.0)
    scenario("+tuned bus (hugepage/prefetch/4core)", 4.5, 4.5, 1.0, 13.5)
    scenario("+FFN VQ~2.8b +lmhead Q3.5 +mem OC", 2.8, 3.5, 1.0, 15.5)
    print("  --- add speculative decoding (MTP draft, measured acceptance) ---")
    scenario("+spec accept 2.5x (conservative)", 2.8, 3.5, 2.5, 15.5)
    scenario("+spec accept 3.0x (MTP typical)", 2.8, 3.5, 3.0, 15.5)
    scenario("+tree-draft accept 4.0x (stretch)", 2.8, 3.5, 4.0, 15.5)
    print("\nNote: acceptance is measured empirically (hardware-independent); fill in real value.")

# ---- appended: 2-bit campaign scenarios (informed by the WebGPU journey) ----
def campaign():
    print("\n=== 2-bit campaign (E2B tolerates 2-bit FFN per WebGPU run) ===")
    # bytes/token if FFN+lm_head+attn pushed toward ~2.2 effective bits, PLE-proj int8
    scenario("Q4_0 baseline (measured x86=25.8 t/s @~35GB/s)", 4.5,4.5,1.0,12.0)
    scenario("2-bit FFN, Q4 lm_head/attn, tuned bus", 2.2,4.5,1.0,13.5)
    scenario("2-bit FFN+lm_head, VQ attn, mem OC", 2.2,2.5,1.0,15.5)
    print("  --- x speculative (MTP) ---")
    scenario("2-bit all + spec 2.5x", 2.2,2.5,2.5,15.5)
    scenario("2-bit all + spec 3.0x", 2.2,2.5,3.0,15.5)
    scenario("2-bit all + tree-draft 4.0x (stretch->100)", 2.2,2.5,4.0,15.5)
campaign()

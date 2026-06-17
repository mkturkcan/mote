#!/usr/bin/env python3
"""Dump real FFN weight tensors from the Gemma 4 E2B Q4_0 GGUF to raw fp32 binaries.

The GGUF stores Q4_0 weights. We dequantize to fp32 (ground truth for the C
benchmark) and dump each tensor as a flat row-major [out, in] fp32 blob plus a
tiny text header describing (out, in). The C microbenchmark reads these.

Layout note: gguf.quants.dequantize returns numpy shape [out, in] (rows = output
features, cols = input features), which is exactly the W[out,in] layout the C
matvec wants for y = W @ x.
"""
import os
import sys
import struct
import gguf
import numpy as np

GGUF_PATH = os.environ.get(
    "GGUF_PATH",
    "/home/mkt2126/cpullm/models/gemma-4-E2B-it-Q4_0.gguf",
)
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(OUT_DIR, "data")

TENSORS = [
    "blk.20.ffn_gate.weight",
    "blk.20.ffn_up.weight",
    "blk.20.ffn_down.weight",
]


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    r = gguf.GGUFReader(GGUF_PATH)
    by_name = {t.name: t for t in r.tensors}
    manifest = []
    for name in TENSORS:
        if name not in by_name:
            print(f"MISSING tensor {name}", file=sys.stderr)
            sys.exit(1)
        t = by_name[name]
        d = gguf.quants.dequantize(t.data, t.tensor_type).astype(np.float32)
        assert d.ndim == 2, f"{name} not 2D: {d.shape}"
        out, inn = d.shape
        short = name.replace("blk.20.ffn_", "").replace(".weight", "")
        bin_path = os.path.join(DATA_DIR, f"{short}.f32")
        d.tofile(bin_path)
        manifest.append((short, out, inn, bin_path))
        print(f"{name}: out={out} in={inn} -> {bin_path} "
              f"({d.nbytes/1e6:.1f} MB) range[{d.min():.4f},{d.max():.4f}]")

    # Write a manifest the C code parses: one line per tensor "short out in path"
    with open(os.path.join(DATA_DIR, "manifest.txt"), "w") as f:
        for short, out, inn, path in manifest:
            f.write(f"{short} {out} {inn} {os.path.basename(path)}\n")
    print("wrote manifest:", os.path.join(DATA_DIR, "manifest.txt"))


if __name__ == "__main__":
    main()

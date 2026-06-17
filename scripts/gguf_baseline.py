#!/usr/bin/env python3
"""'Dumb' baseline: the same Q4_0 GGUF run through a stock GGUF library (llama-cpp-python), single-stream,
NO speculative decoding, default build — i.e. none of this project's work (no MTP, no tuned kernels, no
tile-fill). This is roughly what someone gets when they 'just run the GGUF', to contrast with the tuned stack.

Usage:  python gguf_baseline.py <model.gguf> [n_tokens]
"""
import sys, time
from llama_cpp import Llama, llama_print_system_info

mp = sys.argv[1]
N = int(sys.argv[2]) if len(sys.argv) > 2 else 64

llm = Llama(model_path=mp, n_ctx=2048, n_threads=4, n_gpu_layers=0, verbose=False)
prompt = "Explain how a hash map works and its average time complexity."

# warmup (prompt eval + first tokens, not timed)
llm(prompt, max_tokens=4, temperature=0.0)

t0 = time.time()
out = llm(prompt, max_tokens=N, temperature=0.0)
dt = time.time() - t0
n = out["usage"]["completion_tokens"]
print(f"\n==== stock llama-cpp-python (single-stream, no spec-decode): {n/dt:.2f} tok/s  ({n} tok in {dt:.1f}s) ====")
print("sample:", out["choices"][0]["text"].strip()[:110])

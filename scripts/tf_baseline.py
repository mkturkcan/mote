#!/usr/bin/env python3
"""Naive-baseline benchmark: gemma-4-E2B via HuggingFace transformers on CPU (what most people first try).
Loads the model in PyTorch and times greedy generate(). On a Pi 5 (8 GB) the fp16 weights (~9 GB) don't fit,
so this leans on swap — the result is the honest 'naive user' experience to contrast with the tuned llama.cpp stack.

Usage:  python tf_baseline.py <gguf_path | hf_repo>  [n_tokens]
"""
import sys, time, os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.set_num_threads(int(os.environ.get("NTHREADS", "4")))
src = sys.argv[1]
N = int(sys.argv[2]) if len(sys.argv) > 2 else 8

print(f"torch {torch.__version__}  threads={torch.get_num_threads()}  source={src}", flush=True)
load_kw = dict(torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
if src.endswith(".gguf"):
    d, fn = os.path.dirname(src) or ".", os.path.basename(src)
    tok = AutoTokenizer.from_pretrained(d, gguf_file=fn)
    load_kw["gguf_file"] = fn
    model_id = d
else:
    tok = AutoTokenizer.from_pretrained(src)
    model_id = src

t0 = time.time()
model = AutoModelForCausalLM.from_pretrained(model_id, **load_kw)
model.eval()
print(f"load: {time.time()-t0:.1f}s  (RSS now ~{int(open('/proc/self/status').read().split('VmRSS:')[1].split()[0])//1024} MB)", flush=True)

msgs = [{"role": "user", "content": "Explain how a hash map works and its average time complexity."}]
ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt")

print("warmup (1 token, includes graph build) ...", flush=True)
t0 = time.time()
with torch.no_grad():
    model.generate(ids, max_new_tokens=1, do_sample=False)
print(f"  first-token latency: {time.time()-t0:.1f}s", flush=True)

print(f"timing {N} tokens ...", flush=True)
t0 = time.time()
with torch.no_grad():
    out = model.generate(ids, max_new_tokens=N, do_sample=False)
dt = time.time() - t0
print(f"\n==== transformers (PyTorch CPU) baseline: {N/dt:.4f} tok/s  ({dt:.1f}s for {N} tokens) ====", flush=True)
print("sample:", tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)[:120], flush=True)

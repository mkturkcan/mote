#!/usr/bin/env python3
"""Capture greedy quality-mode outputs from a server (for old-vs-new-binary A/B), or diff two captures.

  capture: cmp_quality.py capture <host> <port> <out.json>
  diff:    cmp_quality.py diff <a.json> <b.json>

Used to verify a kernel change (bf16 NEON path) is distribution-equivalent: same hardware, greedy, only the
binary differs. Expect near-identical text (tiny f32-rounding token flips at most). Garbage => kernel bug.
"""
import sys, json, urllib.request, difflib

PROMPTS = [
    "Write a paragraph about the history of the Roman Empire.",
    "Explain how a hash map works and its average time complexity.",
    "Write a Python function that computes the nth Fibonacci number iteratively.",
    "What is the capital of France, and name two famous landmarks there?",
    "Describe the water cycle in detail.",
    "Implement binary search in Python with comments.",
]

def gen(host, port, p, n=96):
    body = json.dumps({"prompt": p, "n_predict": n, "temperature": 0, "cache_prompt": False}).encode()
    r = json.load(urllib.request.urlopen(urllib.request.Request(
        f"http://{host}:{port}/completion", body, {"Content-Type": "application/json"}), timeout=120))
    return r["content"]

if sys.argv[1] == "capture":
    host, port, out = sys.argv[2], sys.argv[3], sys.argv[4]
    data = {p: gen(host, port, p) for p in PROMPTS}
    json.dump(data, open(out, "w"), indent=2)
    print(f"captured {len(data)} prompts -> {out}")
    for p in PROMPTS:
        print(f"  [{p[:38]!r}] {data[p][:70]!r}")
elif sys.argv[1] == "diff":
    a = json.load(open(sys.argv[2])); b = json.load(open(sys.argv[3]))
    tot = 0.0
    for p in a:
        s = difflib.SequenceMatcher(None, a[p], b[p]).ratio(); tot += s
        flag = "" if s > 0.98 else ("  <-- DIVERGES" if s < 0.8 else "  (minor)")
        print(f"  sim={s:.4f}  {p[:46]!r}{flag}")
    print(f"\nAVG similarity old-vs-new = {tot/len(a):.4f}  (1.0=identical; >0.98=distribution-exact; low=bug)")

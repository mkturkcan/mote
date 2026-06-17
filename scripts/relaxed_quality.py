#!/usr/bin/env python3
"""Quality gate for lossy/relaxed speculative acceptance (run OFF-Pi, on the x86 dev box).

Compares a CANDIDATE server (relaxed acceptance) against a pure-target greedy REFERENCE on prose/factual
prompts. Reports SequenceMatcher similarity (1.0 = identical to greedy = no loss; lower = more deviation)
and prints the candidate text so coherence/correctness can be eyeballed. Quality is hardware-independent,
so this runs on x86, never the Pi.

Usage: relaxed_quality.py <ref_port> <cand_port> [n_predict]
"""
import sys, json, urllib.request, difflib

PROMPTS = [
    "Write a paragraph about the history of the Roman Empire.",
    "Explain how a hash map works and its average time complexity.",
    "What is the capital of France, and name two famous landmarks there?",
    "Summarize the plot of Romeo and Juliet in three sentences.",
    "Explain the difference between TCP and UDP.",
    "Describe the water cycle.",
    "Give three tips for writing clear technical documentation.",
    "What causes the seasons on Earth?",
]

def gen(port, p, n):
    b = json.dumps({"prompt": p, "n_predict": n, "temperature": 0, "cache_prompt": False}).encode()
    r = urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{port}/completion", b, {"Content-Type":"application/json"}), timeout=180)
    return json.load(r)["content"]

def main():
    ref_port, cand_port = sys.argv[1], sys.argv[2]
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 96
    tot = 0.0
    for i, p in enumerate(PROMPTS):
        ref = gen(ref_port, p, n); cand = gen(cand_port, p, n)
        ratio = difflib.SequenceMatcher(None, ref, cand).ratio()
        tot += ratio
        print(f"[sim={ratio:.2f}] {p[:42]!r}")
        print(f"   cand: {cand[:110]!r}")
    print(f"\nAVG similarity-to-greedy = {tot/len(PROMPTS):.3f}  (1.0=lossless; <~0.6 = heavy deviation)")

if __name__ == "__main__":
    main()

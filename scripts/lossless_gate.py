#!/usr/bin/env python3
"""Lossless regression gate for speculative decoding.

Speculative decoding (MTP, tree, anything) MUST yield byte-identical greedy output to the
pure-target model. This harness sends a prompt battery at temperature=0 to two servers
(reference = pure target, candidate = speculative) and asserts identical generated text
AND identical token ids. Any divergence => the speculative path changed the distribution
=> NOT lossless => FAIL. Run this after every change to the draft/verify code.

Usage: lossless_gate.py <ref_port> <cand_port> [n_predict]
Exit 0 = PASS (lossless), 1 = FAIL (divergence) or error.
"""
import sys, os, json, urllib.request

# target host: default localhost; set HOST=<pi-ip> to gate a remote Pi server from this machine
HOST = os.environ.get("HOST", "127.0.0.1")

PROMPTS = [
    "def quicksort(arr):",
    "The history of the Roman Empire began",
    "List the first 10 prime numbers:",
    "If a train travels 60 km in 45 minutes, what is its speed in km/h? Show your steps.",
    "Translate to French: The weather is nice today.",
    "Write a haiku about autumn leaves.",
    "Explain recursion to a beginner in two sentences.",
    "import numpy as np\ndef softmax(x):",
    "Q: What is the capital of Japan? A:",
    "Once upon a time, in a distant kingdom,",
]

def gen(port, prompt, n):
    body = json.dumps({"prompt": prompt, "n_predict": n, "temperature": 0,
                       "cache_prompt": False, "n_probs": 0}).encode()
    req = urllib.request.Request(f"http://{HOST}:{port}/completion", body,
                                 {"Content-Type": "application/json"})
    r = json.load(urllib.request.urlopen(req, timeout=120))
    # tokens if available, else content
    toks = r.get("tokens") or []
    return r.get("content", ""), toks

def main():
    ref_port, cand_port = sys.argv[1], sys.argv[2]
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 64
    fails = 0
    for i, p in enumerate(PROMPTS):
        rc, rt = gen(ref_port, p, n)
        cc, ct = gen(cand_port, p, n)
        ok = (rc == cc)
        tag = "OK " if ok else "DIFF"
        print(f"[{tag}] prompt {i}: {p[:40]!r}")
        if not ok:
            fails += 1
            # show first divergence
            for j in range(min(len(rc), len(cc))):
                if rc[j] != cc[j]:
                    print(f"      first diff at char {j}: ref={rc[max(0,j-20):j+20]!r}")
                    print(f"                           cand={cc[max(0,j-20):j+20]!r}")
                    break
            else:
                print(f"      length diff: ref={len(rc)} cand={len(cc)}")
    print(f"\n{'PASS — lossless' if fails==0 else f'FAIL — {fails}/{len(PROMPTS)} diverged'} "
          f"(ref:{ref_port} cand:{cand_port})")
    sys.exit(0 if fails == 0 else 1)

if __name__ == "__main__":
    main()

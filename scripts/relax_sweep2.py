#!/usr/bin/env python3
"""2D sweep: --spec-draft-p-min (draft depth gate) x relaxed pratio (lossy accept), on x86.

Hypothesis: relaxed acceptance was throttled by p-min=0.5, which makes the MTP draft STOP early on the very
(uncertain/prose) positions where relaxed acceptance would help. Lowering p-min lets the draft propose its full
n_max depth; relaxed acceptance can then accept those near-misses -> higher tok/pass. This sweep finds whether
the combo reaches tok/pass>=3.60 (the >=15 Pi tok/s threshold; ceiling 4.0 -> 16.7 tok/s).
"""
import json, subprocess, sys, time, urllib.request, difflib, os

ROOT = "/home/mkt2126/cpullm"
REF_PORT = 9000
RATE = 4.174
N = 96
TOPK = 8

# (pmin, pratio).  pratio=0 -> pure lossless at that pmin (isolates p-min's own effect).
GRID = []
for pmin in (0.5, 0.2, 0.1, 0.0):
    for pratio in (0.0, 0.3, 0.15, 0.08):
        GRID.append((pmin, pratio))

GEN_SUITE = [
    "Write a paragraph about the history of the Roman Empire.",
    "Write a Python function that computes the nth Fibonacci number iteratively.",
    "List the planets of the solar system in order from the sun and one fact about each.",
    "If a train travels 60 km in 45 minutes, what is its speed in km/h? Show your steps.",
    "Explain how a hash map works and its average time complexity.",
    "Write a function to reverse a linked list in C.",
    "Describe the water cycle in detail.",
    "Implement binary search in Python with comments.",
]
QUAL_SUITE = [
    "Write a paragraph about the history of the Roman Empire.",
    "Explain how a hash map works and its average time complexity.",
    "What is the capital of France, and name two famous landmarks there?",
    "Summarize the plot of Romeo and Juliet in three sentences.",
    "Explain the difference between TCP and UDP.",
    "Describe the water cycle.",
    "Give three tips for writing clear technical documentation.",
    "What causes the seasons on Earth?",
]

def gen(port, prompt, n):
    body = json.dumps({"prompt": prompt, "n_predict": n, "temperature": 0, "cache_prompt": False}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/completion", body, {"Content-Type": "application/json"})
    r = json.load(urllib.request.urlopen(req, timeout=300))
    return r["content"], r["timings"]

def wait_health(port, timeout=120):
    for _ in range(timeout):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=3); return True
        except Exception:
            time.sleep(1)
    return False

def bench_general(port):
    g = a = d = 0; ms = 0.0
    for p in GEN_SUITE:
        _, t = gen(port, p, N)
        g += t["predicted_n"]; a += t.get("draft_n_accepted", 0); d += t.get("draft_n", 0); ms += t["predicted_ms"]
    passes = g - a
    return dict(tok_pass=g/passes, accept=(a/d if d else 0), tok_s=1000*g/ms,
                draft_per_gen=d/g)

print("caching lossless reference outputs (:9000)...", flush=True)
REF = {p: gen(REF_PORT, p, N)[0] for p in QUAL_SUITE}

results = []; samples = {}
for i, (pmin, pratio) in enumerate(GRID):
    port = 9201 + i
    tag = f"pmin{pmin}_p{pratio}"
    print(f"\n[{i+1}/{len(GRID)}] {tag} on :{port} ...", flush=True)
    proc = subprocess.Popen(["setsid", "bash", f"{ROOT}/scripts/x86_serve.sh", str(port), str(pratio), str(TOPK), "3", str(pmin)],
                            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, preexec_fn=os.setpgrp)
    try:
        if not wait_health(port):
            print(f"  !! {tag} failed health"); continue
        b = bench_general(port)
        sims = []; texts = []
        for p in QUAL_SUITE:
            cand, _ = gen(port, p, N)
            sims.append(difflib.SequenceMatcher(None, REF[p], cand).ratio())
            texts.append({"prompt": p, "text": cand})
        sim = sum(sims)/len(sims)
        rec = dict(tag=tag, pmin=pmin, pratio=pratio, tok_pass=round(b["tok_pass"],3),
                   accept=round(b["accept"],3), draft_per_gen=round(b["draft_per_gen"],2),
                   x86_tok_s=round(b["tok_s"],1), pred_pi_tok_s=round(RATE*b["tok_pass"],2),
                   similarity=round(sim,3))
        results.append(rec); samples[tag] = texts
        print(f"  tok/pass={rec['tok_pass']}  accept={rec['accept']}  draft/gen={rec['draft_per_gen']}  "
              f"sim={rec['similarity']}  pred_Pi={rec['pred_pi_tok_s']}", flush=True)
    finally:
        subprocess.run(["fuser", "-k", f"{port}/tcp"], check=False,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(1)

results.sort(key=lambda r: -r["pred_pi_tok_s"])
json.dump({"rate": RATE, "results": results}, open("/tmp/relax_sweep2.json","w"), indent=2)
json.dump(samples, open("/tmp/relax_samples2.json","w"), indent=2)
print("\n==== 2D SWEEP (p-min x pratio), sorted by predicted Pi tok/s ====")
print(f"{'config':18s} {'tok/pass':>8s} {'accept':>7s} {'drft/gen':>8s} {'sim':>6s} {'pred_Pi':>8s} {'>=15':>4s}")
for r in results:
    print(f"{r['tag']:18s} {r['tok_pass']:8.3f} {r['accept']:7.3f} {r['draft_per_gen']:8.2f} "
          f"{r['similarity']:6.3f} {r['pred_pi_tok_s']:8.2f} {'YES' if r['pred_pi_tok_s']>=15 else '':>4s}")
print("\nsaved /tmp/relax_sweep2.json + /tmp/relax_samples2.json")

#!/usr/bin/env python3
"""Relaxed-acceptance grid sweep, run on the x86 dev box.

tok/pass (acceptance multiplier) and similarity-to-greedy are HARDWARE-INDEPENDENT at temp=0, so we explore
the whole (pratio, topk) grid here and PREDICT Pi tok/s = RATE * tok/pass (RATE calibrated from the Pi anchor).
Only the chosen finalists get their real tok/s confirmed on the Pi.

For each grid point: launch a candidate server, measure tok/pass + accept + x86 tok/s on the general suite,
and similarity-to-greedy (vs the lossless reference on :9000) on a prose/factual quality suite, saving the
generated text so an LLM judge can eyeball coherence. Emits a table + JSON (/tmp/relax_sweep.json) + samples.
"""
import json, subprocess, sys, time, urllib.request, difflib, os, signal

ROOT = "/home/mkt2126/cpullm"
REF_PORT = 9000
RATE = 4.174  # Pi passes/sec, from anchor: 9.57 tok/s / 2.293 tok/pass (cool, pre-throttle)
N = 96

# (pratio, topk)
GRID = [
    (0.30, 8), (0.25, 8), (0.20, 8), (0.15, 8), (0.12, 8), (0.10, 8), (0.08, 8), (0.05, 8),
    (0.20, 40), (0.15, 40), (0.12, 40), (0.10, 40), (0.07, 40), (0.05, 40),
]

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
            urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=3)
            return True
        except Exception:
            time.sleep(1)
    return False

def bench_general(port):
    g = a = d = 0; ms = 0.0
    for p in GEN_SUITE:
        _, t = gen(port, p, N)
        g += t["predicted_n"]; a += t.get("draft_n_accepted", 0); d += t.get("draft_n", 0); ms += t["predicted_ms"]
    passes = g - a
    return dict(tok_pass=g/passes, accept=(a/d if d else 0), tok_s=1000*g/ms)

# reference (lossless) quality outputs, computed once
print("caching lossless reference outputs (:9000)...", flush=True)
REF = {p: gen(REF_PORT, p, N)[0] for p in QUAL_SUITE}

results = []
samples = {}
for i, (pratio, topk) in enumerate(GRID):
    port = 9101 + i
    tag = f"p{pratio}_k{topk}"
    print(f"\n[{i+1}/{len(GRID)}] {tag} on :{port} ...", flush=True)
    proc = subprocess.Popen(["setsid", "bash", f"{ROOT}/scripts/x86_serve.sh", str(port), str(pratio), str(topk), "3"],
                            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                            preexec_fn=os.setpgrp)
    try:
        if not wait_health(port):
            print(f"  !! {tag} failed health"); continue
        b = bench_general(port)
        sims = []; texts = []
        for p in QUAL_SUITE:
            cand, _ = gen(port, p, N)
            sims.append(difflib.SequenceMatcher(None, REF[p], cand).ratio())
            texts.append({"prompt": p, "text": cand})
        sim = sum(sims) / len(sims)
        pred_pi = RATE * b["tok_pass"]
        rec = dict(tag=tag, pratio=pratio, topk=topk, tok_pass=round(b["tok_pass"], 3),
                   accept=round(b["accept"], 3), x86_tok_s=round(b["tok_s"], 1),
                   pred_pi_tok_s=round(pred_pi, 2), similarity=round(sim, 3))
        results.append(rec)
        samples[tag] = texts
        print(f"  tok/pass={rec['tok_pass']}  accept={rec['accept']}  sim={rec['similarity']}  "
              f"pred_Pi_tok/s={rec['pred_pi_tok_s']}", flush=True)
    finally:
        subprocess.run(["pkill", "-f", f"llama-server.*--port {port}"], check=False)
        time.sleep(1)

results.sort(key=lambda r: -r["pred_pi_tok_s"])
json.dump({"rate": RATE, "anchor_tok_s": 9.57, "results": results}, open("/tmp/relax_sweep.json", "w"), indent=2)
json.dump(samples, open("/tmp/relax_samples.json", "w"), indent=2)
print("\n==== SWEEP RESULTS (sorted by predicted Pi tok/s) ====")
print(f"{'config':12s} {'tok/pass':>8s} {'accept':>7s} {'sim':>6s} {'pred_Pi_t/s':>11s}  {'>=15?':>5s}")
for r in results:
    print(f"{r['tag']:12s} {r['tok_pass']:8.3f} {r['accept']:7.3f} {r['similarity']:6.3f} "
          f"{r['pred_pi_tok_s']:11.2f}  {'YES' if r['pred_pi_tok_s']>=15 else '':>5s}")
print("\nsaved /tmp/relax_sweep.json + /tmp/relax_samples.json")

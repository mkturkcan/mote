#!/usr/bin/env python3
"""Pi (or local) throughput benchmark for the speculative-decoding stack.

Reports the hardware-INDEPENDENT multiplier (tokens-per-pass = gen/(gen-accepted), i.e. accepted draft
tokens per target weight-read) AND wall-clock tok/s, over three workload suites:
  general    — mixed chat/code/reasoning prompts (the representative case)
  repetition — structured/boilerplate output (JSON/CSV/tables/enums)
  literal    — literal exact repetition (n-gram drafting's best case)

Usage:
  HOST=<ip> python3 pi_bench.py <port> [--suite general|repetition|literal|all] [--n N] [--runs R]
Default HOST=127.0.0.1 (run on the Pi, or from this machine pointed at the Pi's IP).
Compare two configs by running it against two ports/servers and diffing tok/pass + tok/s.
"""
import sys, os, json, time, argparse, urllib.request

HOST = os.environ.get("HOST", "127.0.0.1")

SUITES = {
 "general": [
  "Write a paragraph about the history of the Roman Empire.",
  "Write a Python function that computes the nth Fibonacci number iteratively.",
  "List the planets of the solar system in order from the sun and one fact about each.",
  "If a train travels 60 km in 45 minutes, what is its speed in km/h? Show your steps.",
  "Explain how a hash map works and its average time complexity.",
  "Write a function to reverse a linked list in C.",
  "Describe the water cycle in detail.",
  "Implement binary search in Python with comments.",
 ],
 "repetition": [
  "Write a Python dataclass `Config` with 20 fields, each with a type annotation and a default value, one per line.",
  "Generate a JSON array of 15 objects, each with keys id, name, email, active, and role.",
  "Write a C enum `Opcode` listing 25 instructions, each on its own line with an explicit value.",
  "Write a markdown table with 12 rows comparing programming languages across Name, Year, Paradigm, Typing.",
  "Write 20 SQL INSERT statements into a table users(id, name, email).",
  "Generate 18 lines of CSV with columns: timestamp, sensor_id, temperature, humidity.",
 ],
 "literal": [
  "Repeat this exact line 25 times, one per line: The quick brown fox jumps over the lazy dog.",
  "Print 'GET /api/v1/users HTTP/1.1' exactly 20 times, each on its own line.",
  "Output the line 'x = x + 1;  // increment counter' exactly 25 times.",
  "Write 'ALL WORK AND NO PLAY MAKES JACK A DULL BOY' 20 times, one per line.",
 ],
}

def run_prompt(port, prompt, n):
    body = json.dumps({"prompt": prompt, "n_predict": n, "temperature": 0, "cache_prompt": False}).encode()
    req = urllib.request.Request(f"http://{HOST}:{port}/completion", body, {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=300))["timings"]

def run_suite(port, name, n):
    g = a = d = 0; ms = 0.0
    for p in SUITES[name]:
        t = run_prompt(port, p, n)
        g += t["predicted_n"]; a += t.get("draft_n_accepted", 0); d += t.get("draft_n", 0); ms += t["predicted_ms"]
    passes = g - a
    return dict(gen=g, acc=a, draft=d, passes=passes,
               tok_pass=(g/passes if passes else float('inf')),
               accept=(a/d if d else 0.0),
               tok_s=(1000.0*g/ms if ms else 0.0))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("port")
    ap.add_argument("--suite", default="all", choices=["general", "repetition", "literal", "all"])
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--runs", type=int, default=1, help="repeat each suite R times (tok/s noise); reports last")
    args = ap.parse_args()
    suites = list(SUITES) if args.suite == "all" else [args.suite]
    print(f"host={HOST} port={args.port} n_predict={args.n}")
    for s in suites:
        r = None
        for _ in range(args.runs):
            r = run_suite(args.port, s, args.n)
        print(f"  {s:10s} tok/pass={r['tok_pass']:.3f}  accept={r['accept']:.3f}  tok/s={r['tok_s']:.2f}  "
              f"(gen={r['gen']} passes={r['passes']})")

if __name__ == "__main__":
    main()

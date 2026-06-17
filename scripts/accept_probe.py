import json,urllib.request
URL="http://127.0.0.1:8137/completion"
prompts={
 "prose":"Write a paragraph about the history of the Roman Empire.",
 "code":"Write a Python function that computes the nth Fibonacci number iteratively.",
 "factual":"List the planets of the solar system in order from the sun and one fact about each.",
 "reasoning":"If a train travels 60 km in 45 minutes, what is its speed in km/h? Show your steps.",
 "repetitive":"Count from 1 to 30, one number per line.",
}
tot_d=tot_a=0
for name,p in prompts.items():
    body=json.dumps({"prompt":p,"n_predict":160,"temperature":0,"cache_prompt":False}).encode()
    r=json.load(urllib.request.urlopen(urllib.request.Request(URL,body,{"Content-Type":"application/json"})))
    t=r["timings"]; d=t.get("draft_n",0); a=t.get("draft_n_accepted",0)
    tot_d+=d; tot_a+=a
    acc=100*a/d if d else 0
    print(f"  {name:11s} pred={t['predicted_n']:3d}  draft={d:4d} acc={a:3d} ({acc:4.1f}%)  {t['predicted_per_second']:.1f} t/s")
print(f"  {'OVERALL':11s} acceptance = {100*tot_a/tot_d:.1f}%  (draft={tot_d}, accepted={tot_a})")
print(f"  tokens per target pass ≈ {1 + tot_a/(tot_d/3):.2f}  (n_max=3)")

import json,urllib.request,sys,os
PORT=sys.argv[1]
HOST=os.environ.get("HOST","127.0.0.1")  # set HOST=<pi-ip> to measure a remote Pi server
PROMPTS={
 "prose":"Write a paragraph about the history of the Roman Empire.",
 "code":"Write a Python function that computes the nth Fibonacci number iteratively.",
 "factual":"List the planets of the solar system in order from the sun and one fact about each.",
 "reasoning":"If a train travels 60 km in 45 minutes, what is its speed in km/h? Show your steps.",
}
tot_gen=tot_acc=0
for name,p in PROMPTS.items():
    body=json.dumps({"prompt":p,"n_predict":160,"temperature":0,"cache_prompt":False}).encode()
    r=json.load(urllib.request.urlopen(urllib.request.Request(f"http://{HOST}:{PORT}/completion",body,{"Content-Type":"application/json"}),timeout=120))
    t=r["timings"]; gen=t["predicted_n"]; acc=t.get("draft_n_accepted",0)
    passes=gen-acc
    tpp=gen/passes if passes else float('inf')
    tot_gen+=gen; tot_acc+=acc
    print(f"  {name:10s} gen={gen:3d} acc={acc:3d} passes={passes:3d}  tok/pass={tpp:.2f}")
passes=tot_gen-tot_acc
print(f"  OVERALL gen={tot_gen} acc={tot_acc} passes={passes}  tok/pass={tot_gen/passes:.3f}  (Pi throughput multiplier)")

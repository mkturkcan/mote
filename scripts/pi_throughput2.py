import json,urllib.request,sys
PORT=sys.argv[1]; DRAFT_MB=10.7; TGT_MB=float(sys.argv[2]) if len(sys.argv)>2 else 970.0
P={"prose":"Write a paragraph about the history of the Roman Empire.",
   "code":"Write a Python function that computes the nth Fibonacci number iteratively.",
   "factual":"List the planets of the solar system in order from the sun and one fact about each.",
   "reasoning":"If a train travels 60 km in 45 minutes, what is its speed in km/h? Show your steps."}
tg=ta=td=0
for name,p in P.items():
    body=json.dumps({"prompt":p,"n_predict":160,"temperature":0,"cache_prompt":False}).encode()
    r=json.load(urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{PORT}/completion",body,{"Content-Type":"application/json"}),timeout=120))
    t=r["timings"]; g=t["predicted_n"]; a=t.get("draft_n_accepted",0); d=t.get("draft_n",0)
    tg+=g; ta+=a; td+=d
passes=tg-ta
mb=td*DRAFT_MB + passes*TGT_MB         # total bytes read on the Pi
mbtok=mb/tg
print(f"  gen={tg} acc={ta} drafted={td} passes={passes}  MB/token={mbtok:.0f}  (lower=faster; rel to 406 baseline: {406/mbtok:.3f}x)")

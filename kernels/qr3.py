import numpy as np
IN,MID=1536,12288
GATE=np.fromfile("data/gate.f32",dtype=np.float32,count=MID*IN).reshape(MID,IN)
X=np.fromfile("/tmp/x20.f32",dtype=np.float32).reshape(-1,IN); Xte=X[int(len(X)*0.85):]
def rand_orth(n,seed=1):
    rng=np.random.default_rng(seed); A=rng.standard_normal((n,n)); Q,Rr=np.linalg.qr(A); return (Q*np.sign(np.diag(Rr))).astype(np.float32)
def oe(W,Wq,Xt): yr=Xt.dot(W.T); return 100*np.linalg.norm(Xt.dot(Wq.T)-yr)/np.linalg.norm(yr)
def vq(W,bits,d,iters=15,seed=0):
    rng=np.random.default_rng(seed); rows,cols=W.shape
    V=W.reshape(-1,d).astype(np.float32); K=2**(bits*d)
    S=V[rng.choice(len(V),min(len(V),40000),replace=False)]
    C=S[rng.choice(len(S),K,replace=False)].copy()
    for _ in range(iters):
        a=np.argmin(((S[:,None]-C[None])**2).sum(2),1)
        for k in range(K):
            m=a==k
            if m.any(): C[k]=S[m].mean(0)
    # assign all in chunks (memory)
    out=np.empty(len(V),dtype=np.int32)
    for i in range(0,len(V),200000):
        out[i:i+200000]=np.argmin(((V[i:i+200000,None]-C[None])**2).sum(2),1)
    return C[out].reshape(rows,cols).astype(np.float32), K, d
R=rand_orth(IN); WR=GATE.dot(R); XR=Xte.dot(R)
print("GATE per-tensor output error (ref: Q4_0 ~5.6%, uniform-2bit ~58.7%)",flush=True)
for bits,d in [(2,2),(2,4),(2,8),(3,4),(4,8)]:
    Wq,K,dd=vq(WR,bits,d)
    print(f"  Had+VQ {bits}-bit d={d} ({K} codes): {oe(WR,Wq,XR):.2f}%",flush=True)

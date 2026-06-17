# END-TO-END FFN error: quantize gate/up/down, run full FFN(x)=DOWN@(gelu(GATE@x)*(UP@x)), vs fp32.
import numpy as np
IN,MID=1536,12288
g=np.fromfile("data/gate.f32",dtype=np.float32,count=MID*IN).reshape(MID,IN)
u=np.fromfile("data/up.f32",dtype=np.float32,count=MID*IN).reshape(MID,IN)
d=np.fromfile("data/down.f32",dtype=np.float32,count=1536*MID).reshape(1536,MID)
X=np.fromfile("/tmp/x20.f32",dtype=np.float32).reshape(-1,IN); Xte=X[int(len(X)*0.85):]
def gelu(x): return 0.5*x*(1+np.tanh(0.7978845608*(x+0.044715*x**3)))
def ffn(Wg,Wu,Wd,x): return (gelu(x.dot(Wg.T))*x.dot(Wu.T)).dot(Wd.T)
ref=ffn(g,u,d,Xte)
def uquant(W,bits,group=32):
    Q=W.astype(np.float32).copy(); qm=2**(bits-1)-1
    rows,cols=W.shape
    for c0 in range(0,cols,group):
        b=Q[:,c0:c0+group]; s=np.maximum(np.abs(b).max(1,keepdims=True),1e-8)/qm
        Q[:,c0:c0+group]=np.round(b/s).clip(-qm-1,qm)*s
    return Q
def vq(W,bits,dd,iters=12,seed=0):
    rng=np.random.default_rng(seed); rows,cols=W.shape
    V=W.reshape(-1,dd).astype(np.float32); K=2**(bits*dd)
    S=V[rng.choice(len(V),min(len(V),max(40000,3*K)),replace=False)]
    C=S[rng.choice(len(S),K,replace=False)].copy()
    for _ in range(iters):
        a=np.argmin(((S[:,None]-C[None])**2).sum(2),1)
        for k in range(K):
            m=a==k
            if m.any(): C[k]=S[m].mean(0)
    out=np.empty(len(V),np.int32)
    for i in range(0,len(V),200000): out[i:i+200000]=np.argmin(((V[i:i+200000,None]-C[None])**2).sum(2),1)
    return C[out].reshape(rows,cols).astype(np.float32)
def err(Wg,Wu,Wd): return 100*np.linalg.norm(ffn(Wg,Wu,Wd,Xte)-ref)/np.linalg.norm(ref)
print("END-TO-END FFN output error (vs fp32):",flush=True)
print(f"  uniform 4-bit g32 (Q4-like): {err(uquant(g,4),uquant(u,4),uquant(d,4)):.2f}%",flush=True)
print(f"  uniform 3-bit g32          : {err(uquant(g,3),uquant(u,3),uquant(d,3)):.2f}%",flush=True)
print(f"  VQ 2-bit d=4 (256c)        : {err(vq(g,2,4),vq(u,2,4),vq(d,2,4)):.2f}%",flush=True)
print(f"  VQ 3-bit d=4 (4096c)       : {err(vq(g,3,4),vq(u,3,4),vq(d,3,4)):.2f}%",flush=True)

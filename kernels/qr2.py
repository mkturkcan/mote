# Focused: can a proper CODEBOOK make 2-bit usable? (QuIP#/AQLM core). GATE, Hadamard space.
import numpy as np
IN,MID=1536,12288
GATE=np.fromfile("data/gate.f32",dtype=np.float32,count=MID*IN).reshape(MID,IN)
X=np.fromfile("/tmp/x20.f32",dtype=np.float32).reshape(-1,IN)
Ntr=int(len(X)*0.85); Xtr,Xte=X[:Ntr],X[Ntr:]
def rand_orth(n,seed=1):
    rng=np.random.default_rng(seed); A=rng.standard_normal((n,n)); Q,R=np.linalg.qr(A); return (Q*np.sign(np.diag(R))).astype(np.float32)
def out_err(W,Wq,Xt): yr=Xt.dot(W.T); return np.linalg.norm(Xt.dot(Wq.T)-yr)/np.linalg.norm(yr)

def lloyd_max(w, bits, group, iters=15):
    """optimal scalar (1D k-means) per group."""
    K=2**bits; q=np.empty_like(w); n=len(w)
    for g0 in range(0,n,group):
        blk=w[g0:g0+group]
        c=np.quantile(blk, np.linspace(0.02,0.98,K))  # init centroids
        for _ in range(iters):
            d=np.abs(blk[:,None]-c[None,:]); a=d.argmin(1)
            for k in range(K):
                m=a==k
                if m.any(): c[k]=blk[m].mean()
        d=np.abs(blk[:,None]-c[None,:]); q[g0:g0+group]=c[d.argmin(1)]
    return q
def scalar_apply(W,bits,group,fn):
    Q=np.empty_like(W)
    for r in range(W.shape[0]): Q[r]=fn(W[r],bits,group)
    return Q

def vq(W, bits, d, kmeans_iters=20, seed=0):
    """vector codebook: reshape rows into d-dim vecs, k-means with 2^(bits*d) codes (shared across matrix)."""
    rng=np.random.default_rng(seed)
    rows,cols=W.shape; assert cols%d==0
    V=W.reshape(rows, cols//d, d).reshape(-1,d)   # all d-dim subvectors
    K=2**(bits*d)
    # k-means (sample for speed)
    idx=rng.choice(len(V), min(len(V),20000), replace=False)
    C=V[rng.choice(len(V),K,replace=False)].copy()
    S=V[idx]
    for _ in range(kmeans_iters):
        a=np.argmin(((S[:,None,:]-C[None,:,:])**2).sum(2),axis=1)
        for k in range(K):
            m=a==k
            if m.any(): C[k]=S[m].mean(0)
    # assign all
    a=np.argmin(((V[:,None,:]-C[None,:,:])**2).sum(2),axis=1)
    Q=C[a].reshape(rows,cols//d,d).reshape(rows,cols)
    return Q.astype(np.float32)

R=rand_orth(IN); WR=GATE.dot(R); XteR=Xte.dot(R)
print("reference: Q4_0 ~5.6% ; uniform-2bit Had+GPTQ was 58.7%")
print(f"  Had + Lloyd-Max 2-bit g128: {100*out_err(WR, scalar_apply(WR,2,128,lloyd_max), XteR):.2f}%")
print(f"  Had + Lloyd-Max 2-bit g32 : {100*out_err(WR, scalar_apply(WR,2,32,lloyd_max), XteR):.2f}%")
print(f"  Had + VQ d=4 (2-bit,256c) : {100*out_err(WR, vq(WR,2,4), XteR):.2f}%")
print(f"  Had + VQ d=8 (2-bit,65536c): {100*out_err(WR, vq(WR,2,8), XteR):.2f}%")
print(f"  Had + Lloyd-Max 3-bit g32 : {100*out_err(WR, scalar_apply(WR,3,32,lloyd_max), XteR):.2f}%")

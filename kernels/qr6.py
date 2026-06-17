# Full stack: block-Hadamard incoherence + VQ codebook, end-to-end FFN. Does 2-bit reach the Q3 bar (24%)?
import numpy as np
IN,MID=1536,12288
g=np.fromfile("data/gate.f32",dtype=np.float32,count=MID*IN).reshape(MID,IN)
u=np.fromfile("data/up.f32",dtype=np.float32,count=MID*IN).reshape(MID,IN)
d=np.fromfile("data/down.f32",dtype=np.float32,count=1536*MID).reshape(1536,MID)
X=np.fromfile("/tmp/x20.f32",dtype=np.float32).reshape(-1,IN); Xte=X[int(len(X)*0.85):]
def gelu(x): return 0.5*x*(1+np.tanh(0.7978845608*(x+0.044715*x**3)))
ref=(gelu(Xte.dot(g.T))*Xte.dot(u.T)).dot(d.T)

def block_orth(n, blk=512, seed=0):
    rng=np.random.default_rng(seed); B=n//blk; mats=[]
    for b in range(B):
        A=rng.standard_normal((blk,blk)); Q,R=np.linalg.qr(A); mats.append((Q*np.sign(np.diag(R))).astype(np.float32))
    return mats, blk
def rot_cols(W, mats, blk):  # W[:, :] columns rotated block-wise: returns W @ R (R block-diagonal)
    out=np.empty_like(W)
    for b,Q in enumerate(mats): out[:, b*blk:(b+1)*blk]=W[:, b*blk:(b+1)*blk].dot(Q)
    return out
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

Rg=block_orth(IN,1536,1); Ru=block_orth(IN,1536,2); Rd=block_orth(MID,4096,3)  # full-orth gate/up, big-block down
def quant_all(bits,dd):
    gq=vq(rot_cols(g,*Rg),bits,dd); uq=vq(rot_cols(u,*Ru),bits,dd); dq=vq(rot_cols(d,*Rd),bits,dd)
    return gq,uq,dq
def ffn_q(gq,uq,dq):
    go=rot_cols(Xte,*Rg).dot(gq.T); uo=rot_cols(Xte,*Ru).dot(uq.T)
    geglu=gelu(go)*uo
    return rot_cols(geglu,*Rd).dot(dq.T)
def err(gq,uq,dq): return 100*np.linalg.norm(ffn_q(gq,uq,dq)-ref)/np.linalg.norm(ref)
print("bar: Q4=9.6%, Q3=24% (deployed). FULL STACK block-Hadamard + VQ:",flush=True)
for bits,dd in [(2,4),(3,4)]:
    gq,uq,dq=quant_all(bits,dd)
    print(f"  Had+VQ {bits}-bit d={dd}: {err(gq,uq,dq):.2f}%",flush=True)

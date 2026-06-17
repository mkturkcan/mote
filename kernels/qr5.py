# Per-matrix quant sensitivity + mixed-precision search. End-to-end FFN error vs fp32.
import numpy as np
IN,MID=1536,12288
g=np.fromfile("data/gate.f32",dtype=np.float32,count=MID*IN).reshape(MID,IN)
u=np.fromfile("data/up.f32",dtype=np.float32,count=MID*IN).reshape(MID,IN)
d=np.fromfile("data/down.f32",dtype=np.float32,count=1536*MID).reshape(1536,MID)
X=np.fromfile("/tmp/x20.f32",dtype=np.float32).reshape(-1,IN); Xte=X[int(len(X)*0.85):]
def gelu(x): return 0.5*x*(1+np.tanh(0.7978845608*(x+0.044715*x**3)))
def ffn(Wg,Wu,Wd): return (gelu(Xte.dot(Wg.T))*Xte.dot(Wu.T)).dot(Wd.T)
ref=ffn(g,u,d)
def uq(W,bits,group=32):
    if bits>=16: return W
    Q=W.astype(np.float32).copy(); qm=2**(bits-1)-1
    for c0 in range(0,W.shape[1],group):
        b=Q[:,c0:c0+group]; s=np.maximum(np.abs(b).max(1,keepdims=True),1e-8)/qm
        Q[:,c0:c0+group]=np.round(b/s).clip(-qm-1,qm)*s
    return Q
def e(bg,bu,bd): return 100*np.linalg.norm(ffn(uq(g,bg),uq(u,bu),uq(d,bd))-ref)/np.linalg.norm(ref)
print("PER-MATRIX 2-bit sensitivity (one at 2-bit, rest fp32):",flush=True)
print(f"  gate=2bit only : {e(2,16,16):.2f}%",flush=True)
print(f"  up=2bit only   : {e(16,2,16):.2f}%",flush=True)
print(f"  down=2bit only : {e(16,16,2):.2f}%",flush=True)
print("PER-MATRIX 3-bit sensitivity:",flush=True)
print(f"  gate=3bit only : {e(3,16,16):.2f}%",flush=True)
print(f"  up=3bit only   : {e(16,3,16):.2f}%",flush=True)
print(f"  down=3bit only : {e(16,16,3):.2f}%",flush=True)
print("MIXED configs (bg,bu,bd) -> avg FFN bits, end-to-end error:",flush=True)
for bg,bu,bd in [(4,4,4),(3,3,3),(4,4,2),(3,3,2),(4,4,3),(3,4,2),(4,3,2)]:
    avg=(bg+bu+bd)/3
    print(f"  gate={bg} up={bu} down={bd} (avg {avg:.2f}b): {e(bg,bu,bd):.2f}%",flush=True)

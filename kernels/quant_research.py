#!/usr/bin/env python3
"""Research-grade calibrated low-bit quantization for Gemma 4 E2B FFN.

Implements the core of QuIP#/GPTQ: incoherence preprocessing (random orthogonal rotation, the
deployable version is a fast Hadamard) + GPTQ error-compensated quantization using the REAL
activation Hessian. Measures the honest metric — relative OUTPUT error on held-out real
activations (y = W@x) vs fp32 — which is what actually determines KL. Compares to Q4 (~5.6%).

Goal: can calibrated 2-bit reach Q4 quality? If yes, justify the llama.cpp kernel integration.
"""
import numpy as np, sys

IN, MID = 1536, 12288
def load(path, n): return np.fromfile(path, dtype=np.float32, count=n).astype(np.float32)

# real weights (blk.20): gate/up = [MID, IN] (out,in); down = [1536, MID]
GATE = load("data/gate.f32", MID*IN).reshape(MID, IN)
UP   = load("data/up.f32",   MID*IN).reshape(MID, IN)
DOWN = load("data/down.f32", 1536*MID).reshape(1536, MID)
# real activations
X_norm = np.fromfile("/tmp/x20.f32", dtype=np.float32).reshape(-1, IN)      # gate/up input
X_geglu= np.fromfile("/tmp/geglu20.f32", dtype=np.float32).reshape(-1, MID) # down input
print(f"weights: gate{GATE.shape} down{DOWN.shape}  acts: norm{X_norm.shape} geglu{X_geglu.shape}")

def quant_group(w, bits, group):
    """symmetric per-group RTN of a 1D vector w."""
    n = w.shape[0]; q = np.empty_like(w); qmax = 2**(bits-1)-1
    for g0 in range(0, n, group):
        g1 = min(g0+group, n); blk = w[g0:g1]
        s = np.max(np.abs(blk))/qmax if np.max(np.abs(blk))>0 else 1.0
        q[g0:g1] = np.round(blk/s).clip(-qmax-1, qmax)*s
    return q

def rtn(W, bits, group):
    Q = np.empty_like(W)
    for r in range(W.shape[0]): Q[r] = quant_group(W[r], bits, group)
    return Q

def gptq(W, H, bits, group, percdamp=0.01):
    """GPTQ: error-compensated quant. W [out,in], H [in,in] input Hessian. Per-group symmetric."""
    W = W.copy().astype(np.float64); cols = W.shape[1]
    H = H.copy().astype(np.float64)
    d = np.diag(H).copy(); dead = d==0; H[dead,dead]=1.0; W[:,dead]=0
    damp = percdamp*np.mean(np.diag(H)); H[np.diag_indices(cols)] += damp
    # Hinv via cholesky of H, then inverse-cholesky trick (upper triangular)
    L = np.linalg.cholesky(H); Hinv = np.linalg.inv(L).T.dot(np.linalg.inv(L))  # = H^-1
    Hinv = np.linalg.cholesky(Hinv).T  # upper-tri Cholesky of H^-1 (GPTQ uses this)
    Q = np.zeros_like(W); qmax = 2**(bits-1)-1
    scales = {}
    for i in range(cols):
        if i % group == 0:  # compute per-group scale from current (updated) block
            g1 = min(i+group, cols)
            blk = W[:, i:g1]
            scales['s'] = np.maximum(np.max(np.abs(blk), axis=1, keepdims=True), 1e-8)/qmax
        s = scales['s']                       # [out,1]
        w = W[:, i:i+1]
        q = np.round(w/s).clip(-qmax-1, qmax)*s
        Q[:, i:i+1] = q
        dinv = Hinv[i, i]
        err = (w - q) / dinv
        if i+1 < cols:
            W[:, i+1:] -= err.dot(Hinv[i:i+1, i+1:])
    return Q.astype(np.float32)

def rand_orth(n, seed=0):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n, n)).astype(np.float64)
    Q, R = np.linalg.qr(A); Q *= np.sign(np.diag(R))   # uniform orthogonal
    return Q.astype(np.float32)

def out_err(W, Wq, Xtest):
    yr = Xtest.dot(W.T); yq = Xtest.dot(Wq.T)
    return np.linalg.norm(yq-yr)/np.linalg.norm(yr)

def evaluate(name, W, X, bits, group):
    Ntr = int(len(X)*0.85); Xtr, Xte = X[:Ntr], X[Ntr:]
    H = Xtr.T.astype(np.float64).dot(Xtr.astype(np.float64))/len(Xtr)
    # incoherence rotation R (orthogonal); y = W x = (W R)(R^T x)
    R = rand_orth(W.shape[1], seed=1)
    results = {}
    results['RTN']        = out_err(W, rtn(W, bits, group), Xte)
    results['GPTQ']       = out_err(W, gptq(W, H, bits, group), Xte)
    # incoherence: quantize W@R in rotated space; eval by rotating test acts
    WR = W.dot(R); XteR = Xte.dot(R)
    Hr = (Xtr.dot(R)).T.astype(np.float64).dot((Xtr.dot(R)).astype(np.float64))/len(Xtr)
    results['Had+RTN']    = out_err(WR, rtn(WR, bits, group), XteR)
    results['Had+GPTQ']   = out_err(WR, gptq(WR, Hr, bits, group), XteR)
    print(f"  {name} {bits}-bit g{group}:  " + "  ".join(f"{k}={100*v:.2f}%" for k,v in results.items()))
    return results

print("\n(reference: Q4_0 output error ~5.6%)")
for bits in (3, 2):
    evaluate("GATE", GATE, X_norm, bits, 128)

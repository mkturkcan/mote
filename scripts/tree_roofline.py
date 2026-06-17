#!/usr/bin/env python3
"""Roofline cost model: does tree-draft's +8.7% tokens/pass translate to Pi5 tok/s?

We cannot run on the Pi, so we predict via a per-verify-pass roofline. The deployed decode is
~99.8% weight-DRAM-bound: each verify pass streams the full target weights ONCE regardless of how
many draft rows ride along (up to the A76 compute:BW crossover at M~=11). Tree-draft trades a small
extra per-pass cost (the draft-side branch pass + a few more verify rows) for FEWER passes
(higher tokens/pass). On a bandwidth-bound machine the saved passes save full weight reads, so the
net can be positive even though it is NEGATIVE on a compute-rich/fast-memory dev box.

All inputs are stated; the OUTPUT is the predicted Pi tok/s ratio tree/linear. Measured tokens/pass
come from the 16-prompt battery (the hardware-independent gain). Run: python3 tree_roofline.py
"""

# ---- measured, hardware-independent (16-prompt battery, n_predict=128) ----
TPP_LINEAR = 2.691   # k=1 linear MTP tokens per verify pass
TPP_TREE   = 2.926   # k=2 tree-draft tokens per verify pass  (+8.7%)
PRIMARY_DRAFT_STEPS = 4.0          # avg primary MTP draft steps/pass (draft_n 2804 / 700 passes ~= 4.0)

# ---- Pi5 (Cortex-A76) roofline inputs ----
BW_GBs        = 15.0               # sustained LPDDR DRAM BW, GB/s (summary: 14-17)
W_TGT_MB      = 1383.0             # target weight bytes read per verify pass (Q4_K_M, summary's per-token figure)
W_MTP_STEP_MB = 10.7              # MTP draft head bytes per draft step (summary)
M_CROSSOVER   = 11                 # A76 compute:BW crossover (verify stays BW-bound while M < ~11)

def ms(mb):  # DRAM read time at sustained BW
    return mb / (BW_GBs * 1000.0) * 1000.0

T_tgt   = ms(W_TGT_MB)             # target verify read, dominates everything
T_step  = ms(W_MTP_STEP_MB)       # one MTP draft step

# avg verify M (rows) — both stay under the crossover on average, so verify time ~= T_tgt for both.
M_lin  = 1 + PRIMARY_DRAFT_STEPS                       # s + primary draft rows
def predict(branch_frac, branch_steps):
    # tree adds: a branch DRAFT pass (1 seed decode + branch_steps) on a fraction of passes,
    # plus branch_steps extra VERIFY rows (hidden while M stays < crossover).
    M_tree = M_lin + branch_frac * branch_steps
    T_branch_draft = branch_frac * (1 + branch_steps) * T_step
    # verify time: BW-bound (== T_tgt) while avg M < crossover; else linearly compute-bound past it.
    def T_verify(M):
        return T_tgt if M < M_CROSSOVER else T_tgt * (M / M_CROSSOVER)
    t_lin  = T_verify(M_lin)  + PRIMARY_DRAFT_STEPS * T_step
    t_tree = T_verify(M_tree) + PRIMARY_DRAFT_STEPS * T_step + T_branch_draft
    toks_lin  = TPP_LINEAR  / t_lin
    toks_tree = TPP_TREE    / t_tree
    return M_tree, t_lin, t_tree, (toks_tree / toks_lin - 1.0) * 100.0

print(f"Pi roofline: T_target_read={T_tgt:.1f} ms/pass, T_mtp_step={T_step:.2f} ms, M_lin={M_lin:.1f}")
print(f"{'branch_frac':>11} {'branch_steps':>12} {'avg_M_tree':>10} {'t_lin(ms)':>9} {'t_tree(ms)':>10} {'Pi tok/s gain':>13}")
for bf in (0.35, 0.50, 0.65):
    for bs in (3, 5, 8):
        M_tree, tl, tt, gain = predict(bf, bs)
        print(f"{bf:>11.2f} {bs:>12d} {M_tree:>10.1f} {tl:>9.1f} {tt:>10.1f} {gain:>+12.1f}%")

print("\nInterpretation:")
print(" - As long as avg verify M stays < the A76 crossover (~11), the target read dominates and")
print("   tree-draft's +8.7% tokens/pass survives minus a small branch-draft tax -> NET Pi WIN (~+5-8%).")
print(" - A shorter branch tail (cap branch_steps) lowers both avg M and the tax -> more robust win.")
print(" - Contrast dev box (fast DDR5 ~50-60 GB/s, T_target_read ~30-40ms): the same branch tax is a")
print("   much larger FRACTION of a cheap pass -> measured -23% there. The Pi inverts this economics.")

"""
demo_cmb_baseline.py — worked example of cmb_baseline.py in the theta*-native basis.

Target basis (Fisher stiff directions):  thetastar, ombh2, omch2, ns, logAse2tau
(tau fixed at FID_TAU for T-only; it splits into logA + tau when polarization enters).
"""
import numpy as np, time
import cmb_baseline as cb

rng = np.random.default_rng(0)
LMIN, LMAX = 2, 767
FID = cb.FID
NAMES = ['thetastar', 'ombh2', 'omch2', 'ns', 'logAse2tau']
STEPS = {n: cb._DEFAULT_STEP[n] for n in NAMES}

# 1. Fisher bound in the theta*-native basis --------------------------
names, F = cb.fisher(FID, STEPS, LMIN, LMAX, lensed=True, pol=False)
con = cb.constraints(names, F)
print("Fisher 1-sigma (theta*-basis, T-only, CV-limited, lmax=767):")
for n in names: print(f"   {n:11s} {con['sigma'][n]:.4g}")
print(f"   cond(F) = {con['cond']:.2e}   (physical-H0 basis was ~8e7)")
# show the residual correlations you were told to check, not assume away
print("   correlations to watch:")
i, j = names.index('thetastar'), names.index('omch2')
print(f"     thetastar-omch2   = {con['corr'][i, j]:+.2f}")
ia = names.index('logAse2tau')
print(f"     logAse2tau-omch2  = {con['corr'][ia, j]:+.2f}")

# 2. Discrimination baseline (LCDM vs oscillatory) --------------------
ells, C_lcdm = cb.cov_of(FID, LMIN, LMAX, pol=False)
ells, C_osc  = cb.cov_of(FID, LMIN, LMAX, pol=False, feature=(0.03, 30.0, 0.0))
sep = np.sqrt(cb.kl_separability(C_osc, C_lcdm, ells))
auc_opt = cb.optimal_auc(C_lcdm, C_osc, ells, nsim=250, seed=1)
print(f"\nLCDM vs oscillatory: {sep:.1f} sigma-equiv, optimal AUC = {auc_opt:.4f}")

# 3. Validate latent estimates against the bound ----------------------
sig = con['sigma_vec']
Ptrue = np.array([[FID[n] for n in NAMES]] * 400, float) \
        + rng.standard_normal((400, 5)) * sig * 3.0
net = lambda s: Ptrue + rng.standard_normal((400, 5)) * s * sig
print("\nRecovery (mean z_std: ~1 good | >1 leaves info | <1 leakage):")
for lb, s in [("bound-reaching x1.0", 1.0), ("under-trained x1.8", 1.8), ("leaky x0.5", 0.5)]:
    rep = cb.recovery_report(Ptrue, net(s), con['sigma'], NAMES)
    print(f"   {lb:22s} z_std = {np.mean([rep[n]['z_std'] for n in NAMES]):.2f}")

# 4. theta*-first + DE-before-solve check (Set B readiness) -----------
th = FID['thetastar']
_, Cl = cb.cov_of(FID, LMIN, LMAX, pol=False)                       # LCDM
_, Cw = cb.cov_of({**FID, 'w0': -0.9, 'wa': -0.3}, LMIN, LMAX, pol=False)  # w0wa @ same theta*
d = np.max(np.abs(Cw[:, 0, 0] - Cl[:, 0, 0]) / Cl[:, 0, 0])
print(f"\nSet-B readiness: w0wa vs LCDM at fixed theta* -> max frac TT diff = {d:.1%}")
print("   (small & late-time only, as expected once theta* is held fixed)")

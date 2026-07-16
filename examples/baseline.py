"""
baseline.py — Analytic two-point baseline for the CMB deep-learning programme.

Provides the ground-truth objects the map/Cl networks are measured against:
  * signal (+noise) spectra from CAMB, LCDM / w0waCDM / oscillatory-feature
  * Fisher matrix over T+E (+optional B), with beam, noise, f_sky
  * parameter constraints: 1-sigma, correlation, stiff/soft eigenstructure
  * exact full-sky Gaussian (Wishart) log-likelihood
  * KL separability between two models (analytic discrimination distance)
  * Monte-Carlo likelihood-ratio ROC / AUC (optimal Neyman-Pearson classifier)

Role: CEILING in the Gaussian regime (network validated by REACHING it);
      FLOOR once lensing enters (network validated by EXCEEDING it, the
      excess being non-Gaussian information).

Dependencies: numpy, scipy, camb.
"""
import numpy as np
import camb
from scipy.stats import wishart, rankdata

# ----------------------------------------------------------------------
# 1. Spectra
# ----------------------------------------------------------------------
FID = dict(ombh2=0.02237, omch2=0.1200, H0=67.36, tau=0.0544,
           logA=3.044, ns=0.9649)          # logA = ln(1e10 As)

def _oscillatory_PK(k, As, ns, A_lin, freq, phase, k0=0.05):
    tilt = As * (k / k0) ** (ns - 1.0)
    return tilt * (1.0 + A_lin * np.sin(freq * np.log(k / k0) + phase))

def get_spectra(params, lmax=767, lensed=True, w0=None, wa=None,
                feature=None):
    """Raw C_ell (muK^2) dict: ell, TT, EE, TE, BB. feature=(A_lin,freq,phase)."""
    As = 1e-10 * np.exp(params['logA'])
    kw = dict(H0=params['H0'], ombh2=params['ombh2'], omch2=params['omch2'],
              mnu=0.06, omk=0.0, tau=params['tau'], ns=params['ns'],
              lmax=lmax + 50)
    if w0 is not None:
        kw.update(dark_energy_model='ppf', w=w0, wa=wa)
    if feature is None:
        kw['As'] = As
    pars = camb.set_params(**kw)
    if feature is not None:
        A_lin, freq, phase = feature
        pars.set_initial_power_function(
            _oscillatory_PK, args=(As, params['ns'], A_lin, freq, phase),
            effective_ns_for_nonlinear=params['ns'])
    res = camb.get_results(pars)
    key = 'lensed_scalar' if lensed else 'unlensed_scalar'
    d = res.get_cmb_power_spectra(pars, CMB_unit='muK', raw_cl=True)[key]
    ell = np.arange(d.shape[0])
    return dict(ell=ell, TT=d[:, 0], EE=d[:, 1], TE=d[:, 3], BB=d[:, 2])

# ----------------------------------------------------------------------
# 2. Noise + covariance matrix per ell
# ----------------------------------------------------------------------
def noise_cl(lmax, dT_uK_arcmin=None, dP_uK_arcmin=None, fwhm_arcmin=0.0):
    """White-noise N_ell (muK^2), deconvolved by a Gaussian beam. None => 0."""
    ell = np.arange(lmax + 1)
    sig = np.deg2rad(fwhm_arcmin / 60.0) / np.sqrt(8 * np.log(2))
    binv = np.exp(ell * (ell + 1) * sig ** 2)          # inverse beam^2
    NT = np.zeros(lmax + 1); NP = np.zeros(lmax + 1)
    if dT_uK_arcmin:
        NT = (np.deg2rad(dT_uK_arcmin / 60.0) ** 2) * binv
    if dP_uK_arcmin:
        NP = (np.deg2rad(dP_uK_arcmin / 60.0) ** 2) * binv
    return NT, NP

def cov_TE(params, lmin=2, lmax=767, lensed=True, w0=None, wa=None,
           feature=None, noise=None):
    """Return (ells, C) with C shape (nl,2,2): [[TT,TE],[TE,EE]] signal(+noise)."""
    s = get_spectra(params, lmax, lensed, w0, wa, feature)
    sl = slice(lmin, lmax + 1); ells = s['ell'][sl]
    TT, EE, TE = s['TT'][sl].copy(), s['EE'][sl].copy(), s['TE'][sl].copy()
    if noise is not None:
        NT, NP = noise; TT += NT[sl]; EE += NP[sl]
    C = np.zeros((len(ells), 2, 2))
    C[:, 0, 0] = TT; C[:, 1, 1] = EE; C[:, 0, 1] = C[:, 1, 0] = TE
    return ells, C

# ----------------------------------------------------------------------
# 3. Fisher matrix
# ----------------------------------------------------------------------
def fisher(param_steps, fid=FID, fsky=1.0, lmin=2, lmax=767, lensed=True,
           w0wa=False, noise=None):
    """Fisher over params in param_steps (dict name->finite-diff step).
       w0wa=True adds w0,wa (fiducial -1,0) if present in param_steps."""
    base_w0, base_wa = (-1.0, 0.0) if w0wa else (None, None)
    ells, C = cov_TE(fid, lmin, lmax, lensed, base_w0, base_wa, noise=noise)
    Cinv = np.linalg.inv(C)
    keys = list(param_steps); dC = {}
    for k, step in param_steps.items():
        pp, pm = dict(fid), dict(fid); w0p = w0m = base_w0; wap = wam = base_wa
        if k == 'w0':  w0p, w0m = base_w0 + step, base_w0 - step
        elif k == 'wa': wap, wam = base_wa + step, base_wa - step
        else: pp[k] += step; pm[k] -= step
        _, Cp = cov_TE(pp, lmin, lmax, lensed, w0p, wap, noise=noise)
        _, Cm = cov_TE(pm, lmin, lmax, lensed, w0m, wam, noise=noise)
        dC[k] = (Cp - Cm) / (2 * step)
    n = len(keys); F = np.zeros((n, n)); pref = fsky * (2 * ells + 1) / 2.0
    M = {k: np.einsum('lij,ljk->lik', Cinv, dC[k]) for k in keys}
    for a in range(n):
        for b in range(a, n):
            F[a, b] = F[b, a] = np.sum(
                pref * np.einsum('lij,lji->l', M[keys[a]], M[keys[b]]))
    return keys, F

def constraints(keys, F):
    Finv = np.linalg.inv(F)
    sig = np.sqrt(np.diag(Finv))
    corr = Finv / np.outer(sig, sig)
    Dn = np.diag(1 / np.sqrt(np.diag(F)))
    evals, evecs = np.linalg.eigh(Dn @ F @ Dn)         # normalised
    return dict(sigma=dict(zip(keys, sig)), corr=corr,
                cond=np.linalg.cond(F), norm_eigs=evals, evecs=evecs)

# ----------------------------------------------------------------------
# 4. Exact Gaussian (Wishart) likelihood + KL separability
# ----------------------------------------------------------------------
def m2lnL(Chat, Cmodel, ells, fsky=1.0):
    """-2 ln L (up to const) for observed spectrum matrices Chat given model."""
    Ci = np.linalg.inv(Cmodel); p = Cmodel.shape[-1]
    tr = np.einsum('lij,lji->l', Ci, Chat)
    sign, logdetM = np.linalg.slogdet(Cmodel)
    sign2, logdetH = np.linalg.slogdet(Chat)
    return np.sum(fsky * (2 * ells + 1) * (tr + logdetM - logdetH - p))

def kl_separability(C1, C2, ells, fsky=1.0):
    """2*D_KL(1||2) summed over ell: analytic model-separation distance."""
    Ci2 = np.linalg.inv(C2); p = C1.shape[-1]
    A = np.einsum('lij,ljk->lik', Ci2, C1)
    tr = np.einsum('lii->l', A)
    sign, logdet = np.linalg.slogdet(A)
    return np.sum(fsky * (2 * ells + 1) * (tr - logdet - p))

# ----------------------------------------------------------------------
# 5. Monte-Carlo likelihood-ratio ROC / AUC (optimal classifier)
# ----------------------------------------------------------------------
def _draw_spectra(C, ells, rng):
    """Sample full-sky estimated spectrum matrices Chat ~ Wishart per ell."""
    nu = (2 * ells + 1).astype(int)
    out = np.empty_like(C)
    for i in range(len(ells)):
        out[i] = wishart.rvs(df=nu[i], scale=C[i] / nu[i], random_state=rng)
    return out

def llr_roc(C_A, C_B, ells, nsim=400, fsky=1.0, seed=0):
    """Optimal LLR classifier ROC. Returns dict(auc, llr_A, llr_B)."""
    rng = np.random.default_rng(seed)
    def llr_batch(Ctrue):
        vals = np.empty(nsim)
        for s in range(nsim):
            Chat = _draw_spectra(Ctrue, ells, rng)
            vals[s] = -0.5 * (m2lnL(Chat, C_A, ells, fsky)
                              - m2lnL(Chat, C_B, ells, fsky))
        return vals
    lA, lB = llr_batch(C_A), llr_batch(C_B)          # LLR = lnL_A - lnL_B
    # AUC = P(LLR|A > LLR|B); average ranks handle ties (identical models -> 0.5)
    ranks = rankdata(np.concatenate([lA, lB]))       # 1-based, tie-averaged
    rA = ranks[:nsim].sum()
    auc = (rA - nsim * (nsim + 1) / 2) / (nsim * nsim)
    return dict(auc=auc, llr_A=lA, llr_B=lB)

# ----------------------------------------------------------------------
# Demo
# ----------------------------------------------------------------------
if __name__ == '__main__':
    steps = dict(ombh2=1e-4, omch2=1e-3, H0=0.5, tau=1e-2, logA=2e-2, ns=5e-3)
    print("LCDM Fisher (CV-limited T+E, lmax=767):")
    k, F = fisher(steps); c = constraints(k, F)
    for kk in k: print(f"  {kk:6s} sigma={c['sigma'][kk]:.4g}")
    print(f"  cond={c['cond']:.2e}")

    print("\nw0waCDM Fisher (adds w0,wa):")
    steps2 = dict(steps); steps2['w0'] = 2e-2; steps2['wa'] = 5e-2
    k2, F2 = fisher(steps2, w0wa=True); c2 = constraints(k2, F2)
    for kk in ('H0', 'w0', 'wa'): print(f"  {kk:6s} sigma={c2['sigma'][kk]:.4g}")
    print(f"  cond={c2['cond']:.2e}")

    print("\nDiscrimination: LCDM vs oscillatory feature (A_lin=0.03,freq=30):")
    ells, C_l = cov_TE(FID)
    ells, C_f = cov_TE(FID, feature=(0.03, 30.0, 0.0))
    print(f"  sqrt(2 D_KL) = {np.sqrt(kl_separability(C_f, C_l, ells)):.1f} sigma-equiv")
    roc = llr_roc(C_l, C_f, ells, nsim=300, seed=1)
    print(f"  optimal-LLR AUC = {roc['auc']:.4f}")

    print("\nNull control: LCDM vs identical LCDM:")
    roc0 = llr_roc(C_l, C_l.copy(), ells, nsim=300, seed=2)
    print(f"  AUC = {roc0['auc']:.4f}  (should be ~0.5)")
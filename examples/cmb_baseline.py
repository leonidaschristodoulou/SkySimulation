"""
cmb_baseline.py  —  Analytic two-point baseline for the CMB deep-learning programme.

Assumes you have: (a) the model spectra you built for the sims, (b) their input
parameters, (c) your latent-space parameter estimates, and (d) CAMB.

CAMB is used where parameters must be varied continuously (Fisher derivatives,
per-sim likelihood/MCMC).  Your stored spectra are used directly wherever a fixed
model spectrum is enough (discrimination distance, optimal ROC, observed data).

Contents
  spectra          get_spectra                CAMB TT/EE/TE/BB for a param dict
  covariance       cov_from_arrays            build C from YOUR stored spectra
                   cov_of                     build C from a param dict (CAMB)
                   noise_cl                   white-noise + beam N_ell
  recovery bound   fisher, constraints        Fisher matrix, sigma, eigenstructure
  likelihood       m2lnL                      exact full-sky Gaussian (Wishart)
  discrimination   kl_separability            analytic model-separation distance
                   optimal_auc                optimal (Neyman-Pearson) AUC
                   auc_from_scores            YOUR network's AUC, same statistic
  validation       recovery_report            latent estimates vs Fisher bound
                   coverage                   calibration of network error bars
                   posterior_mcmc             exact-likelihood posterior for one sim

Conventions: spectra are RAW C_ell in muK^2 (NOT D_ell = l(l+1)C_l/2pi).
Parameter dicts use: ombh2, omch2, H0, tau, logA (= ln 1e10 As), ns, and
optionally w0, wa.  Edit get_spectra if your parametrisation differs
(e.g. cosmomc_theta instead of H0, or As instead of logA).

Dependencies: numpy, scipy, camb.  emcee optional (posterior_mcmc falls back to
a Fisher-Laplace Gaussian if emcee is absent).
"""
import numpy as np
import camb
from scipy.stats import wishart, rankdata

# Fiducial in the theta*-native basis (the Fisher stiff directions).
#   thetastar   : exact acoustic scale (CAMB solves H0)
#   logAse2tau  : ln(1e10 As) - 2 tau  = the T-only amplitude combination
# tau is fixed at FID_TAU in T-only mode; it becomes a free target (with logA)
# once E-mode polarisation is added -- see get_spectra / _resolve_amp_tau.
FID_TAU = 0.0544
FID = dict(thetastar=0.0104109, ombh2=0.02237, omch2=0.1200, ns=0.9649,
           logAse2tau=3.044 - 2 * FID_TAU)          # = logA - 2*tau

# Finite-difference steps by parameter name (thetastar needs a small one).
_DEFAULT_STEP = dict(thetastar=1e-6, cosmomc_theta=1e-6, H0=0.5,
                     ombh2=1e-4, omch2=1e-3, ns=5e-3,
                     logA=2e-2, logAse2tau=2e-2, As=4e-11, tau=5e-3,
                     w0=2e-2, wa=5e-2)

# ======================================================================
# 1.  Spectra from CAMB (only needed to vary parameters continuously)
# ======================================================================
def _oscillatory_PK(k, As, ns, A_lin, freq, phase, k0=0.05):
    return As * (k / k0) ** (ns - 1.0) * (1.0 + A_lin * np.sin(freq * np.log(k / k0) + phase))

def _resolve_amp_tau(p):
    """Return (As, tau) from whichever amplitude parametrisation is present.
       T-only  : 'logAse2tau' (or 'Ase2tau') + tau fixed at FID_TAU.
       T+E     : 'logA' (or 'As') together with a free 'tau'."""
    tau = p.get('tau', FID_TAU)
    if 'As' in p:
        As = p['As']
    elif 'logA' in p:
        As = 1e-10 * np.exp(p['logA'])
    elif 'logAse2tau' in p:                       # ln(1e10 As) - 2 tau
        As = 1e-10 * np.exp(p['logAse2tau'] + 2 * tau)
    elif 'Ase2tau' in p:
        As = p['Ase2tau'] * np.exp(2 * tau)
    else:
        raise KeyError("amplitude: need one of As, logA, logAse2tau, Ase2tau")
    return As, tau

def get_spectra(params, lmax=767, lensed=True, feature=None):
    """Raw C_ell (muK^2) dict ell,TT,EE,TE,BB.

    Geometry  : pass 'thetastar' (exact acoustic scale ~0.0104; CAMB solves H0)
                or 'cosmomc_theta', or 'H0' directly.
    Amplitude : 'logAse2tau' = ln(1e10 As) - 2 tau  (T-only; tau fixed at FID_TAU),
                or 'logA'/'As' together with a free 'tau' (T+E stage).
    Dark energy: include 'w0','wa' for w0waCDM. They are set in the SAME
                set_params call as thetastar, so H0 is solved against the correct
                expansion history (essential for Set B; harmless otherwise).
    feature=(A_lin,freq,phase) applies an oscillatory primordial P(k)."""
    p = dict(params)
    w0 = p.pop('w0', None); wa = p.pop('wa', None)
    As, tau = _resolve_amp_tau(p)
    kw = dict(ombh2=p['ombh2'], omch2=p['omch2'], mnu=0.06, omk=0.0,
              tau=tau, ns=p['ns'], lmax=lmax + 50)
    if 'thetastar' in p:                          # CAMB solves H0 from theta*
        kw['thetastar'] = p['thetastar']
    elif 'cosmomc_theta' in p:
        kw['cosmomc_theta'] = p['cosmomc_theta']
    else:
        kw['H0'] = p['H0']
    if w0 is not None:                            # DE set together with theta*
        kw.update(dark_energy_model='ppf', w=w0, wa=(0.0 if wa is None else wa))
    if feature is None:
        kw['As'] = As
    pars = camb.set_params(**kw)
    if feature is not None:
        pars.set_initial_power_function(_oscillatory_PK,
            args=(As, p['ns'], *feature), effective_ns_for_nonlinear=p['ns'])
    d = camb.get_results(pars).get_cmb_power_spectra(
        pars, CMB_unit='muK', raw_cl=True)['lensed_scalar' if lensed else 'unlensed_scalar']
    return dict(ell=np.arange(d.shape[0]), TT=d[:, 0], EE=d[:, 1], TE=d[:, 3], BB=d[:, 2])

# ======================================================================
# 2.  Covariance assembly
# ======================================================================
def noise_cl(lmax, dT_uK_arcmin=None, dP_uK_arcmin=None, fwhm_arcmin=0.0):
    """White-noise N_ell (muK^2), beam-deconvolved. Returns (NT, NP)."""
    ell = np.arange(lmax + 1)
    sig = np.deg2rad(fwhm_arcmin / 60.0) / np.sqrt(8 * np.log(2))
    binv = np.exp(ell * (ell + 1) * sig ** 2)
    NT = (np.deg2rad((dT_uK_arcmin or 0) / 60.0) ** 2) * binv
    NP = (np.deg2rad((dP_uK_arcmin or 0) / 60.0) ** 2) * binv
    return NT, NP

def cov_from_arrays(ells, TT, EE=None, TE=None, noise=None):
    """Per-ell covariance C (L,p,p) from YOUR stored spectra (raw C_ell, muK^2).
       Temperature-only -> (L,1,1); pass EE and TE for the T+E 2x2 block.
       noise=(NT,NP) arrays aligned with `ells`, or None."""
    ells = np.asarray(ells); TT = np.asarray(TT, float).copy()
    if noise is not None:
        TT = TT + noise[0]
    if EE is None:
        return ells, TT[:, None, None]
    EE = np.asarray(EE, float).copy()
    TE = np.zeros_like(TT) if TE is None else np.asarray(TE, float)
    if noise is not None:
        EE = EE + noise[1]
    C = np.zeros((len(ells), 2, 2))
    C[:, 0, 0] = TT; C[:, 1, 1] = EE; C[:, 0, 1] = C[:, 1, 0] = TE
    return ells, C

def cov_of(params, lmin=2, lmax=767, lensed=True, pol=False, noise=None, feature=None):
    """Build C from a parameter dict via CAMB. pol=True -> T+E 2x2, else T-only."""
    s = get_spectra(params, lmax, lensed, feature)
    sl = slice(lmin, lmax + 1); ells = s['ell'][sl]
    nz = (noise[0][sl], noise[1][sl]) if noise is not None else None
    if pol:
        return cov_from_arrays(ells, s['TT'][sl], s['EE'][sl], s['TE'][sl], noise=nz)
    nzT = (noise[0][sl], None) if noise is not None else None
    return cov_from_arrays(ells, s['TT'][sl], noise=nzT)

# ======================================================================
# 3.  Fisher matrix + constraints
# ======================================================================
def fisher(fid, steps, lmin=2, lmax=767, lensed=True, pol=False, noise=None, fsky=1.0):
    """Fisher over the parameters in `steps` (dict name->finite-diff step),
       evaluated at fiducial dict `fid` (must contain every key in `steps`)."""
    ells, C = cov_of(fid, lmin, lmax, lensed, pol, noise)
    Cinv = np.linalg.inv(C); names = list(steps); dC = []
    for k in names:
        fp, fm = dict(fid), dict(fid); fp[k] += steps[k]; fm[k] -= steps[k]
        _, Cp = cov_of(fp, lmin, lmax, lensed, pol, noise)
        _, Cm = cov_of(fm, lmin, lmax, lensed, pol, noise)
        dC.append((Cp - Cm) / (2 * steps[k]))
    M = [np.einsum('lij,ljk->lik', Cinv, d) for d in dC]
    pref = fsky * (2 * ells + 1) / 2.0
    n = len(names); F = np.zeros((n, n))
    for a in range(n):
        for b in range(a, n):
            F[a, b] = F[b, a] = np.sum(pref * np.einsum('lij,lji->l', M[a], M[b]))
    return names, F

def constraints(names, F):
    Finv = np.linalg.inv(F); sig = np.sqrt(np.diag(Finv))
    Dn = np.diag(1 / np.sqrt(np.diag(F)))
    evals, evecs = np.linalg.eigh(Dn @ F @ Dn)
    return dict(sigma=dict(zip(names, sig)), sigma_vec=sig, cov=Finv,
                corr=Finv / np.outer(sig, sig), cond=np.linalg.cond(F),
                norm_eigs=evals, evecs=evecs, names=list(names))

# ======================================================================
# 4.  Exact Gaussian likelihood + discrimination distance
# ======================================================================
def m2lnL(Chat, Cmodel, ells, fsky=1.0):
    """-2 ln L (up to const) of observed spectra Chat given model covariance."""
    Ci = np.linalg.inv(Cmodel); p = Cmodel.shape[-1]
    tr = np.einsum('lij,lji->l', Ci, Chat)
    _, ldM = np.linalg.slogdet(Cmodel); _, ldH = np.linalg.slogdet(Chat)
    return np.sum(fsky * (2 * ells + 1) * (tr + ldM - ldH - p))

def kl_separability(C1, C2, ells, fsky=1.0):
    """2 D_KL(1||2); sqrt is the sigma-equivalent separation between two models."""
    A = np.einsum('lij,ljk->lik', np.linalg.inv(C2), C1); p = C1.shape[-1]
    _, ld = np.linalg.slogdet(A)
    return np.sum(fsky * (2 * ells + 1) * (np.einsum('lii->l', A) - ld - p))

# ======================================================================
# 5.  Optimal (Neyman-Pearson) ROC / AUC  +  your network's AUC
# ======================================================================
def _draw(C, ells, rng):
    nu = (2 * ells + 1).astype(int); out = np.empty_like(C)
    for i in range(len(ells)):
        out[i] = wishart.rvs(df=nu[i], scale=C[i] / nu[i], random_state=rng)
    return out

def _score(Chat, C_A, C_B, ells, fsky):
    return -0.5 * (m2lnL(Chat, C_A, ells, fsky) - m2lnL(Chat, C_B, ells, fsky))

def optimal_auc(C_A, C_B, ells, nsim=400, fsky=1.0, seed=0):
    """Optimal LLR-classifier AUC between two model covariances (the ceiling)."""
    rng = np.random.default_rng(seed)
    lA = np.array([_score(_draw(C_A, ells, rng), C_A, C_B, ells, fsky) for _ in range(nsim)])
    lB = np.array([_score(_draw(C_B, ells, rng), C_A, C_B, ells, fsky) for _ in range(nsim)])
    r = rankdata(np.concatenate([lA, lB]))
    return (r[:nsim].sum() - nsim * (nsim + 1) / 2) / (nsim * nsim)

def auc_from_scores(scores, labels):
    """AUC of YOUR network scores (higher => class 1); labels in {0,1}.
       Identical statistic to optimal_auc, so the two are directly comparable."""
    scores = np.asarray(scores, float); labels = np.asarray(labels).astype(bool)
    n1 = labels.sum(); n0 = (~labels).sum(); r = rankdata(scores)
    return (r[labels].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)

# ======================================================================
# 6.  Validation of YOUR latent estimates against the bound
# ======================================================================
def recovery_report(theta_true, theta_est, sigma, names):
    """Per-parameter bias, RMS error, and RMS / Fisher-sigma.
       theta_true,(M,P): truth for each test sim; theta_est,(M,P): latent estimate.
       sigma: dict name->Fisher sigma from constraints().
       z_std ~ 1 -> reaching the bound; > 1 -> worse than optimal;
       < 1 -> impossible for real information => leakage flag."""
    tt = np.atleast_2d(theta_true); te = np.atleast_2d(theta_est); res = te - tt
    out = {}
    for i, n in enumerate(names):
        r = res[:, i]; sf = sigma[n]
        out[n] = dict(bias=float(r.mean()), rms=float(np.sqrt((r ** 2).mean())),
                      fisher_sigma=float(sf),
                      rms_over_fisher=float(np.sqrt((r ** 2).mean()) / sf),
                      z_std=float(r.std() / sf))
    return out

def coverage(theta_true, theta_est, theta_err, names, nsig=1.0):
    """Fraction with |estimate-truth| < nsig * reported error. Target ~0.68 (nsig=1)."""
    tt = np.atleast_2d(theta_true); te = np.atleast_2d(theta_est); er = np.atleast_2d(theta_err)
    z = np.abs(te - tt) / er
    return {n: float((z[:, i] < nsig).mean()) for i, n in enumerate(names)}

# ======================================================================
# 7.  Per-sim exact-likelihood posterior (CAMB for C(theta); emcee optional)
# ======================================================================
def posterior_mcmc(Chat, ells, fid, names, theta0, lensed=True, pol=False,
                   noise=None, fsky=1.0, bounds=None, nwalkers=32, nsteps=1500,
                   burn=500, seed=0, laplace=False):
    """Exact-likelihood posterior for ONE observed spectrum Chat (L,p,p) aligned
       with `ells`.  Varies the parameters in `names` about fiducial dict `fid`,
       rebuilding C(theta) via CAMB.  Returns a chain (nsamp, P).

       NOTE: each likelihood evaluation is one CAMB call (seconds), so a full
       emcee run is thousands of CAMB calls -- fine for a handful of validation
       sims run offline, but for many sims use an emulator (cobaya's CAMB, or a
       trained emulator such as CosmoPower) as the C(theta) provider instead.
       laplace=True returns the fast Fisher-Laplace Gaussian (no CAMB loop)."""
    lmin, lmax = int(ells[0]), int(ells[-1])
    steps = {n: _DEFAULT_STEP.get(n, 0.01 * abs(theta0[i]) + 1e-4)
             for i, n in enumerate(names)}
    if laplace:
        _, F = fisher(fid, steps, lmin, lmax, lensed, pol, noise, fsky)
        return np.random.default_rng(seed).multivariate_normal(
            np.array(theta0), np.linalg.inv(F), size=4000)
    def logpost(th):
        if bounds is not None and (np.any(th < bounds[0]) or np.any(th > bounds[1])):
            return -np.inf
        d = dict(fid)
        for n, v in zip(names, th):
            d[n] = v
        try:
            _, C = cov_of(d, lmin, lmax, lensed, pol, noise)
        except Exception:
            return -np.inf
        return -0.5 * m2lnL(Chat, C, ells, fsky)
    _, F = fisher(fid, steps, lmin, lmax, lensed, pol, noise, fsky)
    step = np.sqrt(np.diag(np.linalg.inv(F)))
    try:
        import emcee
        rng = np.random.default_rng(seed)
        p0 = np.array(theta0) + 0.1 * step * rng.standard_normal((nwalkers, len(names)))
        sm = emcee.EnsembleSampler(nwalkers, len(names), logpost)
        sm.run_mcmc(p0, nsteps, progress=False)
        return sm.get_chain(discard=burn, flat=True)
    except ImportError:
        return np.random.default_rng(seed).multivariate_normal(
            np.array(theta0), np.linalg.inv(F), size=4000)

if __name__ == '__main__':
    print("Operates on your arrays + CAMB. Run demo_cmb_baseline.py for a worked example.")

"""CMB spectra + map generation, factored out of examples/gen_spectra_maps.ipynb.

The notebook's generation loops grew from a quick visualization scratchpad into a
large pipeline across five simulation sets (Planck LCDM, w0wa, oscillations,
early dark energy, hemispherical asymmetry). That doesn't belong in a live kernel, so the
physics/sampling logic lives here where it can be driven by a parallel runner
(see examples/generate_sims_parallel.py) instead of one sequential notebook loop.
Behaviour is unchanged from the notebook version — same CAMB calls, same map
synthesis order, same LHC samplers — just parameterized via an explicit
PipelineConfig instead of notebook globals, so it's safe to call from worker
processes.
"""
import os
from collections import namedtuple
from dataclasses import dataclass
from typing import Optional

import numpy as np
import healpy as hp
import camb
from scipy.stats import qmc

from .pk import PK as _PK

CMBSpectra = namedtuple('CMBSpectra', ['ell', 'tt', 'te', 'ee', 'bb', 'theta_star'])

# w_n (effective equation of state once the axion field starts oscillating) is not a
# free nuisance parameter of AxionEffectiveFluid: it is fixed by the power n of the
# potential V(phi) ∝ [1-cos(phi/f)]^n via w_n=(n-1)/(n+1), a discrete/theoretical choice
# rather than a continuous one to put a prior on. n=3 -> w_n=1/2 is the standard
# benchmark used across the EDE literature (Poulin, Smith, Karwal & Kamionkowski 2018,
# arXiv:1806.10608; Smith et al. 2020; Hill et al. 2020) — fixed here, not sampled.
EDE_W_N = 0.5


@dataclass
class PipelineConfig:
    """Everything a worker process needs to run one realisation end-to-end."""
    nside: int
    lmax_map: int
    use_lensed_spectra: bool
    include_noise: bool
    include_galactic_mask: bool
    include_polarization: bool
    noise_ukarcmin_t: float
    noise_ukarcmin_p: float
    noise_seed_offset: int
    fwhm_rad: float
    tau_fixed: float
    mask_int: Optional[np.ndarray] = None
    mask_pol: Optional[np.ndarray] = None
    pix_vec: Optional[np.ndarray] = None  # (3, npix) unit vectors, hp.pix2vec — for hemispherical modulation


# ── CAMB dispatch ──────────────────────────────────────────────────────────────

def _set_dark_energy(params, w0, wa, ede_params):
    """Attach the DE sector: AxionEffectiveFluid if ede_params given, else PPF(w0,wa).

    ede_params, when given, is (w_n, fde_zc, zc, theta_i) and fully REPLACES the
    w0/wa dark energy sector (AxionEffectiveFluid models the whole DE fluid, so
    w0/wa are not meaningful alongside it).
    """
    if ede_params is not None:
        w_n, fde_zc, zc, theta_i = ede_params
        params.DarkEnergy = camb.dark_energy.AxionEffectiveFluid()
        params.DarkEnergy.set_params(w_n=w_n, fde_zc=fde_zc, zc=zc, theta_i=theta_i)
    else:
        params.DarkEnergy = camb.dark_energy.DarkEnergyPPF()
        params.DarkEnergy.set_params(w=w0, wa=wa)


def _lensed_spectra(H0, ombh2, omch2, mnu, omk, tau, As, ns,
                    lmax=2507, halofit_version='mead2020_feedback',
                    w0=-1.0, wa=0.0,
                    custom_PK=False, amp=0, freq=0, wid=1, centre=0.05, phase=0,
                    ede_params=None):
    """Lensed CMB Dl spectra (μK²)."""
    params = camb.set_params(
        H0=H0, ombh2=ombh2, omch2=omch2, mnu=mnu, omk=omk, tau=tau,
        As=As, ns=ns, lmax=lmax, halofit_version=halofit_version,
    )
    _set_dark_energy(params, w0, wa, ede_params)
    if custom_PK:
        params.set_initial_power_function(
            _PK, args=(As, ns, amp, freq, wid, centre, phase),
            effective_ns_for_nonlinear=ns,
        )
    else:
        params.InitPower.set_params(As=As, ns=ns)
    results    = camb.get_results(params)
    lensed     = results.get_cmb_power_spectra(params, CMB_unit='muK')['lensed_scalar']
    ell        = np.arange(len(lensed))
    theta_star = results.get_derived_params()['thetastar']
    return CMBSpectra(ell, lensed[:, 0], lensed[:, 3], lensed[:, 1], lensed[:, 2], theta_star)


def _unlensed_spectra(H0, ombh2, omch2, mnu, omk, tau, As, ns,
                      lmax=2507, w0=-1.0, wa=0.0,
                      custom_PK=False, amp=0, freq=0, wid=1, centre=0.05, phase=0,
                      ede_params=None):
    """Unlensed CMB Dl spectra (BB = 0 identically)."""
    params = camb.set_params(
        H0=H0, ombh2=ombh2, omch2=omch2, mnu=mnu, omk=omk, tau=tau,
        As=As, ns=ns, lmax=lmax,
    )
    _set_dark_energy(params, w0, wa, ede_params)
    if custom_PK:
        params.set_initial_power_function(
            _PK, args=(As, ns, amp, freq, wid, centre, phase),
            effective_ns_for_nonlinear=ns,
        )
    else:
        params.InitPower.set_params(As=As, ns=ns)
    results    = camb.get_results(params)
    unlensed   = results.get_cmb_power_spectra(params, CMB_unit='muK')['unlensed_scalar']
    ell        = np.arange(len(unlensed))
    bb         = np.zeros(len(ell))
    theta_star = results.get_derived_params()['thetastar']
    return CMBSpectra(ell, unlensed[:, 0], unlensed[:, 3], unlensed[:, 1], bb, theta_star)


def generate_spectra(cfg, H0, ombh2, omch2, mnu, omk, tau, As, ns,
                     lmax=2507, w0=-1.0, wa=0.0,
                     custom_PK=False, amp=0, freq=0, wid=1, centre=0.05, phase=0,
                     ede_params=None):
    """Dispatch to lensed or unlensed CAMB spectra based on cfg.use_lensed_spectra."""
    fn = _lensed_spectra if cfg.use_lensed_spectra else _unlensed_spectra
    return fn(H0, ombh2, omch2, mnu, omk, tau, As, ns,
              lmax=lmax, w0=w0, wa=wa, custom_PK=custom_PK, amp=amp, freq=freq,
              wid=wid, centre=centre, phase=phase, ede_params=ede_params)


def dl_to_cl(dl_camb, lmax_map):
    """Convert a CAMB D_ell array (μK², indexed from ell=0) to a C_ell array of
    length lmax_map+1 for synfast."""
    ell = np.arange(lmax_map + 1, dtype=float)
    dl  = np.zeros(lmax_map + 1)
    k   = min(len(dl_camb), lmax_map + 1)
    dl[:k] = dl_camb[:k]
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(ell > 1, 2 * np.pi * dl / (ell * (ell + 1)), 0.0)


def save_spectra(cfg, sp, cl_tt, cl_ee, cl_te, cl_bb, out_dir, tag):
    """Save ell, Cl and Dl arrays as a .npz in a spectra/ subdirectory."""
    spectra_dir = os.path.join(out_dir, 'spectra')
    os.makedirs(spectra_dir, exist_ok=True)
    n   = cfg.lmax_map + 1
    ell = np.arange(n)
    data = {
        'ell':   ell,
        'cl_tt': cl_tt,
        'dl_tt': sp.tt[:n],
    }
    if cfg.include_polarization:
        data.update({
            'cl_ee': cl_ee, 'dl_ee': sp.ee[:n],
            'cl_te': cl_te, 'dl_te': sp.te[:n],
            'cl_bb': cl_bb, 'dl_bb': sp.bb[:n],
        })
    np.savez(os.path.join(spectra_dir, f'spectra_{tag}.npz'), **data)


def make_and_save_maps(cfg, cl_tt, cl_ee, cl_bb, cl_te, seed_i, out_dir, tag,
                       asymmetry_params=None):
    """Synthesise CMB maps and save as float32 FITS.
    Order: signal (synfast) -> beam -> optional hemispherical modulation ->
    white noise -> mask -> optional E/B.

    Saves T only, or T+Q+U+E+B when cfg.include_polarization — Q/U (needed for
    anisotropic-cosmology analyses) and E/B (needed for standard cosmological
    analyses) are both kept, rather than deferring E/B to a later on-demand
    map2alm/alm2map pass.

    asymmetry_params, when given as (A, l_deg, b_deg), multiplies the
    beam-smoothed (still-isotropic) signal by a dipolar modulation field
    1 + A*(n_hat . pix_hat) BEFORE noise/mask — the asymmetry is a property of
    the cosmological signal, not of the (isotropic) instrument noise or the
    (survey-geometry) mask, so it must not touch either of those. cfg.pix_vec
    is precomputed once (hp.pix2vec over all pixels) and reused across every
    realisation rather than recomputed per task.
    """
    nside = cfg.nside
    npix  = hp.nside2npix(nside)
    n     = len(cl_tt)

    np.random.seed(seed_i)
    T, Q, U = hp.synfast(
        [cl_tt, cl_ee, cl_bb, cl_te, np.zeros(n), np.zeros(n)],
        nside=nside, new=True, pol=True, lmax=cfg.lmax_map,
    )
    T, Q, U = hp.smoothing([T, Q, U], fwhm=cfg.fwhm_rad, pol=True)

    if asymmetry_params is not None:
        A, l_deg, b_deg = asymmetry_params
        lam_hat    = hp.ang2vec(l_deg, b_deg, lonlat=True)
        modulation = 1.0 + A * (lam_hat @ cfg.pix_vec)
        T = T * modulation
        Q = Q * modulation
        U = U * modulation

    if cfg.include_noise:
        rng_noise  = np.random.default_rng(seed_i + cfg.noise_seed_offset)
        pix_arcmin = np.degrees(hp.nside2resol(nside)) * 60.0
        sigma_T    = cfg.noise_ukarcmin_t / pix_arcmin
        sigma_P    = cfg.noise_ukarcmin_p / pix_arcmin
        T = T + rng_noise.normal(0.0, sigma_T, npix)
        Q = Q + rng_noise.normal(0.0, sigma_P, npix)
        U = U + rng_noise.normal(0.0, sigma_P, npix)

    if cfg.include_galactic_mask:
        T = T * cfg.mask_int
        Q = Q * cfg.mask_pol
        U = U * cfg.mask_pol

    if cfg.include_polarization:
        _, alm_E, alm_B = hp.map2alm([T, Q, U], pol=True)
        E = hp.alm2map(alm_E, nside=nside)
        B = hp.alm2map(alm_B, nside=nside)
        comps = [('T', T), ('Q', Q), ('U', U), ('E', E), ('B', B)]
    else:
        comps = [('T', T)]

    for comp, m in comps:
        hp.write_map(os.path.join(out_dir, f'{comp}_{tag}.fits'),
                     m, dtype=np.float32, overwrite=True)


def run_task(cfg, task):
    """Run one realisation end-to-end and return its metadata row.

    task is a plain (picklable) dict with keys 'out_dir', 'tag', 'seed',
    'camb_kwargs' (passed to generate_spectra), 'meta' (metadata fields to
    echo back verbatim on success — 'theta_star' is added once computed), and
    optionally 'asymmetry_params' (passed through to make_and_save_maps).

    Idempotent: if the T FITS output already exists, returns None without
    recomputing — lets an interrupted/crashed run be resumed by simply
    re-launching the script rather than needing separate checkpoint plumbing.
    """
    T_path = os.path.join(task['out_dir'], f"T_{task['tag']}.fits")
    if os.path.exists(T_path):
        return None

    sp = generate_spectra(cfg, **task['camb_kwargs'])
    cl_tt = dl_to_cl(sp.tt, cfg.lmax_map)
    cl_ee = dl_to_cl(sp.ee, cfg.lmax_map)
    cl_te = dl_to_cl(sp.te, cfg.lmax_map)
    cl_bb = dl_to_cl(sp.bb, cfg.lmax_map)
    make_and_save_maps(cfg, cl_tt, cl_ee, cl_bb, cl_te,
                       task['seed'], task['out_dir'], task['tag'],
                       asymmetry_params=task.get('asymmetry_params'))
    save_spectra(cfg, sp, cl_tt, cl_ee, cl_te, cl_bb, task['out_dir'], task['tag'])

    meta = dict(task['meta'])
    meta['theta_star'] = sp.theta_star
    return meta


# ── LHC samplers (identical to gen_spectra_maps.ipynb Cell "cell-lhc") ────────

def _scale_uniform(u, lo, hi):
    return lo + u * (hi - lo)


def _scale_loguniform(u, lo, hi):
    return np.exp(np.log(lo) + u * (np.log(hi) - np.log(lo)))


def _lhc_tau(u_col, include_polarization, tau_fixed):
    """tau column, gated by include_polarization (see notebook's _lhc_tau)."""
    if include_polarization:
        return _scale_uniform(u_col, 0.02, 0.10)
    return np.full(u_col.shape, tau_fixed)


def lhc_lcdm_6d(n, cfg, seed):
    """6-D LHC: omega_b, omega_cdm, H0, ns, As, tau."""
    sampler = qmc.LatinHypercube(d=6, seed=seed)
    u = sampler.random(n=n)
    return {
        'ombh2': _scale_uniform(   u[:, 0], 0.018,  0.028),
        'omch2': _scale_uniform(   u[:, 1], 0.09,   0.15),
        'H0':    _scale_uniform(   u[:, 2], 60.0,   80.0),
        'ns':    _scale_uniform(   u[:, 3], 0.90,   1.10),
        'As':    _scale_loguniform(u[:, 4], 1.5e-9, 2.5e-9),
        'tau':   _lhc_tau(u[:, 5], cfg.include_polarization, cfg.tau_fixed),
    }


def lhc_w0wa_8d(n, cfg, seed):
    """8-D LHC: 5 LCDM params + w0 + wa + tau. Resamples w0+wa>0 (unphysical)."""
    sampler = qmc.LatinHypercube(d=8, seed=seed)
    u = sampler.random(n=n)
    w0 = _scale_uniform(u[:, 5], -1.5, -0.5)
    wa = _scale_uniform(u[:, 6], -3.0,  1.0)

    rng = np.random.default_rng(seed + 100_000)
    bad = w0 + wa > 0
    while np.any(bad):
        idx = np.flatnonzero(bad)
        w0[idx] = rng.uniform(-1.5, -0.5, size=idx.size)
        wa[idx] = rng.uniform(-3.0,  1.0, size=idx.size)
        bad = w0 + wa > 0

    return {
        'ombh2': _scale_uniform(   u[:, 0], 0.018,  0.028),
        'omch2': _scale_uniform(   u[:, 1], 0.09,   0.15),
        'H0':    _scale_uniform(   u[:, 2], 60.0,   80.0),
        'ns':    _scale_uniform(   u[:, 3], 0.90,   1.10),
        'As':    _scale_loguniform(u[:, 4], 1.5e-9, 2.5e-9),
        'w0':    w0,
        'wa':    wa,
        'tau':   _lhc_tau(u[:, 7], cfg.include_polarization, cfg.tau_fixed),
    }


def lhc_osc_8d(n, cfg, seed):
    """8-D LHC: 4 LCDM params (H0 fixed elsewhere) + A_lin + freq + phase + tau."""
    sampler = qmc.LatinHypercube(d=8, seed=seed)
    u = sampler.random(n=n)
    return {
        'ombh2': _scale_uniform(   u[:, 0], 0.018,  0.028),
        'omch2': _scale_uniform(   u[:, 1], 0.09,   0.15),
        'ns':    _scale_uniform(   u[:, 2], 0.90,   1.10),
        'As':    _scale_loguniform(u[:, 3], 1.5e-9, 2.5e-9),
        'A_lin': _scale_uniform(   u[:, 4], 0.01,   0.06),
        'freq':  _scale_uniform(   u[:, 5], 20.0,   40.0),
        'phase': _scale_uniform(   u[:, 6], 0.0,    2 * np.pi),
        'tau':   _lhc_tau(u[:, 7], cfg.include_polarization, cfg.tau_fixed),
    }


def lhc_ede_9d(n, cfg, seed):
    """9-D LHC: 5 LCDM params (H0 free) + f_EDE + theta_i + logzc + tau.

    f_EDE is floored at 1e-4, not the prior's literal 0.0 — CAMB's
    AxionEffectiveFluid hard-crashes (native abort) at fde_zc=0.0 exactly.
    w_n is NOT sampled — see EDE_W_N above.
    """
    sampler = qmc.LatinHypercube(d=9, seed=seed)
    u = sampler.random(n=n)
    f_ede = np.maximum(_scale_uniform(u[:, 5], 0.0, 0.5), 1e-4)
    return {
        'ombh2':   _scale_uniform(   u[:, 0], 0.018,  0.028),
        'omch2':   _scale_uniform(   u[:, 1], 0.09,   0.15),
        'H0':      _scale_uniform(   u[:, 2], 60.0,   80.0),
        'ns':      _scale_uniform(   u[:, 3], 0.90,   1.10),
        'As':      _scale_loguniform(u[:, 4], 1.5e-9, 2.5e-9),
        'f_ede':   f_ede,
        'theta_i': _scale_uniform(u[:, 6], 0.1, 3.1),
        'logzc':   _scale_uniform(u[:, 7], 3.0, 4.3),
        'tau':     _lhc_tau(u[:, 8], cfg.include_polarization, cfg.tau_fixed),
    }


def lhc_hemispherical_8d(n, cfg, seed):
    """8-D LHC: 4 LCDM params (H0 fixed elsewhere — same 'clean discrimination
    regime' rationale as lhc_osc_8d, since this is a real-space anomaly added
    on top of a baseline LCDM sky rather than something coupled to the
    background expansion the way EDE is) + A + l_deg + sin(b) + tau.

    A (dipolar modulation amplitude) ∈ [0, 0.30], uniform, amplitude-ONLY
    (non-negative) — direction is independently randomized every draw (see
    below), so a fixed sign convention for A loses no generality; letting it
    go negative would just double-cover directions already covered by their
    antipode.

    Direction (l_deg, b_deg) is sampled uniform over the FULL SPHERE, not a
    fixed literature axis: training only ever hitting one preferred direction
    risks the network learning "check these specific pixels" rather than a
    general angular-modulation detector. Uniform-on-sphere requires l ~
    Uniform(0, 360) but b NOT uniform in degrees (that oversamples the poles)
    — sin(b) ~ Uniform(-1, 1) gives the equal-area distribution, then
    b = arcsin(sin_b).
    """
    sampler = qmc.LatinHypercube(d=8, seed=seed)
    u = sampler.random(n=n)
    l_deg = _scale_uniform(u[:, 5], 0.0, 360.0)
    sin_b = _scale_uniform(u[:, 6], -1.0, 1.0)
    b_deg = np.degrees(np.arcsin(sin_b))
    return {
        'ombh2': _scale_uniform(   u[:, 0], 0.018,  0.028),
        'omch2': _scale_uniform(   u[:, 1], 0.09,   0.15),
        'ns':    _scale_uniform(   u[:, 2], 0.90,   1.10),
        'As':    _scale_loguniform(u[:, 3], 1.5e-9, 2.5e-9),
        'A':     _scale_uniform(   u[:, 4], 0.0,    0.30),
        'l_deg': l_deg,
        'b_deg': b_deg,
        'tau':   _lhc_tau(u[:, 7], cfg.include_polarization, cfg.tau_fixed),
    }

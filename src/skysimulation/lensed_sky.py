"""Temperature-only lensed CMB skies, anomaly injection and augmented views.

Two stages, split by which frame each ingredient lives in (see
examples/contrastive_poc/lensed_sky_contrastive_views.ipynb for the walkthrough):

Stage 1 (sky frame, once per universe)
    CAMB spectra -> correlated Gaussian (T, φ) draw -> [anomaly / modulation on the
    primordial, i.e. unlensed, T] -> lenspyx remapping -> lensed T alm.

Stage 2 (per view)
    random rotation of the alm -> beam -> alm2map -> + anisotropic noise ->
    × Galactic mask. Noise and mask stay fixed in Galactic coordinates.

Anomalies are primordial sky physics: they act on the unlensed T (via the
`unlensed_hook` of `lensed_T_alm`) and are then lensed like everything else. The
injectors work at any band limit, so they apply to the unlensed alm (lmax_unl).

For fixed φ, lensing is linear in T: lens(T + ε Δ) = lens(T) + ε lens(Δ). Hence a
modulated sky only costs one extra lensing of Δ = T·M, and any ε follows for free.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import healpy as hp
import camb
import ducc0
import lenspyx
from lenspyx.utils_hp import almxfl
from scipy.spatial.transform import Rotation

from .maps import new_resolution_mask

# Planck 2018 TT,TE,EE+lowE+lensing best fit (ln10^10 As = 3.044).
FIDUCIAL = dict(H0=67.32, ombh2=0.02237, omch2=0.1200, ns=0.9649,
                As=np.exp(3.044) * 1e-10, tau=0.0544, mnu=0.06, omk=0.0)


@dataclass(frozen=True)
class SkyConfig:
    nside: int = 64
    lmax_unl: int = 1200      # unlensed band limit; buffer above lmax for the lensing mode coupling
    nside_lens: int = 512     # grid the lensed sky is evaluated on before going back to alms

    @property
    def lmax(self) -> int:
        return 3 * self.nside - 1


# ═══════════════════════════════ Stage 1 ═══════════════════════════════

def get_spectra_T(cfg: SkyConfig, params: dict = FIDUCIAL) -> dict:
    """Raw C_ell up to cfg.lmax_unl: unlensed TT, lensing potential φφ and φT
    (the late-ISW × lensing correlation), and lensed TT for validation."""
    pars = camb.set_params(
        H0=params["H0"], ombh2=params["ombh2"], omch2=params["omch2"], mnu=params["mnu"],
        omk=params["omk"], tau=params["tau"], As=params["As"], ns=params["ns"],
        lmax=cfg.lmax_unl + 500, lens_potential_accuracy=1, NonLinear="NonLinear_lens",
    )
    res = camb.get_results(pars)
    ps = res.get_cmb_power_spectra(pars, CMB_unit="muK", raw_cl=True, lmax=cfg.lmax_unl)
    return {"TT": ps["unlensed_scalar"][:, 0], "PP": ps["lens_potential"][:, 0],
            "PT": ps["lens_potential"][:, 1], "len_TT": ps["lensed_scalar"][:, 0]}


def get_spectra_T_extended(cfg: SkyConfig, params: dict, w0=None, wa=None, ede=None,
                           thetastar=None) -> dict:
    """Like get_spectra_T, for ΛCDM-adjacent models: w0wa dark energy (DarkEnergyPPF)
    or early dark energy (AxionEffectiveFluid, ede = (w_n, fde_zc, zc, theta_i)).

    With ``thetastar`` (θ*, not 100θ*), H0 is solved to hold the acoustic scale fixed;
    the dark-energy model is set *before* that solve so it enters the distance to last
    scattering. get_spectra_T is left untouched so ΛCDM sims stay reproducible."""
    pars = camb.CAMBparams()
    if ede is not None:
        w_n, fde_zc, zc, theta_i = ede
        pars.DarkEnergy = camb.dark_energy.AxionEffectiveFluid()
        pars.DarkEnergy.set_params(w_n=w_n, fde_zc=fde_zc, zc=zc, theta_i=theta_i)
    elif w0 is not None:
        pars.DarkEnergy = camb.dark_energy.DarkEnergyPPF()
        pars.DarkEnergy.set_params(w=w0, wa=0.0 if wa is None else wa)
    cosmo = dict(ombh2=params["ombh2"], omch2=params["omch2"], mnu=params["mnu"],
                 omk=params["omk"], tau=params["tau"])
    if thetastar is not None:
        pars.set_cosmology(H0=None, thetastar=thetastar, **cosmo)
    else:
        pars.set_cosmology(H0=params["H0"], **cosmo)
    pars.InitPower.set_params(As=params["As"], ns=params["ns"])
    pars.set_for_lmax(cfg.lmax_unl + 500, lens_potential_accuracy=1)
    pars.NonLinear = camb.model.NonLinear_lens
    res = camb.get_results(pars)
    ps = res.get_cmb_power_spectra(pars, CMB_unit="muK", raw_cl=True, lmax=cfg.lmax_unl)
    return {"TT": ps["unlensed_scalar"][:, 0], "PP": ps["lens_potential"][:, 0],
            "PT": ps["lens_potential"][:, 1], "len_TT": ps["lensed_scalar"][:, 0],
            "H0": pars.H0, "thetastar": res.get_derived_params()["thetastar"] / 100}


def unlensed_T_phi(cfg: SkyConfig, spec: dict, seed: int):
    """Unlensed T and lensing potential φ alms (lmax_unl), drawn jointly so the
    T–φ correlation survives; the ordering for hp.synalm(new=True) is [TT, φφ, Tφ].
    Deterministic in `seed`, so any stored sky can be redrawn exactly."""
    np.random.seed(int(seed))
    tlm, plm = hp.synalm([spec["TT"], spec["PP"], spec["PT"]], lmax=cfg.lmax_unl, new=True)
    return tlm, plm


def lens_T(cfg: SkyConfig, tlm, plm, nthreads: int = 1) -> np.ndarray:
    """Lens an unlensed T alm by φ; returns the lensed alm at cfg.lmax. Linear in tlm."""
    ells = np.arange(cfg.lmax_unl + 1, dtype=float)
    dlm = almxfl(plm, np.sqrt(ells * (ells + 1)), None, False)
    T = lenspyx.alm2lenmap(tlm, dlm, geometry=("healpix", {"nside": cfg.nside_lens}),
                           epsilon=1e-6, nthreads=nthreads, pol=False)
    return hp.map2alm(np.asarray(T).reshape(-1), lmax=cfg.lmax, iter=3)


def lensed_T_alm(cfg: SkyConfig, spec: dict, seed: int, nthreads: int = 1,
                 unlensed_hook=None) -> np.ndarray:
    """One lensed T realisation (alm at cfg.lmax). `unlensed_hook(tlm) -> tlm`, if
    given, modifies the primordial (unlensed) T before lensing — this is where
    anomalies are injected."""
    tlm, plm = unlensed_T_phi(cfg, spec, seed)
    if unlensed_hook is not None:
        tlm = unlensed_hook(tlm)
    return lens_T(cfg, tlm, plm, nthreads)


def work_nside(lmax: int) -> int:
    """Smallest power-of-2 nside with 2·nside > lmax (accurate map2alm up to lmax)."""
    return int(2 ** np.ceil(np.log2((lmax + 1) / 2)))


def _add_low(alm, small, lsmall):
    """alm (lmax) + small (lmax lsmall ≤ lmax), coefficient by coefficient."""
    lmax = hp.Alm.getlmax(alm.size)
    out = alm.copy()
    for m in range(lsmall + 1):
        ell = np.arange(m, lsmall + 1)
        out[hp.Alm.getidx(lmax, ell, m)] += small[hp.Alm.getidx(lsmall, ell, m)]
    return out


def random_direction(rng: np.random.Generator) -> np.ndarray:
    v = rng.standard_normal(3)
    return v / np.linalg.norm(v)


def _scale_ells(alm: np.ndarray, fl: np.ndarray) -> np.ndarray:
    return hp.almxfl(alm, fl)


def inject_hemispherical(alm, amplitude, direction, lmax_mod=64):
    """Dipolar modulation T -> T (1 + A n̂·d̂) of the large scales (ℓ ≤ lmax_mod),
    the scale range where the Planck asymmetry is seen."""
    lmax = hp.Alm.getlmax(alm.size)
    ns = work_nside(lmax_mod + 1)
    low = hp.alm2map(hp.resize_alm(alm, lmax, lmax, lmax_mod, lmax_mod), ns)
    vec = np.array(hp.pix2vec(ns, np.arange(hp.nside2npix(ns))))
    term = hp.map2alm(low * (direction @ vec), lmax=lmax_mod + 1, iter=3)
    return _add_low(alm, amplitude * term, lmax_mod + 1)


def inject_low_power(alm, suppression, lmax_low=30):
    """Scale the power of 2 ≤ ℓ ≤ lmax_low by (1 - suppression)."""
    fl = np.ones(hp.Alm.getlmax(alm.size) + 1)
    fl[2:lmax_low + 1] = np.sqrt(1.0 - suppression)
    return _scale_ells(alm, fl)


def inject_parity(alm, p, lmax_par=30):
    """Odd-ℓ power × (1 + p), even-ℓ power × (1 - p) for 2 ≤ ℓ ≤ lmax_par
    (odd-parity preference, as in the observed low-ℓ sky)."""
    fl = np.ones(hp.Alm.getlmax(alm.size) + 1)
    ell = np.arange(2, lmax_par + 1)
    fl[ell] = np.where(ell % 2 == 1, np.sqrt(1 + p), np.sqrt(1 - p))
    return _scale_ells(alm, fl)


def inject_planar_alignment(alm, alpha, rng, ells=(2, 3)):
    """Quadrupole–octopole alignment: mix each ℓ in `ells` with a planar multipole
    (power only in m = ±ℓ) about one common random axis,
        a_ℓ -> sqrt(1-α) a_ℓ + sqrt(α) a_ℓ^planar,
    then restore the realised Ĉ_ℓ, so only the phases change and the power
    spectrum is untouched. α = 1 is a perfectly planar, perfectly aligned pair."""
    lmax = hp.Alm.getlmax(alm.size)
    lsmall = max(ells)
    planar = np.zeros(hp.Alm.getsize(lsmall), dtype=np.complex128)
    cl = hp.alm2cl(alm)
    for ell in ells:  # |a_ℓℓ|² counted twice (m = ±ℓ): 2|a_ℓℓ|² = (2ℓ+1) Ĉ_ℓ
        planar[hp.Alm.getidx(lsmall, ell, ell)] = (
            np.sqrt(cl[ell] * (2 * ell + 1) / 2) * np.exp(2j * np.pi * rng.uniform()))
    psi, theta, phi = Rotation.random(rng=rng).as_euler("zyz")
    planar = ducc0.sht.rotate_alm(planar, lsmall, psi, theta, phi)

    out = alm.copy()
    for ell in ells:
        idx = hp.Alm.getidx(lmax, ell, np.arange(ell + 1))
        mixed = np.sqrt(1 - alpha) * alm[idx] + np.sqrt(alpha) * planar[
            hp.Alm.getidx(lsmall, ell, np.arange(ell + 1))]
        cl_mixed = (np.abs(mixed[0]) ** 2 + 2 * np.sum(np.abs(mixed[1:]) ** 2)) / (2 * ell + 1)
        out[idx] = mixed * np.sqrt(cl[ell] / cl_mixed)
    return out


def inject_cold_spot(alm, amplitude_uK, direction, sigma_deg=5.0, lmax_spot=300):
    """Gaussian cold spot ΔT = -A exp(-θ²/2σ²) centred on `direction`. The σ = 5°
    profile has negligible power above ℓ ~ 150, so it is expanded to lmax_spot only."""
    lsp = min(hp.Alm.getlmax(alm.size), lmax_spot)
    ns = work_nside(lsp)
    vec = np.array(hp.pix2vec(ns, np.arange(hp.nside2npix(ns))))
    theta = np.arccos(np.clip(direction @ vec, -1, 1))
    spot = -amplitude_uK * np.exp(-0.5 * (theta / np.radians(sigma_deg)) ** 2)
    return _add_low(alm, hp.map2alm(spot, lmax=lsp, iter=3), lsp)


def rescale_to_cl(alm, cl_target, lmin=2):
    """Rescale each ℓ ≥ lmin so the realised Ĉ_ℓ equals cl_target[ℓ] exactly.
    Removes the realisation's power-spectrum fingerprint; phases are untouched."""
    lmax = hp.Alm.getlmax(alm.size)
    cl = hp.alm2cl(alm)
    fl = np.ones(lmax + 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        fl[lmin:] = np.sqrt(cl_target[lmin:lmax + 1] / cl[lmin:])
    return hp.almxfl(alm, np.nan_to_num(fl, nan=1.0, posinf=1.0))


def random_modulation_field(L_dev, rng, lmin=2):
    """alm (lmax L_dev) of a Gaussian random field M with power only at
    lmin ≤ L ≤ L_dev, a random spectrum (independent Exp(1) weight per L), and
    unit pixel variance on the sphere. Statistically isotropic, so its 'axis' is
    random by construction."""
    cl = np.zeros(L_dev + 1)
    cl[lmin:] = rng.exponential(size=L_dev + 1 - lmin)
    # draw a_LM directly from rng (hp.synalm would use numpy's global state)
    alm = np.zeros(hp.Alm.getsize(L_dev), dtype=np.complex128)
    for L in range(lmin, L_dev + 1):
        idx = hp.Alm.getidx(L_dev, L, np.arange(L + 1))
        re, im = rng.standard_normal(L + 1), rng.standard_normal(L + 1)
        a = np.sqrt(cl[L] / 2) * (re + 1j * im)
        a[0] = np.sqrt(cl[L]) * re[0]
        alm[idx] = a
    return alm / hp.alm2map(alm, 32).std()


def random_remap_field(rng, lmin=2, lmax_range=(4, 48), slope_range=(0.0, 4.0)):
    """alm of a random gradient-remap potential ψ, for T(n̂) -> T(n̂ + ∇ψ).

    Power only at lmin ≤ L ≤ L_max (L_max uniform in lmax_range), with a power-law
    spectrum C_L ∝ L^(−n) (n uniform in slope_range) and random phases. Normalised
    to an rms deflection |∇ψ| of 1 degree, so a remap amplitude ε is the rms
    deflection in degrees. lmin = 2 excludes the L = 1 (boost/aberration-like)
    remap; gradient remaps never contain rigid rotations (those are curl modes).
    Returns (psi_alm, L_max, slope)."""
    L_max = int(rng.integers(lmax_range[0], lmax_range[1] + 1))
    slope = float(rng.uniform(*slope_range))
    L = np.arange(L_max + 1, dtype=float)
    cl = np.zeros(L_max + 1)
    cl[lmin:] = L[lmin:] ** (-slope)
    alm = np.zeros(hp.Alm.getsize(L_max), dtype=np.complex128)
    for ell in range(lmin, L_max + 1):
        idx = hp.Alm.getidx(L_max, ell, np.arange(ell + 1))
        re, im = rng.standard_normal(ell + 1), rng.standard_normal(ell + 1)
        a = np.sqrt(cl[ell] / 2) * (re + 1j * im)
        a[0] = np.sqrt(cl[ell]) * re[0]
        alm[idx] = a
    # rms deflection² = Σ_L (2L+1)/(4π) L(L+1) Ĉ_L of this realisation
    cl_real = hp.alm2cl(alm)
    rms = np.sqrt(np.sum((2 * L + 1) / (4 * np.pi) * L * (L + 1) * cl_real))
    return alm * (np.radians(1.0) / rms), L_max, slope


def remap_term(tlm, psi_alm):
    """First-order remap Δ = ∇T·∇ψ as an alm at the band limit of tlm, so that
    T(n̂ + ε∇ψ) ≈ tlm + εΔ. Gradients from healpy (∂θ and ∂φ/sinθ components)."""
    lmax = hp.Alm.getlmax(tlm.size)
    L_psi = hp.Alm.getlmax(psi_alm.size)
    ns = work_nside(lmax + L_psi)
    _, t_th, t_ph = hp.alm2map_der1(tlm, ns)
    _, p_th, p_ph = hp.alm2map_der1(psi_alm, ns, lmax=L_psi)
    return hp.map2alm(t_th * p_th + t_ph * p_ph, lmax=lmax, iter=1)


def lens_T_with_remap(cfg, tlm, plm, psi_alm, eps, nthreads=1):
    """Exact reference for validation: remap by ε∇ψ and lens by ∇φ in one deflection
    ∇(φ + εψ) — equal to 'remap the primordial T, then lens' up to O(ε ∇∇ψ·∇φ)."""
    L_psi = hp.Alm.getlmax(psi_alm.size)
    return lens_T(cfg, tlm, _add_low(plm, eps * psi_alm, L_psi), nthreads)


def modulation_term(tlm, M_alm):
    """Δ = T·M as an alm at the band limit of tlm, so that T(1 + εM) = tlm + εΔ.
    Evaluated on a grid fine enough for the product's extra L_dev of bandwidth."""
    lmax = hp.Alm.getlmax(tlm.size)
    ns = work_nside(lmax + hp.Alm.getlmax(M_alm.size))
    return hp.map2alm(hp.alm2map(tlm, ns) * hp.alm2map(M_alm, ns), lmax=lmax, iter=1)


# ═══════════════════════════════ Stage 2 ═══════════════════════════════

@dataclass
class ViewModel:
    """Everything fixed in the Galactic frame, plus the (isotropic) beam."""
    nside: int
    lmax: int
    bl: np.ndarray       # beam transfer function, lmax+1
    mask: np.ndarray     # apodised Galactic mask, RING
    sigma: np.ndarray    # per-pixel noise std [µK], RING


def ecliptic_hit_shape(nside: int) -> np.ndarray:
    """Synthetic Planck-like scan pattern: hits ∝ 1/cos β (great circles through the
    ecliptic poles) plus deep fields at the poles. Returned as a noise-std shape
    normalised to <σ²> = 1."""
    npix = hp.nside2npix(nside)
    vec_ecl = hp.Rotator(coord=["G", "E"])(np.array(hp.pix2vec(nside, np.arange(npix))))
    beta = np.arcsin(np.clip(vec_ecl[2], -1, 1))
    hits = 1.0 / np.maximum(np.cos(beta), 0.05)
    hits += 8.0 * np.exp(-0.5 * ((np.pi / 2 - np.abs(beta)) / np.radians(8)) ** 2)
    shape = 1.0 / np.sqrt(hits)
    return shape / np.sqrt(np.mean(shape ** 2))


def build_view_model(cfg: SkyConfig, mask_path: str, fwhm_arcmin: float = 160.0,
                     noise_ukarcmin: float = 40.0, apod_deg: float = 2.0,
                     noise_std_map: np.ndarray | None = None) -> ViewModel:
    """Mask recipe as in generate_sims_parallel.py (binarise at 0.9, apodise), and
    anisotropic white noise normalised to `noise_ukarcmin` on average — or taken
    from `noise_std_map` (µK per pixel at cfg.nside) when given."""
    hard = new_resolution_mask(mask_path, target_nside=cfg.nside, threshold=0.9)
    mask = np.clip(hp.smoothing(hard, fwhm=np.radians(apod_deg)), 0.0, 1.0)
    if noise_std_map is None:
        pix_arcmin = np.degrees(hp.nside2resol(cfg.nside)) * 60.0
        sigma = noise_ukarcmin / pix_arcmin * ecliptic_hit_shape(cfg.nside)
    else:
        sigma = np.asarray(noise_std_map, dtype=float)
    bl = hp.gauss_beam(np.radians(fwhm_arcmin / 60.0), lmax=cfg.lmax)
    return ViewModel(cfg.nside, cfg.lmax, bl, mask, sigma)


def rotate_alm(alm: np.ndarray, R: Rotation, lmax: int, nthreads: int = 1) -> np.ndarray:
    """Exact harmonic-space rotation; matches hp.rotate_alm(alm, matrix=R.as_matrix()).
    The rotated map satisfies m'(n̂) = m(Rᵀ n̂)."""
    psi, theta, phi = R.as_euler("zyz")
    return ducc0.sht.rotate_alm(alm, lmax, psi, theta, phi, nthreads=nthreads)


def make_view_T(alm: np.ndarray, vm: ViewModel, rng: np.random.Generator,
                rotate: bool = True, noise: bool = True, mask: bool = True,
                nthreads: int = 1):
    """One augmented view: rotate -> beam -> alm2map -> + noise -> × mask.
    Returns (T map in RING ordering [µK], rotation matrix)."""
    alm = np.asarray(alm, dtype=np.complex128)
    R = Rotation.random(rng=rng) if rotate else Rotation.identity()
    if rotate:
        alm = rotate_alm(alm, R, vm.lmax, nthreads)
    T = hp.alm2map(hp.almxfl(alm, vm.bl), vm.nside)
    if noise:
        T = T + vm.sigma * rng.standard_normal(T.size)
    if mask:
        T = T * vm.mask
    return T, R.as_matrix()

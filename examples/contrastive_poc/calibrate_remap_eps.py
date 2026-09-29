#!/usr/bin/env python3
"""Calibrate the amplitude prior of the remap hard negatives,
T(n̂) -> T(n̂ + ε∇ψ) on the primordial T, then lensed.

ψ: random gradient-remap potential (lensed_sky.random_remap_field: 2 ≤ L ≤ L_max,
L_max uniform in 4..48, power-law spectrum with random slope, rms deflection 1°), so
ε is the rms deflection in degrees. Remaps are computed *exactly* (one lenspyx
deflection ∇(φ + εψ)): the first-order expansion is already ~20% off at ε = 0.2°.

Targeted estimator (knows ψ): a remap stretches the field locally, so the local
gradient variance responds to ∇²ψ. Before the beam |∇T'|² ≈ |∇T|² (1 + ε∇²ψ), but
the compression moves power to higher ℓ, which the 160′ beam suppresses, so the
*observed* response has the opposite sign and ~0.3× the size. The test is therefore
two-sided (|z|) on the matched filter
    S = < ∇²ψ(n̂) |∇T_obs(n̂)|² >  over well-observed pixels (mask > 0.99),
on views mimicking training (skies rescaled to the fiducial C_ℓ, 160′ beam,
noise, Galactic mask at a random orientation). S is standardised per ψ with
un-remapped skies; the 5% threshold comes from separate null draws. Output: the ε
at which the detection power crosses 50% and 90%.

    python examples/contrastive_poc/calibrate_remap_eps.py [--workers 32]
Writes remap_eps_calibration.json (+ .png) into the PoC data directory.
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import healpy as hp
from scipy.spatial.transform import Rotation
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from skysimulation import lensed_sky as ls

DATA_DIR  = Path("/nvme/h/lchristodoulou/data_p318/simulations/contrastive_poc_nside64")
MASK_PATH = "/nvme/h/lchristodoulou/cmb/SkySimulation/data/Planck/COM_Mask_CMB-common-Mask-Int_2048_R3.00.fits"
EPS_GRID  = np.geomspace(0.02, 1.0, 12)      # rms deflection [deg]
N_ALT     = 200     # (sky, ψ) pairs
N_STD     = 40      # un-remapped skies per ψ to standardise S
N_THR     = 10      # further un-remapped skies per ψ for the pooled 5% threshold
SEED      = 525252

_G = {}


def _init():
    with open(DATA_DIR / "config.json") as fh:
        cfg = ls.SkyConfig(**json.load(fh)["sky_config"])
    tr = np.load(DATA_DIR / "lcdm_train.npz")
    vm = ls.build_view_model(cfg, MASK_PATH)
    _G.update(cfg=cfg, alm=tr["alm"], seeds=tr["seed"], vm=vm,
              spec=dict(np.load(DATA_DIR / "spectra.npz")),
              vec=np.array(hp.pix2vec(cfg.nside, np.arange(hp.nside2npix(cfg.nside)))))


def _stat(alm, lap_psi, rng):
    """Matched filter on the local gradient variance, for one view of a sky."""
    cfg, vm, vec = _G["cfg"], _G["vm"], _G["vec"]
    alm = ls.rescale_to_cl(np.asarray(alm, dtype=np.complex128), _G["spec"]["len_TT"])
    pix = hp.vec2pix(cfg.nside, *(Rotation.random(rng=rng).as_matrix() @ vec))  # mask/noise pulled back to the sky frame
    mask, sigma = vm.mask[pix], vm.sigma[pix]
    T = (hp.alm2map(hp.almxfl(alm, vm.bl), cfg.nside) + sigma * rng.standard_normal(vec.shape[1])) * mask
    _, gth, gph = hp.alm2map_der1(hp.map2alm(T, lmax=cfg.lmax, iter=1), cfg.nside)
    good = mask > 0.99
    return np.mean(lap_psi[good] * (gth[good] ** 2 + gph[good] ** 2))


def _one(n):
    cfg, alm = _G["cfg"], _G["alm"]
    rng = np.random.default_rng([SEED, n])
    i = int(rng.integers(len(alm)))
    psi, L_max, slope = ls.random_remap_field(rng)
    ell = np.arange(L_max + 1)
    lap_psi = hp.alm2map(hp.almxfl(psi, -ell * (ell + 1.0)), cfg.nside, lmax=L_max)
    others = rng.choice(np.delete(np.arange(len(alm)), i), size=N_STD + N_THR, replace=False)
    s0 = np.array([_stat(alm[j], lap_psi, rng) for j in others])
    mu, sd = s0[:N_STD].mean(), s0[:N_STD].std(ddof=1)
    tlm, plm = ls.unlensed_T_phi(cfg, _G["spec"], _G["seeds"][i])
    z_alt = [(_stat(ls.lens_T_with_remap(cfg, tlm, plm, psi, eps), lap_psi, rng) - mu) / sd for eps in EPS_GRID]
    return list((s0[N_STD:] - mu) / sd), z_alt, L_max, slope


def eps_at(power, target):
    p = np.maximum.accumulate(power)
    if p[-1] < target:
        return float("nan")
    return float(np.exp(np.interp(target, p, np.log(EPS_GRID))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=32)
    args = ap.parse_args()
    with ProcessPoolExecutor(args.workers, initializer=_init) as ex:
        res = list(ex.map(_one, range(N_ALT)))
    z_null = np.concatenate([r[0] for r in res])
    z_alt = np.array([r[1] for r in res])
    thr = np.quantile(np.abs(z_null), 0.95)          # two-sided
    power = (np.abs(z_alt) > thr).mean(axis=0)
    print("mean z of remapped skies per ε:", np.round(z_alt.mean(axis=0), 2))
    out = {"eps_grid_deg": EPS_GRID.tolist(), "power": power.tolist(), "z_threshold": float(thr),
           "eps50": eps_at(power, 0.5), "eps90": eps_at(power, 0.9),
           "n_alt": N_ALT, "n_std": N_STD, "n_thr": N_THR, "fpr": 0.05,
           "field": {"lmin": 2, "lmax_range": [4, 48], "slope_range": [0.0, 4.0], "normalisation": "rms deflection 1 deg"}}
    print(f"eps50 = {out['eps50']:.3f} deg   eps90 = {out['eps90']:.3f} deg")
    print("power:", dict(zip(np.round(EPS_GRID, 3), np.round(power, 3))))
    with open(DATA_DIR / "remap_eps_calibration.json", "w") as fh:
        json.dump(out, fh, indent=2)
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.semilogx(EPS_GRID, power, "o-")
    for y in (0.5, 0.9):
        ax.axhline(y, color="k", lw=0.6, ls=":")
    ax.set(xlabel="rms deflection ε [deg]", ylabel="targeted power (5% FPR)",
           title="exact remap T(n̂ + ε∇ψ) on the primordial T, lensed")
    plt.tight_layout()
    plt.savefig(DATA_DIR / "remap_eps_calibration.png", dpi=130)
    print("wrote", DATA_DIR / "remap_eps_calibration.json")


if __name__ == "__main__":
    main()

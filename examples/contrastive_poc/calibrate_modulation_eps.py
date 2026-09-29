#!/usr/bin/env python3
"""Calibrate the amplitude prior of the modulation hard negatives.

A hard negative is the training sky with its *primordial* T modulated,
T -> T (1 + ε M), then lensed; by linearity of lensing it is  alm + ε·delta  with
delta = lens(T·M) stored in train_modulation.npz. Both the sky and its negative are
rescaled to the fiducial lensed C_ℓ, exactly as in training.

For each L_dev, find the ε at which a *targeted* estimator that knows M reaches 50%
and 90% detection power at a 5% false-alarm rate. Training draws ε log-uniformly
between those two values, so the network sees both hard and easy negatives.

Targeted estimator (matched filter on the local variance):
    S = < M(n̂) [T_obs(n̂)² − E₀ T_obs(n̂)²] >  over well-observed pixels (mask > 0.99),
which responds to T'² ≈ T² (1 + 2εM). Observations mimic a training view: 160′
beam, anisotropic noise and the Galactic mask at a random orientation relative to
the sky. For each M, S is standardised with unmodulated skies seen through the same
M; the 5% threshold comes from separate null draws.

    python examples/contrastive_poc/calibrate_modulation_eps.py [--workers 9]
Writes modulation_eps_calibration.json (+ .png) into the PoC data directory.
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
EPS_GRID  = np.geomspace(0.003, 0.5, 14)
N_ALT     = 200     # modulated (sky, M) pairs per L_dev
N_STD     = 40      # unmodulated skies per M to standardise S
N_THR     = 10      # further unmodulated skies per M for the pooled 5% threshold
SEED      = 424242

_G = {}


def _init():
    with open(DATA_DIR / "config.json") as fh:
        cfg = ls.SkyConfig(**json.load(fh)["sky_config"])
    mod = np.load(DATA_DIR / "train_modulation.npz")
    cl_fid = np.load(DATA_DIR / "spectra.npz")["len_TT"]
    vm = ls.build_view_model(cfg, MASK_PATH)
    ell = np.arange(cfg.lmax + 1)
    _G.update(cfg=cfg, alm=np.load(DATA_DIR / "lcdm_train.npz")["alm"], delta=mod["delta"],
              M_alm=mod["M_alm"], L_dev=mod["L_dev"], cl_fid=cl_fid, vm=vm,
              var_sig=np.sum((2 * ell + 1) * cl_fid[: cfg.lmax + 1] * vm.bl ** 2) / (4 * np.pi),
              vec=np.array(hp.pix2vec(cfg.nside, np.arange(hp.nside2npix(cfg.nside)))))


def _stat(alm, M, rng):
    """Matched-filter statistic for one view of a (fiducial-rescaled) sky alm."""
    cfg, vm, vec = _G["cfg"], _G["vm"], _G["vec"]
    alm = ls.rescale_to_cl(np.asarray(alm, dtype=np.complex128), _G["cl_fid"])
    pix = hp.vec2pix(cfg.nside, *(Rotation.random(rng=rng).as_matrix() @ vec))  # mask/noise pulled back to the sky frame
    mask, sigma = vm.mask[pix], vm.sigma[pix]
    T = (hp.alm2map(hp.almxfl(alm, vm.bl), cfg.nside) + sigma * rng.standard_normal(vec.shape[1])) * mask
    good = mask > 0.99
    return np.mean(M[good] * (T[good] ** 2 - _G["var_sig"] - sigma[good] ** 2))


def _calibrate(L_dev):
    rng = np.random.default_rng([SEED, L_dev])
    cfg, alm, delta = _G["cfg"], _G["alm"], _G["delta"]
    ii, kk = np.nonzero(_G["L_dev"] == L_dev)
    pick = rng.choice(len(ii), size=min(N_ALT, len(ii)), replace=False)
    z_alt = np.zeros((len(pick), len(EPS_GRID)))
    z_null = []
    for n, p in enumerate(pick):
        i, k = ii[p], kk[p]
        M = hp.alm2map(_G["M_alm"][i, k].astype(np.complex128), cfg.nside)
        others = rng.choice(np.delete(np.arange(len(alm)), i), size=N_STD + N_THR, replace=False)
        s0 = np.array([_stat(alm[j], M, rng) for j in others])
        mu, sd = s0[:N_STD].mean(), s0[:N_STD].std(ddof=1)
        z_null.extend((s0[N_STD:] - mu) / sd)
        base, d = alm[i].astype(np.complex128), delta[i, k].astype(np.complex128)
        z_alt[n] = [(_stat(base + eps * d, M, rng) - mu) / sd for eps in EPS_GRID]
    thr = np.quantile(z_null, 0.95)
    return L_dev, (z_alt > thr).mean(axis=0), float(thr)


def eps_at(power, target):
    """Log-interpolated ε where the (monotone-ised) power curve crosses target."""
    p = np.maximum.accumulate(power)
    if p[-1] < target:
        return float("nan")
    return float(np.exp(np.interp(target, p, np.log(EPS_GRID))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=9)
    args = ap.parse_args()

    _init()
    L_devs = sorted(set(_G["L_dev"].ravel().tolist()))
    with ProcessPoolExecutor(args.workers, initializer=_init) as ex:
        results = sorted(ex.map(_calibrate, L_devs))

    table = {}
    fig, ax = plt.subplots(figsize=(7, 4))
    for L_dev, power, thr in results:
        table[int(L_dev)] = {"eps50": eps_at(power, 0.5), "eps90": eps_at(power, 0.9),
                             "power": power.tolist(), "z_threshold": thr}
        ax.semilogx(EPS_GRID, power, "o-", ms=3, label=f"L_dev={L_dev}")
        print(f"L_dev={L_dev:2d}  eps50={table[L_dev]['eps50']:.4f}  eps90={table[L_dev]['eps90']:.4f}",
              flush=True)
    ax.axhline(0.5, color="k", lw=0.6, ls=":")
    ax.axhline(0.9, color="k", lw=0.6, ls=":")
    ax.set(xlabel="ε", ylabel="targeted detection power (5% FPR)",
           title="primordial modulation T(1+εM), 2 ≤ L ≤ L_dev, after lensing")
    ax.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    plt.savefig(DATA_DIR / "modulation_eps_calibration.png", dpi=130)

    with open(DATA_DIR / "modulation_eps_calibration.json", "w") as fh:
        json.dump({"eps_grid": EPS_GRID.tolist(), "n_alt": N_ALT, "n_std": N_STD, "n_thr": N_THR,
                   "fpr": 0.05, "table": table}, fh, indent=2)
    print("wrote", DATA_DIR / "modulation_eps_calibration.json")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Exact remap hard negatives for contrastive training (second hard-negative family).

    python examples/contrastive_poc/generate_remap_negatives.py [--workers 32]

For each training sky, K_REMAP random gradient-remap potentials ψ
(lensed_sky.random_remap_field) and, for each, the sky remapped *exactly* at
N_LEVELS amplitudes: T(n̂) -> T(n̂ + ε∇ψ) on the primordial T, then lensed (one
lenspyx deflection ∇(φ + εψ)). The first-order expansion is too inaccurate above
ε ~ 0.1°, so amplitudes are stored as a discrete ladder instead of a linear term;
training picks the level matching its (adaptive) ε.

Each field has its own log-spaced ladder from min(LADDER_LO × ε50, ε_max/4) up to its
ε_max = min(ε_hi, FOLD_LIMIT / max|∇²ψ|). The folding guard keeps every negative a
genuine diffeomorphism: the local convergence ε|∇²ψ| stays ≤ FOLD_LIMIT, so the remap
never folds. Smooth fields reach ε_hi, small-scale fields stop at smaller deflections.
ε_hi is the calibrated ε90, or the largest calibrated ε when 90% power is not reached.
LADDER_LO < 1 leaves room for the adaptive ε to shrink below the calibrated prior.

Writes into the PoC data directory (existing outputs are never touched):
  train_remap_alm.npy   complex64 [N_TRAIN, K_REMAP, N_LEVELS, nalm]  (memory-mappable)
  train_remap_meta.npz  eps_levels [N_TRAIN, K_REMAP, N_LEVELS] (deg), L_max and slope
                        per (sky, field), the prior range (eps50, eps_hi), seeds
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import healpy as hp
from tqdm import tqdm

from skysimulation import lensed_sky as ls

DATA_DIR = Path("/nvme/h/lchristodoulou/data_p318/simulations/contrastive_poc_nside64")
K_REMAP, N_LEVELS, LADDER_LO, FOLD_LIMIT = 2, 8, 0.15, 0.5
SEED_REMAP = 990_000_000

_G = {}


def _init(eps_lo, eps_hi):
    with open(DATA_DIR / "config.json") as fh:
        cfg = ls.SkyConfig(**json.load(fh)["sky_config"])
    _G.update(cfg=cfg, spec=dict(np.load(DATA_DIR / "spectra.npz")), eps_lo=eps_lo, eps_hi=eps_hi)


def ladder(psi, L_max):
    """This field's ε ladder [deg], capped by the folding guard."""
    ell = np.arange(L_max + 1)
    lap_max = np.abs(hp.alm2map(hp.almxfl(psi, -ell * (ell + 1.0)), 128, lmax=L_max)).max()
    eps_max = min(_G["eps_hi"], FOLD_LIMIT / lap_max)
    return np.geomspace(min(_G["eps_lo"], eps_max / 4), eps_max, N_LEVELS)


def _one(task):
    i, seed = task
    cfg = _G["cfg"]
    tlm, plm = ls.unlensed_T_phi(cfg, _G["spec"], seed)
    out = np.zeros((K_REMAP, N_LEVELS, (cfg.lmax + 1) * (cfg.lmax + 2) // 2), dtype=np.complex64)
    lmaxes, slopes, levels = [], [], []
    for k in range(K_REMAP):
        rng = np.random.default_rng([SEED_REMAP, i, k])
        psi, L_max, slope = ls.random_remap_field(rng)
        lev = ladder(psi, L_max)
        for j, eps in enumerate(lev):
            out[k, j] = ls.lens_T_with_remap(cfg, tlm, plm, psi, eps)
        lmaxes.append(L_max)
        slopes.append(slope)
        levels.append(lev)
    return i, out, lmaxes, slopes, levels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=32)
    args = ap.parse_args()
    alm_path, meta_path = DATA_DIR / "train_remap_alm.npy", DATA_DIR / "train_remap_meta.npz"
    if alm_path.exists() and meta_path.exists():
        print("remap negatives exist; nothing to do")
        return

    with open(DATA_DIR / "remap_eps_calibration.json") as fh:
        cal = json.load(fh)
    eps50 = cal["eps50"]
    eps_hi = cal["eps90"] if np.isfinite(cal["eps90"] or np.nan) else max(cal["eps_grid_deg"])
    eps_lo = LADDER_LO * eps50
    print(f"prior [ε50, ε_hi] = [{eps50:.3f}, {eps_hi:.3f}] deg; ladders from ≤ {eps_lo:.3f} deg")

    seeds = np.load(DATA_DIR / "lcdm_train.npz")["seed"]
    with open(DATA_DIR / "config.json") as fh:
        lmax = json.load(fh)["lmax"]
    nalm = (lmax + 1) * (lmax + 2) // 2
    part = alm_path.with_suffix(".part.npy")
    alm = np.lib.format.open_memmap(part, mode="w+", dtype=np.complex64,
                                    shape=(len(seeds), K_REMAP, N_LEVELS, nalm))
    L_max = np.zeros((len(seeds), K_REMAP), dtype=int)
    slope = np.zeros((len(seeds), K_REMAP))
    eps_levels = np.zeros((len(seeds), K_REMAP, N_LEVELS))
    with ProcessPoolExecutor(args.workers, initializer=_init, initargs=(eps_lo, eps_hi)) as ex:
        for i, out, lm, sl, lev in tqdm(ex.map(_one, enumerate(seeds), chunksize=2), total=len(seeds),
                                        desc="remap negatives"):
            alm[i], L_max[i], slope[i], eps_levels[i] = out, lm, sl, lev
    alm.flush()
    del alm
    part.rename(alm_path)
    np.savez(meta_path, eps_levels=eps_levels, L_max=L_max, slope=slope, seeds=seeds,
             eps50=eps50, eps_hi=eps_hi, fold_limit=FOLD_LIMIT, k_remap=K_REMAP, seed_remap=SEED_REMAP)
    print("fraction of fields capped below ε_hi by the folding guard:",
          round(float(np.mean(eps_levels[..., -1] < eps_hi - 1e-9)), 3))
    print("wrote", alm_path, meta_path)


if __name__ == "__main__":
    main()

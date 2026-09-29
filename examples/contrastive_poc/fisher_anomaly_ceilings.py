#!/usr/bin/env python3
"""Fisher-optimal AUC ceilings for the PoC anomaly sets.

    python examples/contrastive_poc/fisher_anomaly_ceilings.py [--out fisher_auc.csv]

For each parametric anomaly, the separation an ideal targeted test would reach,
AUC against the ΛCDM null, using the same fiducial spectra, 160′ beam, noise level
and (effective) sky fraction as the PoC views. Gaussian-likelihood Fisher
information at zero amplitude:

  low_power      C_ℓ → (1 − s) C_ℓ, 2 ≤ ℓ ≤ 30:   F_s = f_sky Σ (2ℓ+1)/2 · R_ℓ²,
                 R_ℓ = S_ℓ / (S_ℓ + N_ℓ) (signal fraction of the observed power).
                 AUC = Φ(s / (√2 σ_s)).
  parity         odd ℓ ×(1 + p), even ×(1 − p), 2 ≤ ℓ ≤ 30: same F, AUC = Φ(p / (√2 σ_p)).
  hemispherical  T → T (1 + A n̂·d̂) on ℓ ≤ 64, direction unknown. Each dipole
                 component: F = f_sky Σ_ℓ (ℓ+1)/3 · δ_ℓ² / (O_ℓ O_ℓ+1), with
                 δ_ℓ = b_ℓ b_ℓ+1 (C_ℓ[ℓ≤64] + C_ℓ+1[ℓ+1≤64]) and O = observed power.
                 The optimal direction-blind test is |Â|²/σ² ~ χ²_3 (null) vs
                 non-central χ²_3(λ = A²/σ²); AUC by Monte Carlo.
  cold_spot      known 5° Gaussian profile at a known, unmasked location (upper
                 bound: a blind search pays a look-elsewhere cost over positions):
                 SNR² per unit depth = Σ_{ℓ≥2} (2ℓ+1) s_ℓ b_ℓ² / O_ℓ (s_ℓ: profile power),
                 AUC = Φ(A·SNR / √2).

Alignment changes only phases at fixed power and has no clean Fisher form here.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import healpy as hp
import pandas as pd
from scipy.stats import norm

from skysimulation import lensed_sky as ls

POC_DIR   = Path("/nvme/h/lchristodoulou/data_p318/simulations/contrastive_poc_nside64")
MASK_PATH = "/nvme/h/lchristodoulou/cmb/SkySimulation/data/Planck/COM_Mask_CMB-common-Mask-Int_2048_R3.00.fits"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(POC_DIR / "fisher_auc.csv"))
    args = ap.parse_args()

    with open(POC_DIR / "config.json") as fh:
        cfg_json = json.load(fh)
    cfg = ls.SkyConfig(**cfg_json["sky_config"])
    amps = cfg_json["anomalies"]
    vm = ls.build_view_model(cfg, MASK_PATH)
    lmax = cfg.lmax
    ell = np.arange(lmax + 2)
    C = np.load(POC_DIR / "spectra.npz")["len_TT"][: lmax + 2]
    b = hp.gauss_beam(np.radians(160 / 60), lmax=lmax + 1)
    N = np.mean(vm.sigma ** 2) * 4 * np.pi / hp.nside2npix(cfg.nside)   # white-noise power (µK²)
    S = C * b ** 2
    O = S + N
    fsky = np.mean(vm.mask ** 2) ** 2 / np.mean(vm.mask ** 4)          # effective sky fraction
    rng = np.random.default_rng(0)
    rows = []

    # low power and parity: amplitude of C_ℓ at 2 ≤ ℓ ≤ 30
    L = np.arange(2, 31)
    F = fsky * np.sum((2 * L + 1) / 2 * (S[L] / O[L]) ** 2)
    sig = 1 / np.sqrt(F)
    for name in ["low_power", "parity"]:
        for a in amps[name]:
            rows.append({"anomaly": name, "amplitude": a, "sigma_1": sig,
                         "fisher_auc": norm.cdf(a / (np.sqrt(2) * sig))})

    # hemispherical: dipole modulation of ℓ ≤ 64, unknown direction
    lm = 64
    Lh = np.arange(2, lm + 1)
    delta = b[Lh] * b[Lh + 1] * (C[Lh] * (Lh <= lm) + C[Lh + 1] * (Lh + 1 <= lm))
    F = fsky * np.sum((Lh + 1) / 3 * delta ** 2 / (O[Lh] * O[Lh + 1]))
    sig = 1 / np.sqrt(F)
    z0 = rng.standard_normal((400_000, 3))
    null = np.sum(z0 ** 2, axis=1)
    for a in amps["hemispherical"]:
        alt = np.sum((rng.standard_normal((400_000, 3)) + [a / sig, 0, 0]) ** 2, axis=1)
        auc = (np.searchsorted(np.sort(null), alt).mean()) / len(null)
        rows.append({"anomaly": "hemispherical", "amplitude": a, "sigma_1": sig, "fisher_auc": auc})

    # cold spot: known profile, known unmasked location (upper bound)
    ns = 256
    vec = np.array(hp.pix2vec(ns, np.arange(hp.nside2npix(ns))))
    theta = np.arccos(np.clip(vec[2], -1, 1))
    prof = np.exp(-0.5 * (theta / np.radians(5.0)) ** 2)                 # unit depth
    s_l = hp.anafast(prof, lmax=lmax)
    Lc = np.arange(2, lmax + 1)  # monopole and dipole are not CMB observables
    snr1 = np.sqrt(np.sum((2 * Lc + 1) * s_l[Lc] * b[Lc] ** 2 / O[Lc]))
    for a in amps["cold_spot"]:
        rows.append({"anomaly": "cold_spot", "amplitude": a, "sigma_1": 1 / snr1,
                     "fisher_auc": norm.cdf(a * snr1 / np.sqrt(2))})

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print(f"f_sky,eff = {fsky:.3f}   noise power N = {N:.2e} µK²")
    print(df.round(4).to_string(index=False))


if __name__ == "__main__":
    main()

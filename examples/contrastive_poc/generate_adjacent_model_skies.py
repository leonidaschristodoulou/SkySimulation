#!/usr/bin/env python3
"""ΛCDM-adjacent cosmologies (w0wa, early dark energy) through the PoC pipeline,
to check that the anomaly statistics do not flag viable nearby models: these are
statistically isotropic Gaussian skies whose C_ℓ differs slightly from the fiducial.

    python examples/contrastive_poc/generate_adjacent_model_skies.py [--workers 32]

  w0wa__0  N_W0WA skies, (w0, wa, H0, ω_b, ω_c, n_s, A_s, τ) drawn (weighted) from the
           DESI BAO + Planck 2018 posterior chain: data-consistent by construction.
  ede__0   N_EDE_SETS early-dark-energy parameter sets × SEEDS_PER_SET skies.
           f_EDE ∈ [1e-4, 0.08], log10 z_c ∈ [3.3, 3.8], θ_i ∈ [0.5, 3.0], w_n = 0.5
           (AxionEffectiveFluid, as in pipeline.py). θ* is held at the fiducial value
           (H0 solved) and (ω_c, n_s, A_s) are refit to the fiducial lensed TT over
           2 ≤ ℓ ≤ 2000 with a Planck-like χ² (cosmic variance + noise + beam), so
           the model stays data-consistent the way real EDE fits do.

Writes anomalies_adjacent.npz / anomalies_adjacent_meta.csv (the anomaly-set format,
so contrastive_eval.py can score them — these are *not* anomalies) and
adjacent_model_params.csv (parameters, fit quality, C_ℓ change at ℓ ≤ 191).
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from getdist import loadMCSamples
from scipy.optimize import least_squares
from tqdm import tqdm

from skysimulation import lensed_sky as ls

DATA_DIR = Path("/nvme/h/lchristodoulou/data_p318/simulations/contrastive_poc_nside64")
W0WA_CHAIN = ("/nvme/h/lchristodoulou/data_p318/base_w_wa-desi-bao-all_planck2018-lowl-TT-clik_"
              "planck2018-lowl-EE-clik_planck-NPIPE-highl-CamSpec-TTTEEE/chain")
CFG = ls.SkyConfig(nside=64)
N_W0WA, N_EDE_SETS, SEEDS_PER_SET = 1000, 100, 10
SEED_W0WA, SEED_EDE = 1_010_000_000, 1_020_000_000
EDE_W_N = 0.5
FIT_LMAX, FSKY_FIT, NOISE_UKARCMIN, BEAM_ARCMIN = 2000, 0.7, 30.0, 7.0   # Planck-like TT
_G = {}


def _init(fid_len, thetastar):
    _G.update(fid=fid_len, thetastar=thetastar)


def _chi_residuals(spec_len, fid):
    ell = np.arange(2, FIT_LMAX + 1)
    b2 = np.exp(-ell * (ell + 1) * (np.radians(BEAM_ARCMIN / 60) / np.sqrt(8 * np.log(2))) ** 2)
    N = (np.radians(NOISE_UKARCMIN / 60)) ** 2 / b2
    sigma = np.sqrt(2 / ((2 * ell + 1) * FSKY_FIT)) * (fid[ell] + N)
    return (spec_len[ell] - fid[ell]) / sigma


def _ede_spectrum(theta_vec, ede, fit_cfg):
    omch2, ns, lnAs = theta_vec
    params = dict(ls.FIDUCIAL, omch2=omch2, ns=ns, As=np.exp(lnAs) * 1e-10)
    return ls.get_spectra_T_extended(fit_cfg, params, ede=ede, thetastar=_G["thetastar"])


def _fit_ede(task):
    """Refit (ω_c, n_s, ln A_s) for one EDE parameter set, θ* fixed."""
    k, f_ede, logzc, theta_i = task
    ede = (EDE_W_N, f_ede, 10 ** logzc, theta_i)
    fit_cfg = ls.SkyConfig(nside=CFG.nside, lmax_unl=FIT_LMAX, nside_lens=CFG.nside_lens)
    x0 = [ls.FIDUCIAL["omch2"], ls.FIDUCIAL["ns"], np.log(ls.FIDUCIAL["As"] * 1e10)]
    fid = _G["fid"]
    r = least_squares(lambda x: _chi_residuals(_ede_spectrum(x, ede, fit_cfg)["len_TT"], fid), x0,
                      x_scale=[0.002, 0.005, 0.01], diff_step=[1e-3, 1e-3, 1e-3], max_nfev=40)
    omch2, ns, lnAs = r.x
    chi2_nofit = float(np.sum(_chi_residuals(_ede_spectrum(x0, ede, fit_cfg)["len_TT"], fid) ** 2))
    return {"set": k, "f_ede": f_ede, "logzc": logzc, "theta_i": theta_i, "omch2": omch2, "ns": ns,
            "As": np.exp(lnAs) * 1e-10, "chi2_fit": float(np.sum(r.fun ** 2)), "chi2_nofit": chi2_nofit}


def _sky(task):
    """One lensed sky for a given model: returns (alm, H0, ΔC/C rms at ℓ ≤ 30 and 31–191)."""
    seed, params, w0, wa, ede = task
    spec = ls.get_spectra_T_extended(CFG, params, w0=w0, wa=wa, ede=ede,
                                     thetastar=_G["thetastar"] if ede is not None else None)
    r = spec["len_TT"][2:CFG.lmax + 1] / _G["fid"][2:CFG.lmax + 1] - 1
    alm = ls.lensed_T_alm(CFG, spec, seed).astype(np.complex64)
    return alm, spec["H0"], float(np.sqrt(np.mean(r[:29] ** 2))), float(np.sqrt(np.mean(r[29:] ** 2)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=32)
    args = ap.parse_args()
    fid_cfg = ls.SkyConfig(nside=CFG.nside, lmax_unl=FIT_LMAX, nside_lens=CFG.nside_lens)
    fid_spec = ls.get_spectra_T_extended(fid_cfg, ls.FIDUCIAL)
    fid_len, thetastar = fid_spec["len_TT"], fid_spec["thetastar"]
    print(f"fiducial θ* = {thetastar:.6e}")

    # w0wa: weighted posterior draws
    chains = loadMCSamples(W0WA_CHAIN, settings={"ignore_rows": 0.3})
    p = chains.getParams()
    w = chains.weights / chains.weights.sum()
    idx = np.random.default_rng(SEED_W0WA).choice(len(chains.samples), size=N_W0WA, replace=False, p=w)
    w0wa_tasks = [(SEED_W0WA + n,
                   dict(ls.FIDUCIAL, H0=float(p.H0[k]), ombh2=float(p.ombh2[k]), omch2=float(p.omch2[k]),
                        ns=float(p.ns[k]), tau=max(float(p.tau[k]), 0.01), As=float(np.exp(p.logA[k]) * 1e-10)),
                   float(p.w[k]), float(p.wa[k]), None) for n, k in enumerate(idx)]

    # EDE: parameter sets, refit, then skies
    rng = np.random.default_rng(SEED_EDE)
    ede_sets = [(k, float(rng.uniform(1e-4, 0.08)), float(rng.uniform(3.3, 3.8)), float(rng.uniform(0.5, 3.0)))
                for k in range(N_EDE_SETS)]
    with ProcessPoolExecutor(args.workers, initializer=_init, initargs=(fid_len, thetastar)) as ex:
        fits = list(tqdm(ex.map(_fit_ede, ede_sets), total=N_EDE_SETS, desc="EDE refits"))
        ede_tasks, ede_rows = [], []
        for fit in fits:
            ede = (EDE_W_N, fit["f_ede"], 10 ** fit["logzc"], fit["theta_i"])
            params = dict(ls.FIDUCIAL, omch2=fit["omch2"], ns=fit["ns"], As=fit["As"])
            for s in range(SEEDS_PER_SET):
                ede_tasks.append((SEED_EDE + 1000 * fit["set"] + s, params, None, None, ede))
                ede_rows.append({"model": "ede", **fit})
        w0wa_res = list(tqdm(ex.map(_sky, w0wa_tasks, chunksize=4), total=len(w0wa_tasks), desc="w0wa skies"))
        ede_res = list(tqdm(ex.map(_sky, ede_tasks, chunksize=4), total=len(ede_tasks), desc="EDE skies"))

    np.savez(DATA_DIR / "anomalies_adjacent.npz",
             **{"w0wa__0": np.stack([r[0] for r in w0wa_res]), "ede__0": np.stack([r[0] for r in ede_res])})
    rows = [{"key": k, "anomaly": k.split("__")[0], "amplitude": 0.0, "null_index": 20_000 + i}
            for k, n in [("w0wa__0", len(w0wa_res)), ("ede__0", len(ede_res))] for i in range(n)]
    pd.DataFrame(rows).to_csv(DATA_DIR / "anomalies_adjacent_meta.csv", index=False)

    par = [{"model": "w0wa", "seed": t[0], **{k: t[1][k] for k in ["H0", "ombh2", "omch2", "ns", "As", "tau"]},
            "w0": t[2], "wa": t[3]} for t in w0wa_tasks] + \
          [{**row, "seed": t[0]} for row, t in zip(ede_rows, ede_tasks)]
    par = pd.DataFrame(par)
    par["H0_solved"] = [r[1] for r in w0wa_res + ede_res]
    par["dcl_rms_l2_30"] = [r[2] for r in w0wa_res + ede_res]
    par["dcl_rms_l31_191"] = [r[3] for r in w0wa_res + ede_res]
    par.to_csv(DATA_DIR / "adjacent_model_params.csv", index=False)
    print(par.groupby("model")[["H0_solved", "dcl_rms_l2_30", "dcl_rms_l31_191"]].describe().T.round(4))
    if len(fits):
        f = pd.DataFrame(fits)
        print("EDE refit χ² (ℓ ≤ 2000): before", f.chi2_nofit.median().round(1), " after", f.chi2_fit.median().round(1))
    print("wrote", DATA_DIR / "anomalies_adjacent.npz")


if __name__ == "__main__":
    main()

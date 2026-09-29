#!/usr/bin/env python3
"""Fresh ΛCDM skies to check the false-alarm calibration of the PoC scores.

    python examples/contrastive_poc/generate_fpr_check_skies.py [--workers 32]

Writes anomalies_fpr.npz / anomalies_fpr_meta.csv (in the anomaly-set format, so
contrastive_eval.py can score them) plus fpr_posterior_params.csv:

  lcdm_fixed__0      N new seeds at the fiducial cosmology (the one all PoC sims use).
                     Scored against the null threshold, ~5% should be flagged: a direct
                     test of the calibration.
  lcdm_posterior__0  N new seeds, each with parameters drawn from the Planck 2018
                     TT,TE,EE+lowE+lensing posterior chain (weighted, correlated). The
                     excess over 5% is how much parameter uncertainty inflates the
                     false-alarm rate of a fixed-cosmology null.

These are not anomalies; the file format is only reused for scoring.
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from getdist import loadMCSamples
from tqdm import tqdm

from skysimulation import lensed_sky as ls

OUT_DIR = Path("/nvme/h/lchristodoulou/data_p318/simulations/contrastive_poc_nside64")
CHAIN = ("/nvme/h/lchristodoulou/data_p318/COM_CosmoParams_base-plikHM-TTTEEE-lowl-lowE_R3/"
         "base/plikHM_TTTEEE_lowl_lowE_lensing/base_plikHM_TTTEEE_lowl_lowE_lensing")
CFG = ls.SkyConfig(nside=64)
N = 1000
SEED_FIXED, SEED_POST = 970_000_000, 980_000_000

_SPEC = None


def _init(spec):
    global _SPEC
    _SPEC = spec


def _fixed(seed):
    return ls.lensed_T_alm(CFG, _SPEC, seed).astype(np.complex64)


def _posterior(task):
    seed, params = task
    spec = ls.get_spectra_T(CFG, params)
    return ls.lensed_T_alm(CFG, spec, seed).astype(np.complex64)


def draw_posterior(n, seed):
    chains = loadMCSamples(CHAIN, settings={"ignore_rows": 0.3})
    p = chains.getParams()
    w = chains.weights / chains.weights.sum()
    idx = np.random.default_rng(seed).choice(len(chains.samples), size=n, replace=False, p=w)
    return [dict(ls.FIDUCIAL, H0=float(p.H0[k]), ombh2=float(p.omegabh2[k]), omch2=float(p.omegach2[k]),
                 ns=float(p.ns[k]), tau=max(float(p.tau[k]), 0.01), As=float(np.exp(p.logA[k]) * 1e-10))
            for k in idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=32)
    args = ap.parse_args()
    spec = dict(np.load(OUT_DIR / "spectra.npz"))

    seeds_fixed = SEED_FIXED + np.arange(N)
    seeds_post = SEED_POST + np.arange(N)
    params = draw_posterior(N, SEED_POST)
    pd.DataFrame(params).assign(seed=seeds_post).to_csv(OUT_DIR / "fpr_posterior_params.csv", index=False)

    with ProcessPoolExecutor(args.workers, initializer=_init, initargs=(spec,)) as ex:
        fixed = list(tqdm(ex.map(_fixed, seeds_fixed, chunksize=8), total=N, desc="fixed cosmology"))
        post = list(tqdm(ex.map(_posterior, zip(seeds_post, params), chunksize=2), total=N, desc="posterior"))

    np.savez(OUT_DIR / "anomalies_fpr.npz", **{"lcdm_fixed__0": np.stack(fixed), "lcdm_posterior__0": np.stack(post)})
    # null_index only sets the (seeded) view draws; offset past the null set so no views are shared
    rows = [{"key": k, "anomaly": k.split("__")[0], "amplitude": 0.0, "null_index": 10_000 + i}
            for k in ["lcdm_fixed__0", "lcdm_posterior__0"] for i in range(N)]
    pd.DataFrame(rows).to_csv(OUT_DIR / "anomalies_fpr_meta.csv", index=False)
    print("wrote", OUT_DIR / "anomalies_fpr.npz")


if __name__ == "__main__":
    main()

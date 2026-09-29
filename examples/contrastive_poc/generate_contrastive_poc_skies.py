#!/usr/bin/env python3
"""Stage 1 skies for the contrastive-learning anomaly proof of concept (T only, nside 64).

    python examples/contrastive_poc/generate_contrastive_poc_skies.py [--workers 32]

Writes to OUT_DIR (each output is skipped if it already exists):
  lcdm_train.npz      N_TRAIN ΛCDM lensed T alms (training: positives = two views of one sky)
  lcdm_null.npz       N_NULL held-out ΛCDM skies (null distribution for the anomaly scores)
  lcdm_ref.npz        N_REF held-out ΛCDM skies (reference cloud for the outlier scores)
  anomalies.npz       the first N_ANOM null skies with each anomaly injected at each
                      amplitude (paired with their un-injected twin in lcdm_null.npz)
  anomalies_meta.csv  per injected sky: anomaly, amplitude, null index, direction
  train_modulation.npz  hard negatives: for each training sky, K_NEG lensed modulation
                      terms lens(T·M), so that the modulated sky is  alm + ε·delta
  config.json

Order of operations (sky frame): draw (T, φ) -> anomaly / modulation on the
primordial (unlensed) T -> lens. All skies share the Planck 2018 best-fit
cosmology, so "normal" means ΛCDM + cosmic variance.

Hard negatives: M is a Gaussian random field with 2 ≤ L ≤ L_dev (L_dev uniform in
2..10; the dipole is held out so the hemispherical test stays blind), random
spectrum, unit variance. Lensing is linear in T for fixed φ, so
lens(T(1 + εM)) = lens(T) + ε lens(T·M): storing lens(T·M) lets training draw ε
freshly every epoch (prior: calibrate_modulation_eps.py).
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import healpy as hp
from tqdm import tqdm

from skysimulation import lensed_sky as ls

OUT_DIR = Path("/nvme/h/lchristodoulou/data_p318/simulations/contrastive_poc_nside64")
CFG     = ls.SkyConfig(nside=64)

N_TRAIN, N_NULL, N_ANOM, N_REF = 4000, 500, 300, 1500
SEED_TRAIN, SEED_NULL, SEED_ANOM, SEED_REF = 910_000_000, 920_000_000, 930_000_000, 940_000_000
K_NEG, L_DEV_RANGE, SEED_NEG = 4, (2, 10), 950_000_000

ANOMALIES = {  # name -> amplitudes (see the injector docstrings for units)
    "hemispherical": [0.02, 0.04, 0.07, 0.10],   # modulation amplitude A, ℓ ≤ 64
    "low_power":     [0.10, 0.20, 0.35, 0.50],   # fractional power deficit, 2 ≤ ℓ ≤ 30
    "parity":        [0.10, 0.20, 0.30, 0.50],   # odd ×(1+p), even ×(1−p), 2 ≤ ℓ ≤ 30
    "alignment":     [0.25, 0.50, 0.75, 1.00],   # planar ℓ=2,3 mixing fraction α
    "cold_spot":     [50.0, 100.0, 150.0, 200.0],  # central depth [µK], σ = 5°
}

_SPEC = None


def _init(spec):
    global _SPEC
    _SPEC = spec


def _lens(seed):
    return ls.lensed_T_alm(CFG, _SPEC, seed).astype(np.complex64)


def lens_many(seeds, spec, workers, desc):
    with ProcessPoolExecutor(workers, initializer=_init, initargs=(spec,)) as ex:
        out = list(tqdm(ex.map(_lens, seeds, chunksize=8), total=len(seeds), desc=desc))
    return np.stack(out)


def injector(name, amp, rng):
    """Returns (hook acting on the unlensed tlm, direction or None). The rng is
    consumed identically whatever the anomaly, so directions are reproducible."""
    d = ls.random_direction(rng)
    if name == "hemispherical":
        return (lambda t: ls.inject_hemispherical(t, amp, d)), d
    if name == "low_power":
        return (lambda t: ls.inject_low_power(t, amp)), None
    if name == "parity":
        return (lambda t: ls.inject_parity(t, amp)), None
    if name == "alignment":
        return (lambda t: ls.inject_planar_alignment(t, amp, rng)), None
    if name == "cold_spot":
        return (lambda t: ls.inject_cold_spot(t, amp, d)), d
    raise ValueError(name)


def _anomaly(task):
    k, j, i, name, amp, seed = task
    rng = np.random.default_rng([SEED_ANOM, k, j, i])
    hook, d = injector(name, amp, rng)
    return ls.lensed_T_alm(CFG, _SPEC, seed, unlensed_hook=hook).astype(np.complex64), d


def _negatives(task):
    i, seed = task
    tlm, plm = ls.unlensed_T_phi(CFG, _SPEC, seed)
    deltas, Ms, Ls = [], [], []
    lmax_M = L_DEV_RANGE[1]
    for k in range(K_NEG):
        rng = np.random.default_rng([SEED_NEG, i, k])
        L_dev = int(rng.integers(L_DEV_RANGE[0], L_DEV_RANGE[1] + 1))
        M = ls.random_modulation_field(L_dev, rng)
        deltas.append(ls.lens_T(CFG, ls.modulation_term(tlm, M), plm).astype(np.complex64))
        Ms.append(hp.resize_alm(M, L_dev, L_dev, lmax_M, lmax_M).astype(np.complex64))
        Ls.append(L_dev)
    return np.stack(deltas), np.stack(Ms), np.array(Ls)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=32)
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    spec = ls.get_spectra_T(CFG)
    np.savez(OUT_DIR / "spectra.npz", **spec)

    seeds = {"train": SEED_TRAIN + np.arange(N_TRAIN), "null": SEED_NULL + np.arange(N_NULL),
             "ref": SEED_REF + np.arange(N_REF)}
    for name, s in seeds.items():
        if not (OUT_DIR / f"lcdm_{name}.npz").exists():
            np.savez(OUT_DIR / f"lcdm_{name}.npz", alm=lens_many(s, spec, args.workers, name), seed=s)

    if not (OUT_DIR / "anomalies.npz").exists():
        tasks = [(k, j, i, name, amp, seeds["null"][i])
                 for k, (name, amps) in enumerate(ANOMALIES.items())
                 for j, amp in enumerate(amps) for i in range(N_ANOM)]
        with ProcessPoolExecutor(args.workers, initializer=_init, initargs=(spec,)) as ex:
            res = list(tqdm(ex.map(_anomaly, tasks, chunksize=4), total=len(tasks), desc="anomalies"))
        out, rows = {}, []
        for (k, j, i, name, amp, _), (alm, d) in zip(tasks, res):
            key = f"{name}__{amp:g}"
            out.setdefault(key, []).append(alm)
            rows.append({"key": key, "anomaly": name, "amplitude": amp, "null_index": i,
                         **({"dir_x": d[0], "dir_y": d[1], "dir_z": d[2]} if d is not None else {})})
        np.savez(OUT_DIR / "anomalies.npz", **{k: np.stack(v) for k, v in out.items()})
        pd.DataFrame(rows).to_csv(OUT_DIR / "anomalies_meta.csv", index=False)

    if not (OUT_DIR / "train_modulation.npz").exists():
        with ProcessPoolExecutor(args.workers, initializer=_init, initargs=(spec,)) as ex:
            res = list(tqdm(ex.map(_negatives, enumerate(seeds["train"]), chunksize=4),
                            total=N_TRAIN, desc="hard negatives"))
        np.savez(OUT_DIR / "train_modulation.npz", delta=np.stack([r[0] for r in res]),
                 M_alm=np.stack([r[1] for r in res]), L_dev=np.stack([r[2] for r in res]))

    with open(OUT_DIR / "config.json", "w") as fh:
        json.dump({"sky_config": asdict(CFG), "lmax": CFG.lmax, "fiducial": ls.FIDUCIAL,
                   "n_train": N_TRAIN, "n_null": N_NULL, "n_anom": N_ANOM, "n_ref": N_REF,
                   "seeds": {"train": SEED_TRAIN, "null": SEED_NULL, "anom": SEED_ANOM,
                             "ref": SEED_REF, "neg": SEED_NEG},
                   "anomalies": ANOMALIES, "anomaly_injection": "unlensed (primordial) T",
                   "hard_negatives": {"k_per_sky": K_NEG, "L_dev_range": L_DEV_RANGE, "lmin": 2}},
                  fh, indent=2)
    print(f"done in {time.time() - t0:.0f}s -> {OUT_DIR}")


if __name__ == "__main__":
    main()

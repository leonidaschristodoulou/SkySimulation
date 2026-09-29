#!/usr/bin/env python3
"""Extra anomaly test sets for the contrastive PoC (power studies only — the scores
never see anomaly models).

    python examples/contrastive_poc/generate_extra_anomaly_sets.py [--workers 32]

Writes anomalies_extra.npz / anomalies_extra_meta.csv next to the main sets, built
on the same first N_ANOM null seeds, so every sky still has its un-injected twin in
lcdm_null.npz. All injections act on the primordial (unlensed) T before lensing.

  hemi_planckdir__A   dipolar modulation (ℓ ≤ 64) along the observed Planck
                      asymmetry axis, (l, b) = (221°, −20°) Galactic.
  combo_*__level      several anomalies on one sky, injected in the order
                      low power → parity → alignment → hemispherical → cold spot,
                      with the observed directions (asymmetry axis above; cold spot
                      at the Cold Spot position (l, b) = (209°, −57°)).
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import healpy as hp
from tqdm import tqdm

from skysimulation import lensed_sky as ls

OUT_DIR = Path("/nvme/h/lchristodoulou/data_p318/simulations/contrastive_poc_nside64")
CFG = ls.SkyConfig(nside=64)
N_ANOM, SEED_NULL, SEED_EXTRA = 300, 920_000_000, 960_000_000

D_ASYM = hp.ang2vec(221.0, -20.0, lonlat=True)      # Planck hemispherical-asymmetry axis
D_COLD = hp.ang2vec(209.0, -57.0, lonlat=True)      # the Cold Spot

COMBOS = {  # name -> (level, {anomaly: amplitude})
    "combo_mild":        (1, {"low_power": 0.10, "parity": 0.10, "alignment": 0.50, "hemispherical": 0.04}),
    "combo_moderate":    (2, {"low_power": 0.20, "parity": 0.20, "alignment": 0.75, "hemispherical": 0.07}),
    "combo_moderate_cs": (3, {"low_power": 0.20, "parity": 0.20, "alignment": 0.75, "hemispherical": 0.07,
                              "cold_spot": 150.0}),
    "combo_phase":       (2, {"alignment": 0.75, "hemispherical": 0.07}),
}
ORDER = ["low_power", "parity", "alignment", "hemispherical", "cold_spot"]

_SPEC = None


def _init(spec):
    global _SPEC
    _SPEC = spec


def combined_hook(amps, rng):
    def hook(t):
        for name in ORDER:
            if name not in amps:
                continue
            a = amps[name]
            if name == "low_power":
                t = ls.inject_low_power(t, a)
            elif name == "parity":
                t = ls.inject_parity(t, a)
            elif name == "alignment":
                t = ls.inject_planar_alignment(t, a, rng)
            elif name == "hemispherical":
                t = ls.inject_hemispherical(t, a, D_ASYM)
            elif name == "cold_spot":
                t = ls.inject_cold_spot(t, a, D_COLD)
        return t
    return hook


def _one(task):
    k, key, amps, i, seed = task
    rng = np.random.default_rng([SEED_EXTRA, k, i])
    return ls.lensed_T_alm(CFG, _SPEC, seed, unlensed_hook=combined_hook(amps, rng)).astype(np.complex64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=32)
    args = ap.parse_args()
    spec = dict(np.load(OUT_DIR / "spectra.npz"))
    seeds = SEED_NULL + np.arange(N_ANOM)

    sets = {f"hemi_planckdir__{A:g}": {"hemispherical": A} for A in [0.02, 0.04, 0.07, 0.10]}
    sets.update({f"{name}__{level}": amps for name, (level, amps) in COMBOS.items()})
    tasks = [(k, key, amps, i, seeds[i]) for k, (key, amps) in enumerate(sets.items()) for i in range(N_ANOM)]
    with ProcessPoolExecutor(args.workers, initializer=_init, initargs=(spec,)) as ex:
        res = list(tqdm(ex.map(_one, tasks, chunksize=4), total=len(tasks), desc="extra anomalies"))

    out, rows = {}, []
    for (_, key, amps, i, _), alm in zip(tasks, res):
        out.setdefault(key, []).append(alm)
        name, level = key.split("__")
        rows.append({"key": key, "anomaly": name, "amplitude": float(level), "null_index": i,
                     **{f"amp_{k}": v for k, v in amps.items()}})
    np.savez(OUT_DIR / "anomalies_extra.npz", **{k: np.stack(v) for k, v in out.items()})
    pd.DataFrame(rows).to_csv(OUT_DIR / "anomalies_extra_meta.csv", index=False)
    print("wrote", OUT_DIR / "anomalies_extra.npz", list(out))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Download Planck PR3 component-separated temperature products and store them
as nside-256, T-only, µK maps matching the lensed_sky_contrastive_views sims.

    python examples/fetch_planck_T.py [--n-noise 300] [--workers 2]

Downloads:
  * real CMB maps (PLA; SEVEM from IRSA): SMICA / Commander / NILC / SEVEM full mission, plus
    SMICA hm1 / hm2 half-missions — foreground-residual templates and a real-noise
    cross-check via (hm1 - hm2) / 2;
  * FFP10 SMICA noise simulations (PLA): dx12_v3_smica_noise_mc_%05d_raw.fits,
    noise + systematics propagated through the SMICA weights.

Each file is processed as soon as it lands, then the nside-2048 original is
deleted (each noise sim is ~600 MB; all 300 would be ~180 GB):
  T (K_CMB) -> µK -> alm (lmax 767) / pixwin(2048) -> alm2map(nside 256).
Band-limiting in harmonic space avoids aliasing, and like the simulated maps no
nside-256 pixel window is applied. The native 5' SMICA beam is kept, matching the
5' beam of the sims.

Idempotent: finished outputs are skipped, so a crashed run resumes on relaunch.
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "2")

import argparse
import csv
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import healpy as hp

ROOT      = Path("/nvme/h/lchristodoulou/data_p318/planck")
RAW_DIR   = ROOT / "raw"
OUT_DIR   = ROOT / "nside256"
NSIDE_OUT = 256
LMAX      = 3 * NSIDE_OUT - 1
NSIDE_IN  = 2048

# PLA serves ~15 MB/s vs ~0.5 MB/s per stream from IRSA, but refuses SEVEM (401), so that one comes from IRSA.
IRSA     = "https://irsa.ipac.caltech.edu/data/Planck/release_3/all-sky-maps/maps/component-maps/cmb/"
PLA_MAP  = "https://pla.esac.esa.int/pla/aio/product-action?MAP.MAP_ID="
PLA_SIM  = "https://pla.esac.esa.int/pla/aio/product-action?SIMULATED_MAP.FILE_ID="

DATA_MAPS = {  # output name -> (base URL, source file)
    "smica_full":     (PLA_MAP, "COM_CMB_IQU-smica_2048_R3.00_full.fits"),
    "smica_hm1":      (PLA_MAP, "COM_CMB_IQU-smica_2048_R3.00_hm1.fits"),
    "smica_hm2":      (PLA_MAP, "COM_CMB_IQU-smica_2048_R3.00_hm2.fits"),
    "commander_full": (PLA_MAP, "COM_CMB_IQU-commander_2048_R3.00_full.fits"),
    "nilc_full":      (PLA_MAP, "COM_CMB_IQU-nilc_2048_R3.00_full.fits"),
    "sevem_full":     (IRSA,    "COM_CMB_IQU-sevem_2048_R3.00_full.fits"),
}


def tasks(n_noise):
    out = [(base + src, src, OUT_DIR / "data" / f"planck_T_{name}_nside{NSIDE_OUT}.fits")
           for name, (base, src) in DATA_MAPS.items()]
    for i in range(n_noise):
        src = f"dx12_v3_smica_noise_mc_{i:05d}_raw.fits"
        out.append((PLA_SIM + src, src,
                    OUT_DIR / "noise_ffp10_smica" / f"ffp10_smica_noise_T_{i:05d}_nside{NSIDE_OUT}.fits"))
    return out


def download(url, dest):
    # curl to a .part file then rename, so an interrupted download is never mistaken for a full one.
    # No overall time limit (IRSA is slow); a stalled transfer is aborted and retried. The PLA
    # ignores Range requests, so retries restart from zero, and more than ~2 parallel streams get dropped.
    if dest.exists():
        return
    part = dest.with_suffix(".part")
    subprocess.run(["curl", "-sSfL", "--retry", "5", "--retry-delay", "10", "--retry-all-errors",
                    "--speed-limit", "50000", "--speed-time", "120",
                    "-o", str(part), url], check=True)
    part.rename(dest)


def downgrade_T(path):
    T = hp.read_map(str(path), field=0, dtype=np.float64) * 1e6         # K_CMB -> µK
    alm = hp.map2alm(T, lmax=LMAX, iter=1, use_pixel_weights=False)
    alm = hp.almxfl(alm, 1.0 / hp.pixwin(NSIDE_IN, lmax=LMAX))
    return hp.alm2map(alm, NSIDE_OUT).astype(np.float32)


def run_one(task):
    url, src, out = task
    if out.exists():
        return out, "skip", 0.0
    t0 = time.time()
    raw = RAW_DIR / src
    download(url, raw)
    m = downgrade_T(raw)
    out.parent.mkdir(parents=True, exist_ok=True)
    hp.write_map(str(out), m, dtype=np.float32, overwrite=True,
                 column_names=["TEMPERATURE"], column_units="uK_CMB",
                 extra_header=[("SOURCE", src), ("LMAX", LMAX), ("BEAM", "native 5' SMICA-like")])
    raw.unlink()
    return out, "ok", time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-noise", type=int, default=300)
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    todo = tasks(args.n_noise)
    failed = []
    with ProcessPoolExecutor(args.workers) as ex:
        futs = {ex.submit(run_one, t): t for t in todo}
        for k, f in enumerate(as_completed(futs), 1):
            try:
                out, status, dt = f.result()
                print(f"[{k}/{len(todo)}] {status:4s} {dt:6.1f}s {out.name}", flush=True)
            except Exception as e:  # keep going; failures are listed at the end and retried on relaunch
                failed.append(futs[f][1])
                print(f"[{k}/{len(todo)}] FAIL {futs[f][1]}: {e}", flush=True)

    with open(OUT_DIR / "manifest.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["kind", "path", "source"])
        for url, src, out in todo:
            if out.exists():
                w.writerow(["noise" if "noise" in src else "data", out.relative_to(OUT_DIR), src])
    print(f"done; {len(failed)} failed: {failed}")


if __name__ == "__main__":
    main()

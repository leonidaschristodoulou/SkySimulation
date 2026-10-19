#!/usr/bin/env python3
"""Simulation suite for the a priori anomaly detector (spec: cmb-latent/sims_suite.md).

Stage 1 skies only — natural, lensed, full sky, no instrument — stored as T alms to ℓ = 191
(complex64). Views (rotation, 160′ smoothing, noise, mask) and the training-only C_ℓ
rescaling are made on the fly by cmb-latent, exactly as for the PoC; the output files keep
the PoC format, so contrastive_train / contrastive_eval read them unchanged.

    python generate_suite.py plan                      # parameters, seeds, injections, task list
    python generate_suite.py run --task K --workers 40 # one shard (a slurm array element)
    python generate_suite.py merge                     # shards -> final files
    python generate_suite.py check                     # CI gates of sims_suite.md §8 (those not needing a network)

Every step is deterministic in MASTER and resumable (finished shards are skipped).

Cosmology: every sky draws its own parameters from the Planck 2018 TT,TE,EE+lowE+lensing chain
(one CAMB spectrum per sky). The chain's four files are split into halves: training draws from
files 1–2, every evaluation set from files 3–4. An anomaly sky takes the parameters *and* the
realization of its null twin, so the pair differs by the injection alone.

Seeds: hp.synalm draws from numpy's legacy global state (32-bit seed), so the realization seed is
REAL_BASE + absolute sky index — distinct by construction (hashing 86k skies into 32 bits would
collide). Everything else (parameter draws, injections, hard negatives) uses
SeedSequence([MASTER, stream, ...]).

Reproducibility: plan/ holds every sky's parameters, chain row and seeds; input_spectra/ holds the
exact CAMB spectra each sky was drawn from ([TT, φφ, Tφ], float64, ℓ ≤ 1200; anomaly skies use their
twin's), so any sky can be regenerated without rerunning CAMB; `check` regenerates a few and compares.
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import hashlib
import io
import json
import shutil
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path

import healpy as hp
import numpy as np
import pandas as pd

from skysimulation import lensed_sky as ls

OUT_DIR = Path(os.environ.get("SUITE_OUT_DIR", "/nvme/h/lchristodoulou/data_p318/simulations/suite_nside64"))
DATA = Path("/nvme/h/lchristodoulou/data_p318")
PLANCK_ROOT = DATA / ("COM_CosmoParams_base-plikHM-TTTEEE-lowl-lowE_R3/base/plikHM_TTTEEE_lowl_lowE_lensing/"
                      "base_plikHM_TTTEEE_lowl_lowE_lensing")
W0WA_CHAIN = DATA / ("base_w_wa-desi-bao-all_planck2018-lowl-TT-clik_planck2018-lowl-EE-clik_"
                     "planck-NPIPE-highl-CamSpec-TTTEEE/chain")
POC_DIR = DATA / "simulations/contrastive_poc_nside64"

CFG = ls.SkyConfig(nside=64)
MASTER = 20261015
REAL_BASE = 3_000_000_000              # + absolute index < 2**32
BURN_IN = 0.3                          # fraction of each chain file dropped
CHAIN_HALVES = {"T": (1, 2), "E": (3, 4)}
STREAM = {"params": 1, "negatives": 2, "injection": 3, "control": 4}

# ΛCDM sets: name -> (first absolute index, N, chain half, shard size)   (sims_suite.md §2, §4)
LCDM = {"train":   (0,       20_000, "T", 500),
        "ref":     (100_000, 10_000, "E", 1000),
        "null":    (200_000, 10_000, "E", 1000),
        "holdout": (300_000,  5_000, "E", 1000),
        "gateA":   (400_000,  5_000, "E", None),     # gates are generated interleaved, A/B alternating,
        "gateB":   (405_000,  5_000, "E", None)}     # in the "gates" shards below
GATE_SHARD = 1000                                    # skies per gates shard (half A, half B)
# Extensions, planned after the suite was frozen (`extend`): name -> (first absolute index, N, chain half, shard
# size, null_index base for the evaluator's view seeds). They leave the original plan and its hashes untouched.
#   null2: a second, independent null (2026-10-19). The 10,000-sky null did not fix the 0.1% threshold of the
#          global statistic stably (holdout/gate checks), so thresholds are to be set from the pooled null.
EXTENSIONS = {"null2": (700_000, 30_000, "E", 1000, 6_000_000)}
# Anomaly extensions: name -> (first absolute index, {anomaly: (amplitudes, N twins)}); injected on the null twins
# like §3a, so the evaluator pairs them with the same view seeds.
#   combo: alignment and hemispherical asymmetry on one sky (2026-10-19), at the amplitudes of the single sets
#          they are compared with (alignment α = 1, hemispherical A = 0.07 along the Planck axis).
ANOMALY_EXTENSIONS = {"combo": (740_000, {"combo_phase": ([1.0], 2000)})}
COMBOS = {"combo_phase": [("alignment", 1.0), ("hemi_planckdir", 0.07)]}   # injected in this order (the PoC's)
K_NEG, L_DEV_RANGE = 4, (2, 10)                      # hard negatives per training sky

# Anomalies (§3a, §3c): file -> {anomaly: (amplitudes, N twins)}; twins are null skies 0..N-1
D_ASYM = hp.ang2vec(221.0, -20.0, lonlat=True)       # Planck hemispherical-asymmetry axis
ANOMALY_FILES = {
    "anomalies": {"hemispherical": ([0.04, 0.07, 0.10, 0.14], 2000),
                  "alignment":     ([1.0], 2000),
                  "low_power":     ([0.20], 2000),
                  "parity":        ([0.30], 2000),
                  "cold_spot":     ([200.0], 2000)},
    "anomalies_planck": {"hemi_planckdir": ([0.07, 0.10], 2000)},
    "anomalies_posctrl": {"modulation_L2-10": ([round(a / np.sqrt(3), 5) for a in (0.04, 0.07, 0.10, 0.14)] + [0.12],
                                               1000)},
}
ANOM_FIRST, ANOM_TWINS_PER_SHARD = 500_000, 40   # one twin per worker

# Controls (§3d)
W0WA_N, EDE_SETS, EDE_SEEDS = 2000, 400, 5           # per w0wa point; EDE sets × seeds each
CTRL_FIRST = 550_000                                 # w0wa peak, w0wa 95% edge, then EDE
EDE_W_N, EDE_SETS_PER_SHARD = 0.5, 40
FIT_LMAX, FSKY_FIT, NOISE_FIT, BEAM_FIT = 2000, 0.7, 30.0, 7.0   # Planck-like TT χ² of the EDE refit (as the PoC)
CTRL_VIEW_BASE = 1_000_000                           # null_index for controls: view seeds disjoint from null/ref

LMAX = CFG.lmax


# ═══════════════════════════════ plan ═══════════════════════════════

def load_planck_half(files):
    names = [l.split()[0] for l in open(f"{PLANCK_ROOT}.paramnames")]
    rows = []
    for f in files:
        a = np.loadtxt(f"{PLANCK_ROOT}_{f}.txt")
        a = a[int(BURN_IN * len(a)):]
        rows.append(pd.DataFrame(a[:, 2:2 + len(names)], columns=names).assign(weight=a[:, 0], chain_file=f,
                                                                              chain_row=np.arange(len(a))))
    return pd.concat(rows, ignore_index=True)


def chain_params(row):
    return dict(ls.FIDUCIAL, H0=float(row["H0*"]), ombh2=float(row["omegabh2"]), omch2=float(row["omegach2"]),
                ns=float(row["ns"]), tau=max(float(row["tau"]), 0.01), As=float(np.exp(row["logA"]) * 1e-10))


def draw_lcdm(name, first, n, half, chains):
    ch = chains[half]
    rng = np.random.default_rng(np.random.SeedSequence([MASTER, STREAM["params"], first]))
    pick = rng.choice(len(ch), size=n, replace=True, p=ch.weight / ch.weight.sum())
    df = pd.DataFrame([chain_params(ch.iloc[k]) for k in pick])
    df.insert(0, "abs_index", first + np.arange(n))
    df["real_seed"] = REAL_BASE + df.abs_index
    df["chain_file"], df["chain_row"] = ch.chain_file.to_numpy()[pick], ch.chain_row.to_numpy()[pick]
    return df


def w0wa_edges():
    """Two w0wa control points from the DESI BAO + Planck chain, read off the smoothed 2D marginal
    posterior of (w0, wa) (banana-shaped, so not a Gaussian ellipse): its peak (the best-supported
    dynamical dark energy) and the point on its 95% contour farthest from ΛCDM (w0, wa) = (−1, 0) in
    the posterior metric. Each is replaced by the nearest chain sample, so the other parameters stay
    consistent with it."""
    from getdist import loadMCSamples
    from scipy.ndimage import gaussian_filter
    s = loadMCSamples(str(W0WA_CHAIN), settings={"ignore_rows": BURN_IN})
    p = s.getParams()
    x = np.stack([p.w, p.wa], 1)
    w = s.weights / s.weights.sum()
    C = np.cov(x.T, aweights=w)
    Ci = np.linalg.inv(C)
    H, ex, ey = np.histogram2d(x[:, 0], x[:, 1], bins=200, weights=w)
    H = gaussian_filter(H, 3)
    H /= H.sum()
    order = np.sort(H.ravel())[::-1]
    level95 = order[np.searchsorted(np.cumsum(order), 0.95)]
    cx, cy = 0.5 * (ex[1:] + ex[:-1]), 0.5 * (ey[1:] + ey[:-1])
    gx, gy = np.meshgrid(cx, cy, indexing="ij")
    peak = np.array([gx.flat[H.argmax()], gy.flat[H.argmax()]])
    on95 = np.abs(H - level95) < 0.1 * level95                       # grid cells along the 95% contour
    pts = np.stack([gx[on95], gy[on95]], 1)
    d_lcdm = np.einsum("ni,ij,nj->n", pts - [-1, 0], Ci, pts - [-1, 0])
    edge = pts[np.argmax(d_lcdm)]
    out = []
    for name, target in [("peak", peak), ("edge95", edge)]:
        d = np.einsum("ni,ij,nj->n", x - target, Ci, x - target)
        k = int(np.argmin(d))
        out.append(dict(ls.FIDUCIAL, H0=float(p.H0[k]), ombh2=float(p.ombh2[k]), omch2=float(p.omch2[k]),
                        ns=float(p.ns[k]), tau=max(float(p.tau[k]), 0.01), As=float(np.exp(p.logA[k]) * 1e-10),
                        w0=float(p.w[k]), wa=float(p.wa[k]), point=name,
                        target_w0=float(target[0]), target_wa=float(target[1])))
    return out, {"mean": (w @ x).tolist(), "cov": C.tolist(), "peak": peak.tolist(), "edge95": edge.tolist()}


def plan(args):
    out = OUT_DIR
    (out / "plan").mkdir(parents=True, exist_ok=True)
    (out / "shards").mkdir(exist_ok=True)
    chains = {h: load_planck_half(f) for h, f in CHAIN_HALVES.items()}
    for name, (first, n, half, _) in LCDM.items():
        draw_lcdm(name, first, n, half, chains).to_csv(out / "plan" / f"params_{name}.csv", index=False)
    null = pd.read_csv(out / "plan" / "params_null.csv")

    # anomalies: one row per injected sky, ordered by file, anomaly, amplitude, twin
    rows, k = [], ANOM_FIRST
    for fname, spec in ANOMALY_FILES.items():
        for anomaly, (amps, n_twins) in spec.items():
            for amp in amps:
                for i in range(n_twins):
                    rows.append({"file": fname, "key": f"{anomaly}__{amp:g}", "anomaly": anomaly, "amplitude": amp,
                                 "null_index": i, "abs_index": k})
                    k += 1
    assert k <= CTRL_FIRST
    pd.DataFrame(rows).to_csv(out / "plan" / "anomalies.csv", index=False)

    # controls
    edges, w0wa_stats = w0wa_edges()
    rng = np.random.default_rng(np.random.SeedSequence([MASTER, STREAM["control"]]))
    ede_sets = [{"set": j, "f_ede": float(rng.uniform(1e-4, 0.08)), "logzc": float(rng.uniform(3.3, 3.8)),
                 "theta_i": float(rng.uniform(0.5, 3.0))} for j in range(EDE_SETS)]
    ctrl, k = [], CTRL_FIRST
    for e, pe in enumerate(edges):
        for i in range(W0WA_N):
            ctrl.append({"key": f"w0wa_{pe['point']}__0", "model": "w0wa", "edge": e, "abs_index": k}); k += 1
    for s in ede_sets:
        for i in range(EDE_SEEDS):
            ctrl.append({"key": "ede__0", "model": "ede", "ede_set": s["set"], "abs_index": k}); k += 1
    ctrl = pd.DataFrame(ctrl)
    ctrl["real_seed"] = REAL_BASE + ctrl.abs_index
    ctrl["null_index"] = CTRL_VIEW_BASE + np.arange(len(ctrl))
    ctrl.to_csv(out / "plan" / "controls.csv", index=False)
    pd.DataFrame(edges).to_csv(out / "plan" / "w0wa_edges.csv", index=False)
    pd.DataFrame(ede_sets).to_csv(out / "plan" / "ede_sets.csv", index=False)

    # tasks (one slurm array element each)
    tasks = []
    for name in ["train", "ref", "null", "holdout"]:
        first, n, _, size = LCDM[name]
        tasks += [{"kind": "lcdm", "set": name, "lo": lo, "hi": min(lo + size, n)} for lo in range(0, n, size)]
    tasks += [{"kind": "gates", "set": "gates", "lo": lo, "hi": lo + GATE_SHARD // 2}
              for lo in range(0, LCDM["gateA"][1], GATE_SHARD // 2)]
    n_twins = max(t for spec in ANOMALY_FILES.values() for _, t in spec.values())
    tasks += [{"kind": "anomalies", "set": "anomalies", "lo": lo, "hi": min(lo + ANOM_TWINS_PER_SHARD, n_twins)}
              for lo in range(0, n_twins, ANOM_TWINS_PER_SHARD)]
    tasks += [{"kind": "w0wa", "set": "w0wa", "lo": lo, "hi": lo + 1000} for lo in range(0, 2 * W0WA_N, 1000)]
    tasks += [{"kind": "ede", "set": "ede", "lo": lo, "hi": min(lo + EDE_SETS_PER_SHARD, EDE_SETS)}
              for lo in range(0, EDE_SETS, EDE_SETS_PER_SHARD)]
    for j, t in enumerate(tasks):
        t["task"] = j
    pd.DataFrame(tasks).to_csv(out / "plan" / "tasks.csv", index=False)

    # the fiducial spectra (training-time C_ℓ rescaling target) and the PoC ε calibration
    fid = ls.get_spectra_T(CFG)
    np.savez(out / "spectra.npz", **fid)
    shutil.copy(POC_DIR / "modulation_eps_calibration.json", out / "modulation_eps_calibration.json")

    # provenance; the plan hash is asserted at merge time
    def git(*a):
        repo = Path(__file__).resolve().parents[2]
        return subprocess.run(["git", "-C", str(repo), *a], capture_output=True, text=True).stdout.strip()
    import camb, lenspyx, ducc0
    ver = lambda m: getattr(m, "__version__", "unknown")
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((out / "plan").glob("*.csv"))}
    cfg = {"spec": "cmb-latent/sims_suite.md", "sky_config": asdict(CFG), "lmax": LMAX, "fiducial": ls.FIDUCIAL,
           "master": MASTER, "real_base": REAL_BASE, "streams": STREAM, "burn_in": BURN_IN,
           "planck_chain": str(PLANCK_ROOT), "chain_halves": CHAIN_HALVES, "w0wa_chain": str(W0WA_CHAIN),
           "w0wa_posterior": w0wa_stats,
           "sets": {k: {"first": v[0], "n": v[1], "chain_half": v[2]} for k, v in LCDM.items()},
           "hard_negatives": {"k_per_sky": K_NEG, "L_dev_range": L_DEV_RANGE, "lmin": 2},
           "anomalies": {f: {a: {"amplitudes": am, "n_twins": n} for a, (am, n) in s.items()}
                         for f, s in ANOMALY_FILES.items()},
           "planck_axis_lb_deg": [221.0, -20.0], "anomaly_injection": "unlensed (primordial) T",
           "controls": {"w0wa_n_per_edge": W0WA_N, "ede_sets": EDE_SETS, "ede_seeds_per_set": EDE_SEEDS,
                        "ede_w_n": EDE_W_N},
           "eps_calibration": f"copied from {POC_DIR} (same view model; skies rescaled to the same fiducial)",
           "git": {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain", "--", "."))},
           "versions": {m.__name__: ver(m) for m in (camb, hp, lenspyx, ducc0, np)},
           "plan_sha256": hashes}
    with open(out / "config.json", "w") as fh:
        json.dump(cfg, fh, indent=2)
    shutil.copy(__file__, out / "generate_suite.py")   # the exact code that made the suite
    print(f"plan: {len(tasks)} tasks -> {out / 'plan' / 'tasks.csv'}")
    print(pd.DataFrame(tasks).groupby("kind").size().to_string())
    print("w0wa edges:\n", pd.DataFrame(edges)[["point", "w0", "wa", "target_w0", "target_wa", "H0"]].round(3).to_string())


# ═══════════════════════════════ workers ═══════════════════════════════

def _lcdm_sky(row, negatives=False):
    """(alm, len_TT[:LMAX+1], input spectra [TT, PP, PT] (float64, to lmax_unl), [delta, M_alm, L_dev])
    for one ΛCDM sky of the plan. The input spectra make the sky regenerable without CAMB."""
    spec = ls.get_spectra_T(CFG, row)
    tlm, plm = ls.unlensed_T_phi(CFG, spec, int(row["real_seed"]))
    alm = ls.lens_T(CFG, tlm, plm).astype(np.complex64)
    out = [alm, spec["len_TT"][:LMAX + 1].astype(np.float32), input_spectra(spec)]
    if negatives:
        deltas, Ms, Ls = [], [], []
        for k in range(K_NEG):
            rng = np.random.default_rng(np.random.SeedSequence([MASTER, STREAM["negatives"], int(row["abs_index"]), k]))
            L_dev = int(rng.integers(L_DEV_RANGE[0], L_DEV_RANGE[1] + 1))
            M = ls.random_modulation_field(L_dev, rng)
            deltas.append(ls.lens_T(CFG, ls.modulation_term(tlm, M), plm).astype(np.complex64))
            Ms.append(hp.resize_alm(M, L_dev, L_dev, L_DEV_RANGE[1], L_DEV_RANGE[1]).astype(np.complex64))
            Ls.append(L_dev)
        out += [np.stack(deltas), np.stack(Ms), np.array(Ls)]
    return out


def input_spectra(spec):
    return np.stack([spec["TT"], spec["PP"], spec["PT"]]).astype(np.float64)


def _w_train(row):
    return _lcdm_sky(row, negatives=True)


def _w_lcdm(row):
    return _lcdm_sky(row)


def injection_hook(anomaly, amp, rng):
    """(hook on the unlensed tlm, metadata). Directions and M are drawn from rng."""
    if anomaly == "hemispherical":
        d = ls.random_direction(rng)
        return (lambda t: ls.inject_hemispherical(t, amp, d)), {"dir_x": d[0], "dir_y": d[1], "dir_z": d[2]}
    if anomaly == "hemi_planckdir":
        return (lambda t: ls.inject_hemispherical(t, amp, D_ASYM)), {}
    if anomaly == "low_power":
        return (lambda t: ls.inject_low_power(t, amp)), {}
    if anomaly == "parity":
        return (lambda t: ls.inject_parity(t, amp)), {}
    if anomaly == "alignment":
        return (lambda t: ls.inject_planar_alignment(t, amp, rng)), {}
    if anomaly == "cold_spot":
        d = ls.random_direction(rng)
        return (lambda t: ls.inject_cold_spot(t, amp, d)), {"dir_x": d[0], "dir_y": d[1], "dir_z": d[2]}
    if anomaly in COMBOS:          # amp is a level label; the parts and their amplitudes are fixed in COMBOS
        parts = [injection_hook(a, x, rng) for a, x in COMBOS[anomaly]]
        def hook(t):
            for h, _ in parts:
                t = h(t)
            return t
        return hook, {k: v for _, m in parts for k, v in m.items()}
    if anomaly == "modulation_L2-10":
        L_dev = int(rng.integers(L_DEV_RANGE[0], L_DEV_RANGE[1] + 1))
        M = ls.random_modulation_field(L_dev, rng)
        return (lambda t: t + amp * ls.modulation_term(t, M)), {"L_dev": L_dev}
    raise ValueError(anomaly)


def _w_twin(task):
    """All injections on one null twin: its spectrum and (T, φ) drawn once, each hook on a copy."""
    twin, rows = task
    spec = ls.get_spectra_T(CFG, twin)
    tlm, plm = ls.unlensed_T_phi(CFG, spec, int(twin["real_seed"]))
    out = []
    for r in rows:
        rng = np.random.default_rng(np.random.SeedSequence([MASTER, STREAM["injection"], int(r["abs_index"])]))
        hook, meta = injection_hook(r["anomaly"], r["amplitude"], rng)
        out.append((int(r["abs_index"]), ls.lens_T(CFG, hook(tlm.copy()), plm).astype(np.complex64), meta))
    return out


_FID = {}


def _init_ede():
    fid_cfg = ls.SkyConfig(nside=CFG.nside, lmax_unl=FIT_LMAX, nside_lens=CFG.nside_lens)
    fid = ls.get_spectra_T_extended(fid_cfg, ls.FIDUCIAL)
    _FID.update(len_TT=fid["len_TT"], thetastar=fid["thetastar"], fit_cfg=fid_cfg)


def _chi_residuals(spec_len):
    ell = np.arange(2, FIT_LMAX + 1)
    b2 = np.exp(-ell * (ell + 1) * (np.radians(BEAM_FIT / 60) / np.sqrt(8 * np.log(2))) ** 2)
    N = (np.radians(NOISE_FIT / 60)) ** 2 / b2
    fid = _FID["len_TT"]
    return (spec_len[ell] - fid[ell]) / (np.sqrt(2 / ((2 * ell + 1) * FSKY_FIT)) * (fid[ell] + N))


def _w_ede(task):
    """Refit (ω_c, n_s, ln A_s) at fixed θ* for one EDE set (as the PoC), then its skies."""
    from scipy.optimize import least_squares
    s, rows = task
    ede = (EDE_W_N, s["f_ede"], 10 ** s["logzc"], s["theta_i"])

    def spec_of(x, cfg):
        p = dict(ls.FIDUCIAL, omch2=x[0], ns=x[1], As=np.exp(x[2]) * 1e-10)
        return ls.get_spectra_T_extended(cfg, p, ede=ede, thetastar=_FID["thetastar"])

    x0 = [ls.FIDUCIAL["omch2"], ls.FIDUCIAL["ns"], np.log(ls.FIDUCIAL["As"] * 1e10)]
    r = least_squares(lambda x: _chi_residuals(spec_of(x, _FID["fit_cfg"])["len_TT"]), x0,
                      x_scale=[0.002, 0.005, 0.01], diff_step=[1e-3, 1e-3, 1e-3], max_nfev=40)
    spec = spec_of(r.x, CFG)
    fit = {**s, "omch2": r.x[0], "ns": r.x[1], "As": np.exp(r.x[2]) * 1e-10, "H0": spec["H0"],
           "chi2_fit": float(np.sum(r.fun ** 2))}
    out = [(int(row["abs_index"]), ls.lensed_T_alm(CFG, spec, int(row["real_seed"])).astype(np.complex64),
            spec["len_TT"][:LMAX + 1].astype(np.float32), input_spectra(spec)) for row in rows]
    return fit, out


def _w_w0wa(task):
    edge, rows = task
    p = {k: edge[k] for k in ["H0", "ombh2", "omch2", "ns", "tau", "As", "mnu", "omk"]}
    spec = ls.get_spectra_T_extended(CFG, p, w0=edge["w0"], wa=edge["wa"])
    return [(int(row["abs_index"]), ls.lensed_T_alm(CFG, spec, int(row["real_seed"])).astype(np.complex64),
             spec["len_TT"][:LMAX + 1].astype(np.float32), input_spectra(spec)) for row in rows]


# ═══════════════════════════════ run ═══════════════════════════════

def run(args):
    task = pd.read_csv(OUT_DIR / "plan" / args.tasks_file, keep_default_na=False).iloc[args.task]  # a set is named "null"
    path = OUT_DIR / "shards" / f"{task['set']}_{int(task['task']):03d}.npz"
    if path.exists():
        print(f"{path.name} exists, skipping"); return
    t0 = time.time()
    lo, hi, kind, name = int(task["lo"]), int(task["hi"]), task["kind"], task["set"]
    if args.limit:                   # testing only: the first few skies / twins / sets of the shard
        hi = min(hi, lo + args.limit)
    pool = lambda init=None: ProcessPoolExecutor(args.workers, initializer=init)
    res = {}
    if kind in ("lcdm", "gates"):
        if kind == "gates":     # interleaved: A_lo, B_lo, A_lo+1, B_lo+1, ...
            a = pd.read_csv(OUT_DIR / "plan" / "params_gateA.csv").iloc[lo:hi]
            b = pd.read_csv(OUT_DIR / "plan" / "params_gateB.csv").iloc[lo:hi]
            df = pd.concat([a, b]).iloc[np.ravel(np.column_stack([np.arange(len(a)), len(a) + np.arange(len(b))]))]
        else:
            df = pd.read_csv(OUT_DIR / "plan" / f"params_{name}.csv").iloc[lo:hi]
        rows = [r.to_dict() for _, r in df.iterrows()]
        fn = _w_train if name == "train" else _w_lcdm
        with pool() as ex:
            out = list(ex.map(fn, rows, chunksize=1))
        res = {"abs_index": df.abs_index.to_numpy(), "alm": np.stack([o[0] for o in out]),
               "len_TT": np.stack([o[1] for o in out]), "in_spec": np.stack([o[2] for o in out])}
        if name == "train":
            res.update(delta=np.stack([o[3] for o in out]), M_alm=np.stack([o[4] for o in out]),
                       L_dev=np.stack([o[5] for o in out]))
    elif kind in ("anomalies", "anomalies_ext"):
        null = pd.read_csv(OUT_DIR / "plan" / "params_null.csv")
        an = pd.read_csv(OUT_DIR / "plan" / ("anomalies.csv" if kind == "anomalies" else f"anomalies_{name}.csv"))
        an = an[(an.null_index >= lo) & (an.null_index < hi)]
        jobs = [(null.iloc[i].to_dict(), [r.to_dict() for _, r in g.iterrows()]) for i, g in an.groupby("null_index")]
        with pool() as ex:
            out = [x for o in ex.map(_w_twin, jobs, chunksize=1) for x in o]
        out.sort(key=lambda x: x[0])
        meta = pd.DataFrame([{"abs_index": k, **m} for k, _, m in out])
        res = {"abs_index": np.array([o[0] for o in out]), "alm": np.stack([o[1] for o in out]),
               "meta_json": np.array(meta.to_json(orient="records"))}
    elif kind == "w0wa":
        ctrl = pd.read_csv(OUT_DIR / "plan" / "controls.csv")
        edges = pd.read_csv(OUT_DIR / "plan" / "w0wa_edges.csv")
        c = ctrl[ctrl.model == "w0wa"].iloc[lo:hi]
        chunks = [(edges.iloc[int(e)].to_dict(), [r.to_dict() for _, r in g.iloc[j:j + 25].iterrows()])
                  for e, g in c.groupby("edge") for j in range(0, len(g), 25)]
        with pool() as ex:
            out = [x for o in ex.map(_w_w0wa, chunks, chunksize=1) for x in o]
        out.sort(key=lambda x: x[0])
        res = {"abs_index": np.array([o[0] for o in out]), "alm": np.stack([o[1] for o in out]),
               "len_TT": np.stack([o[2] for o in out]), "in_spec": np.stack([o[3] for o in out])}
    elif kind == "ede":
        ctrl = pd.read_csv(OUT_DIR / "plan" / "controls.csv")
        sets = pd.read_csv(OUT_DIR / "plan" / "ede_sets.csv").iloc[lo:hi]
        jobs = [(s.to_dict(), [r.to_dict() for _, r in ctrl[(ctrl.model == "ede") & (ctrl.ede_set == s["set"])].iterrows()])
                for _, s in sets.iterrows()]
        with pool(_init_ede) as ex:
            out = list(ex.map(_w_ede, jobs, chunksize=1))
        skies = sorted([x for _, o in out for x in o], key=lambda x: x[0])
        res = {"abs_index": np.array([x[0] for x in skies]), "alm": np.stack([x[1] for x in skies]),
               "len_TT": np.stack([x[2] for x in skies]), "in_spec": np.stack([x[3] for x in skies]),
               "fits_json": np.array(pd.DataFrame([f for f, _ in out]).to_json(orient="records"))}
    tmp = path.with_suffix(".tmp.npz")
    path.parent.mkdir(exist_ok=True)   # shards/ may have been removed after an earlier merge
    np.savez(tmp, **res)
    tmp.rename(path)                 # atomic: a shard either exists complete or not at all
    print(f"{path.name}: {len(res['abs_index'])} skies in {time.time() - t0:.0f}s")


# ═══════════════════════════════ merge ═══════════════════════════════

def merge(args):
    out = OUT_DIR
    cfg = json.loads((out / "config.json").read_text())
    for name, h in cfg["plan_sha256"].items():
        assert hashlib.sha256((out / "plan" / name).read_bytes()).hexdigest() == h, f"plan file {name} changed"
    tasks = pd.read_csv(out / "plan" / "tasks.csv", keep_default_na=False)  # a set is named "null"
    (out / "input_spectra").mkdir(exist_ok=True)
    missing = [f"{t.set}_{t.task:03d}" for t in tasks.itertuples() if not (out / "shards" / f"{t.set}_{t.task:03d}.npz").exists()]
    if missing:
        raise SystemExit(f"{len(missing)} shards missing, e.g. {missing[:5]}")

    def shards(s):
        return [np.load(out / "shards" / f"{s}_{t.task:03d}.npz") for t in tasks[tasks["set"] == s].itertuples()]

    def cat(parts, key):
        return np.concatenate([p[key] for p in parts])

    # ΛCDM sets
    for name in ["train", "ref", "null", "holdout"]:
        parts = shards(name)
        idx = cat(parts, "abs_index")
        par = pd.read_csv(out / "plan" / f"params_{name}.csv")
        assert np.array_equal(idx, par.abs_index.to_numpy()), name
        np.savez(out / f"lcdm_{name}.npz", alm=cat(parts, "alm"), seed=par.real_seed.to_numpy(), abs_index=idx,
                 len_TT=cat(parts, "len_TT"))
        np.save(out / "input_spectra" / f"{name}.npy", cat(parts, "in_spec"))
        if name == "train":
            np.savez(out / "train_modulation.npz", delta=cat(parts, "delta"), M_alm=cat(parts, "M_alm"),
                     L_dev=cat(parts, "L_dev"))
        print(f"lcdm_{name}.npz: {len(idx)}")
    parts = shards("gates")
    idx, alm, cl, ins = cat(parts, "abs_index"), cat(parts, "alm"), cat(parts, "len_TT"), cat(parts, "in_spec")
    for g in ["gateA", "gateB"]:
        par = pd.read_csv(out / "plan" / f"params_{g}.csv")
        sel = np.isin(idx, par.abs_index)
        assert np.array_equal(idx[sel], par.abs_index.to_numpy()), g
        np.savez(out / f"lcdm_{g}.npz", alm=alm[sel], seed=par.real_seed.to_numpy(), abs_index=idx[sel], len_TT=cl[sel])
        np.save(out / "input_spectra" / f"{g}.npy", ins[sel])
        print(f"lcdm_{g}.npz: {sel.sum()}")

    # anomalies: one file per group, keys "<anomaly>__<amp>", rows sorted by null_index (the evaluator's order)
    parts = shards("anomalies")
    idx, alm = cat(parts, "abs_index"), cat(parts, "alm")
    meta = pd.concat([pd.read_json(io.StringIO(str(p["meta_json"])), orient="records") for p in parts], ignore_index=True)
    an = pd.read_csv(out / "plan" / "anomalies.csv").merge(meta, on="abs_index", how="left")
    pos = pd.Series(np.arange(len(idx)), index=idx)
    assert len(idx) == len(an) and pos.index.is_unique and set(idx) == set(an.abs_index)
    for fname, g in an.groupby("file"):
        g = g.sort_values(["key", "null_index"])
        np.savez(out / f"{fname}.npz", **{k: alm[pos[gk.abs_index].to_numpy()] for k, gk in g.groupby("key", sort=False)})
        g.drop(columns="file").to_csv(out / f"{fname}_meta.csv", index=False)
        print(f"{fname}.npz: {g.key.nunique()} keys, {len(g)} skies")

    # controls -> anomalies_adjacent.npz (the anomaly-set format, so contrastive_eval scores them)
    ctrl = pd.read_csv(out / "plan" / "controls.csv")
    parts = shards("w0wa") + shards("ede")
    idx, alm, cl = cat(parts, "abs_index"), cat(parts, "alm"), cat(parts, "len_TT")
    pos = pd.Series(np.arange(len(idx)), index=idx)
    assert set(idx) == set(ctrl.abs_index)
    np.savez(out / "anomalies_adjacent.npz",
             **{k: alm[pos[g.sort_values("null_index").abs_index].to_numpy()] for k, g in ctrl.groupby("key")})
    ctrl.assign(anomaly=ctrl.key.str.split("__").str[0], amplitude=0.0).to_csv(out / "anomalies_adjacent_meta.csv", index=False)
    np.savez(out / "controls_len_TT.npz", abs_index=idx, len_TT=cl)
    np.savez(out / "input_spectra" / "controls.npz", abs_index=idx, spectra=cat(parts, "in_spec"))
    fits = pd.concat([pd.read_json(io.StringIO(str(p["fits_json"])), orient="records") for p in shards("ede")], ignore_index=True)
    fits.to_csv(out / "ede_fits.csv", index=False)
    print(f"anomalies_adjacent.npz: {ctrl.key.nunique()} keys, {len(ctrl)} skies; EDE refit χ² median {fits.chi2_fit.median():.1f}")


# ═══════════════════════════════ check ═══════════════════════════════

def check(args):
    """CI gates of sims_suite.md §8 that do not need a trained network (gate A/B uses a classifier)."""
    out, ok = OUT_DIR, True

    def gate(name, passed, detail=""):
        nonlocal ok
        ok &= bool(passed)
        print(f"[{'PASS' if passed else 'FAIL'}] {name}  {detail}")

    # 1. realization seeds distinct across every set; chain halves disjoint
    seeds = np.concatenate([pd.read_csv(f).real_seed.to_numpy() for f in sorted((out / "plan").glob("params_*.csv"))]
                           + [pd.read_csv(out / "plan" / "controls.csv").real_seed.to_numpy()])
    gate("realization seeds unique", len(np.unique(seeds)) == len(seeds), f"{len(seeds)} skies")
    tr = pd.read_csv(out / "plan" / "params_train.csv")
    ev = pd.concat([pd.read_csv(out / "plan" / f"params_{s}.csv") for s in ["ref", "null", "holdout", "gateA", "gateB"]])
    gate("chain halves disjoint", set(tr.chain_file).isdisjoint(set(ev.chain_file)),
         f"train files {sorted(set(tr.chain_file))}, evaluation {sorted(set(ev.chain_file))}")

    # 2. recovered C_ℓ of the null matches the input spectra within cosmic variance (χ² over 2 ≤ ℓ ≤ 191)
    nul = np.load(out / "lcdm_null.npz")
    cl_hat = np.mean([hp.alm2cl(a.astype(np.complex128)) for a in nul["alm"][:2000]], 0)
    cl_in = nul["len_TT"][:2000].mean(0)
    ell = np.arange(2, LMAX + 1)
    chi2 = np.sum(((cl_hat[ell] - cl_in[ell]) / (cl_in[ell] * np.sqrt(2 / ((2 * ell + 1) * 2000)))) ** 2)
    gate("null C_ℓ vs input", chi2 < len(ell) + 4 * np.sqrt(2 * len(ell)), f"χ² = {chi2:.0f} for {len(ell)} ℓ")

    # 3. T–φ cross-spectrum of the joint draw (regenerated for 200 null skies)
    par = pd.read_csv(out / "plan" / "params_null.csv").iloc[:200]
    r_hat, r_in = [], []
    for _, row in par.iterrows():
        spec = ls.get_spectra_T(CFG, row.to_dict())
        tlm, plm = ls.unlensed_T_phi(CFG, spec, int(row.real_seed))
        r_hat.append(hp.alm2cl(tlm, plm, lmax_out=30)[2:]); r_in.append(spec["PT"][2:31])
    r_hat, r_in = np.mean(r_hat, 0), np.mean(r_in, 0)
    rel = np.sum(r_hat * r_in) / np.sum(r_in ** 2)
    gate("T–φ cross-spectrum (ℓ 2–30)", abs(rel - 1) < 0.2, f"recovered / input = {rel:.3f}")

    # 4. stored skies are reproducible and a zero-amplitude injection equals the null twin
    for i in [0, 1, 1999]:
        row = pd.read_csv(out / "plan" / "params_null.csv").iloc[i].to_dict()
        again = _lcdm_sky(row)[0]
        twin = _w_twin((row, [{"anomaly": a, "amplitude": 0.0, "abs_index": 0}
                              for a in ["hemispherical", "low_power", "parity", "cold_spot", "modulation_L2-10"]]))
        gate(f"null {i} reproducible, zero injection = twin", np.allclose(again, nul["alm"][i], atol=1e-4) and
             all(np.allclose(t[1], nul["alm"][i], atol=1e-4) for t in twin))

    # 5. counts
    for f, n in [("lcdm_train", 20_000), ("lcdm_ref", 10_000), ("lcdm_null", 10_000), ("lcdm_holdout", 5_000),
                 ("lcdm_gateA", 5_000), ("lcdm_gateB", 5_000)]:
        gate(f"{f} count", len(np.load(out / f"{f}.npz")["abs_index"]) == n)
    print("ALL GATES PASSED" if ok else "SOME GATES FAILED")
    if not ok:
        raise SystemExit(1)


# ═══════════════════════════════ extensions ═══════════════════════════════

def extend(args):
    """Plan the EXTENSIONS not planned yet: parameters (plan/params_<name>.csv) and tasks (plan/tasks_ext.csv),
    recorded with their hashes under "extensions" in config.json. The original plan files are not touched."""
    out = OUT_DIR
    cfg = json.loads((out / "config.json").read_text())
    ext = cfg.setdefault("extensions", {})
    todo = [k for k in EXTENSIONS if k not in ext]
    todo_anom = [k for k in ANOMALY_EXTENSIONS if k not in ext]
    if not todo and not todo_anom:
        print("all extensions planned"); return
    used = np.concatenate([pd.read_csv(f).real_seed.to_numpy() for f in sorted((out / "plan").glob("params_*.csv"))]
                          + [pd.read_csv(out / "plan" / "controls.csv").real_seed.to_numpy()])
    chains = {h: load_planck_half(f) for h, f in CHAIN_HALVES.items()}
    tasks_path = out / "plan" / "tasks_ext.csv"
    tasks = pd.read_csv(tasks_path, keep_default_na=False).to_dict("records") if tasks_path.exists() else []

    def git(*a):
        repo = Path(__file__).resolve().parents[2]
        return subprocess.run(["git", "-C", str(repo), *a], capture_output=True, text=True).stdout.strip()
    for name in todo:
        first, n, half, size, view_base = EXTENSIONS[name]
        df = draw_lcdm(name, first, n, half, chains)
        assert not np.isin(df.real_seed, used).any(), f"{name}: realization seeds overlap the suite"
        f = out / "plan" / f"params_{name}.csv"
        df.to_csv(f, index=False)
        tasks += [{"kind": "lcdm", "set": name, "lo": lo, "hi": min(lo + size, n)} for lo in range(0, n, size)]
        ext[name] = {"first": first, "n": n, "chain_half": half, "view_seed_base": view_base,
                     "sha256": hashlib.sha256(f.read_bytes()).hexdigest(), "planned": time.strftime("%Y-%m-%d"),
                     "git": {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain", "--", "."))}}
    planned_idx = np.concatenate([pd.read_csv(out / "plan" / "anomalies.csv").abs_index.to_numpy()]
                                 + [pd.read_csv(f).abs_index.to_numpy() for f in (out / "plan").glob("anomalies_*.csv")])
    for name in todo_anom:
        first, spec = ANOMALY_EXTENSIONS[name]
        rows, k = [], first
        for anomaly, (amps, n_twins) in spec.items():
            for amp in amps:
                for i in range(n_twins):
                    rows.append({"file": f"anomalies_{name}", "key": f"{anomaly}__{amp:g}", "anomaly": anomaly,
                                 "amplitude": amp, "null_index": i, "abs_index": k})
                    k += 1
        df = pd.DataFrame(rows)
        assert not np.isin(df.abs_index, planned_idx).any(), f"{name}: absolute indices overlap the suite"
        f = out / "plan" / f"anomalies_{name}.csv"
        df.to_csv(f, index=False)
        n_tw = int(df.null_index.max()) + 1
        per = max(ANOM_TWINS_PER_SHARD, 400 // df.key.nunique())   # few skies per twin: larger shards
        tasks += [{"kind": "anomalies_ext", "set": name, "lo": lo, "hi": min(lo + per, n_tw)} for lo in range(0, n_tw, per)]
        ext[name] = {"kind": "anomalies", "first": first, "sets": {a: {"amplitudes": am, "n_twins": n}
                                                                   for a, (am, n) in spec.items()},
                     "combos": {a: COMBOS[a] for a in spec if a in COMBOS},
                     "sha256": hashlib.sha256(f.read_bytes()).hexdigest(), "planned": time.strftime("%Y-%m-%d"),
                     "git": {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain", "--", "."))}}
    for j, t in enumerate(tasks):
        t["task"] = j
    pd.DataFrame(tasks).to_csv(tasks_path, index=False)
    (out / "config.json").write_text(json.dumps(cfg, indent=2))
    shutil.copy(__file__, out / "generate_suite.py")
    print(f"planned {todo + todo_anom}: {len(tasks)} extension tasks -> {tasks_path}")


def merge_ext(args):
    """Shards of the extensions -> lcdm_<name>.npz, input_spectra/<name>.npy, and an anomaly-format copy
    anomalies_<name>.npz (key "<name>__0", view seeds from view_seed_base) so contrastive_eval can score it."""
    out = OUT_DIR
    cfg = json.loads((out / "config.json").read_text())
    tasks = pd.read_csv(out / "plan" / "tasks_ext.csv", keep_default_na=False)
    for name, e in cfg.get("extensions", {}).items():
        if e.get("kind") == "anomalies":
            merge_anomaly_ext(out, name, e, tasks); continue
        f = out / "plan" / f"params_{name}.csv"
        assert hashlib.sha256(f.read_bytes()).hexdigest() == e["sha256"], f"plan file {f.name} changed"
        if (out / f"lcdm_{name}.npz").exists():
            print(f"lcdm_{name}.npz exists, skipping"); continue
        paths = [out / "shards" / f"{name}_{t.task:03d}.npz" for t in tasks[tasks["set"] == name].itertuples()]
        missing = [p.name for p in paths if not p.exists()]
        if missing:
            raise SystemExit(f"{name}: {len(missing)} shards missing, e.g. {missing[:3]}")
        parts = [np.load(p) for p in paths]
        cat = lambda k: np.concatenate([q[k] for q in parts])
        par = pd.read_csv(f)
        idx = cat("abs_index")
        assert np.array_equal(idx, par.abs_index.to_numpy()), name
        alm = cat("alm")
        np.savez(out / f"lcdm_{name}.npz", alm=alm, seed=par.real_seed.to_numpy(), abs_index=idx, len_TT=cat("len_TT"))
        np.save(out / "input_spectra" / f"{name}.npy", cat("in_spec"))
        np.savez(out / f"anomalies_{name}.npz", **{f"{name}__0": alm})
        pd.DataFrame({"key": f"{name}__0", "anomaly": name, "amplitude": 0.0, "abs_index": idx,
                      "null_index": e["view_seed_base"] + np.arange(len(idx))}).to_csv(out / f"anomalies_{name}_meta.csv", index=False)
        print(f"lcdm_{name}.npz + anomalies_{name}.npz: {len(idx)} skies")


def merge_anomaly_ext(out, name, e, tasks):
    """Shards of an anomaly extension -> anomalies_<name>.npz (keys "<anomaly>__<amp>", rows sorted by null_index,
    the evaluator's order) and anomalies_<name>_meta.csv."""
    f = out / "plan" / f"anomalies_{name}.csv"
    assert hashlib.sha256(f.read_bytes()).hexdigest() == e["sha256"], f"plan file {f.name} changed"
    if (out / f"anomalies_{name}.npz").exists():
        print(f"anomalies_{name}.npz exists, skipping"); return
    paths = [out / "shards" / f"{name}_{t.task:03d}.npz" for t in tasks[tasks["set"] == name].itertuples()]
    missing = [p.name for p in paths if not p.exists()]
    if missing:
        raise SystemExit(f"{name}: {len(missing)} shards missing, e.g. {missing[:3]}")
    parts = [np.load(p) for p in paths]
    idx, alm = np.concatenate([q["abs_index"] for q in parts]), np.concatenate([q["alm"] for q in parts])
    meta = pd.concat([pd.read_json(io.StringIO(str(q["meta_json"])), orient="records") for q in parts], ignore_index=True)
    an = pd.read_csv(f).merge(meta, on="abs_index", how="left")
    pos = pd.Series(np.arange(len(idx)), index=idx)
    assert len(idx) == len(an) and pos.index.is_unique and set(idx) == set(an.abs_index), name
    an = an.sort_values(["key", "null_index"])
    np.savez(out / f"anomalies_{name}.npz", **{k: alm[pos[g.abs_index].to_numpy()] for k, g in an.groupby("key", sort=False)})
    an.drop(columns="file").to_csv(out / f"anomalies_{name}_meta.csv", index=False)
    print(f"anomalies_{name}.npz: {an.key.nunique()} keys, {len(an)} skies")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["plan", "run", "merge", "check", "extend", "merge-ext"])
    ap.add_argument("--task", type=int, help="run: row of plan/tasks.csv (the slurm array index)")
    ap.add_argument("--workers", type=int, default=40)
    ap.add_argument("--limit", type=int, default=0, help="run: testing only, truncate the shard")
    ap.add_argument("--tasks-file", default="tasks.csv", help="run: tasks.csv, or tasks_ext.csv for the extensions")
    args = ap.parse_args()
    {"plan": plan, "run": run, "merge": merge, "check": check, "extend": extend, "merge-ext": merge_ext}[args.step](args)


if __name__ == "__main__":
    main()

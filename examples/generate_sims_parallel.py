#!/usr/bin/env python3
"""Parallel runner for the CMB simulation sets defined in gen_spectra_maps.ipynb
(Planck LCDM, w0wa, oscillations, early dark energy, hemispherical asymmetry).

Generation grew from a quick visualization notebook into a large pipeline —
that doesn't belong in a live kernel anymore. Run this directly:

    python examples/generate_sims_parallel.py

Every realisation across every set is flattened into ONE task queue and
divided across N_WORKERS single-threaded processes, rather than one process
per set. With only a handful of independent sets, one-process-per-set caps
the speedup at (number of sets)x and leaves the run bounded by whichever set
is slowest. Flattening the queue means wall-clock is bounded by
total_work / N_WORKERS instead.

The notebook is still the place to load and visualise what this script
produces — its own generation cells are superseded by this script and can be
treated as reference/documentation for the per-realisation logic, which now
lives in skysimulation/pipeline.py.
"""
import os

# Must precede importing camb/numpy/healpy: CAMB uses OpenMP internally, and with
# N_WORKERS processes each otherwise free to grab every visible core, an unset
# OMP_NUM_THREADS oversubscribes the machine and can erase the whole benefit of
# parallelising by process. One thread per worker process is the right split
# once the queue already holds thousands of independent, single-realisation
# tasks — the parallelism belongs at the task level, not inside CAMB.
os.environ.setdefault('OMP_NUM_THREADS', '1')

import csv
import time
from concurrent.futures import ProcessPoolExecutor, as_completed, process as _cf_process

import numpy as np
import healpy as hp
from tqdm import tqdm
from getdist import loadMCSamples

from skysimulation import new_resolution_mask
from skysimulation.pipeline import (
    PipelineConfig, EDE_W_N, run_task,
    lhc_lcdm_6d, lhc_w0wa_8d, lhc_osc_8d, lhc_ede_9d, lhc_hemispherical_8d,
)

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION — mirrors gen_spectra_maps.ipynb Cell 2
# ═══════════════════════════════════════════════════════════════════════════════
INCLUDE_NOISE         = True
INCLUDE_GALACTIC_MASK = False
USE_LENSED_SPECTRA    = False
INCLUDE_POLARIZATION  = True   # saves T,Q,U,E,B (5 FITS/realisation)

N_POSTERIOR_PLANCK = 500    # Planck 2018 posterior-chain draws, extra on top of N_BROAD_PLANCK
N_POSTERIOR_W0WA   = 100    # DESI 2024 posterior-chain draws, extra on top of N_BROAD_W0WA
N_BROAD_PLANCK     = 5000   # LHC broad draws for the Planck LCDM set
N_BROAD_W0WA       = 1000   # LHC broad draws for the w0wa set
N_OSC              = 1000   # LHC broad draws for the oscillations set
N_EDE              = 1000   # LHC broad draws for the early dark energy set
N_HEMI             = 1000   # LHC broad draws for the hemispherical asymmetry set

NOISE_UKARCMIN_T = 40.0
NOISE_UKARCMIN_P = 40.0 * 2 ** 0.5

BASE_SEED = 314100
LHC_SEED  = 42
SEED_OFFSET       = {'planck': 0, 'w0wa': 1_000_000, 'oscillations': 2_000_000, 'ede': 3_000_000,
                     'hemispherical': 4_000_000}
NOISE_SEED_OFFSET = 7_000_000

nside     = 256
lmax_map  = 3 * nside - 1
tau_fixed = 0.0544
mnu, omk  = 0.06, 0.0
H0_osc    = 67.4   # oscillations set: H0 fixed at Planck best-fit (see notebook Set 3)
H0_hemi   = 67.4   # hemispherical set: H0 fixed too, same 'clean discrimination regime'
                    # rationale as oscillations — see lhc_hemispherical_8d docstring

N_WORKERS = 8   # queue-level parallelism; each worker pinned to 1 OMP thread (see above)

data_dir      = '/nvme/h/lchristodoulou/cmb/SkySimulation/data/Planck/'
mask_path_int = os.path.join(data_dir, 'COM_Mask_CMB-common-Mask-Int_2048_R3.00.fits')
mask_path_pol = os.path.join(data_dir, 'COM_Mask_CMB-common-Mask-Pol_2048_R3.00.fits')

_config_tag = (
    f"noise{'T' if INCLUDE_NOISE else 'F'}"
    f"_mask{'T' if INCLUDE_GALACTIC_MASK else 'F'}"
    f"_lens{'T' if USE_LENSED_SPECTRA else 'F'}"
    f"_pol{'T' if INCLUDE_POLARIZATION else 'F'}"
)
_maps_base = os.path.join('/nvme/h/lchristodoulou/data_p318/simulations/', _config_tag)
out_planck = os.path.join(_maps_base, 'planck')
out_w0wa   = os.path.join(_maps_base, 'w0wa')
out_osc    = os.path.join(_maps_base, 'oscillations')
out_ede    = os.path.join(_maps_base, 'ede')
out_hemi   = os.path.join(_maps_base, 'hemispherical')

FIELDNAMES = {
    'planck': ['i', 'component', 'seed', 'H0', 'ombh2', 'omch2', 'tau', 'lnAs', 'As', 'ns',
               'theta_star'],
    'w0wa':   ['i', 'component', 'seed', 'H0', 'ombh2', 'omch2', 'tau', 'lnAs', 'As', 'ns',
               'w0', 'wa', 'theta_star'],
    'oscillations': ['i', 'component', 'seed', 'H0', 'ombh2', 'omch2', 'tau', 'lnAs', 'As', 'ns',
                      'A_lin', 'freq', 'phase', 'theta_star'],
    'ede': ['i', 'component', 'seed', 'H0', 'ombh2', 'omch2', 'tau', 'lnAs', 'As', 'ns',
            'f_ede', 'theta_i', 'zc', 'logzc', 'w_n', 'theta_star'],
    'hemispherical': ['i', 'component', 'seed', 'H0', 'ombh2', 'omch2', 'tau', 'lnAs', 'As', 'ns',
                       'A', 'l_deg', 'b_deg', 'theta_star'],
}
META_PATHS = {
    'planck':       os.path.join(out_planck, f'metadata_planck_nside{nside}.csv'),
    'w0wa':         os.path.join(out_w0wa,   f'metadata_w0wa_nside{nside}.csv'),
    'oscillations': os.path.join(out_osc,    f'metadata_oscillations_nside{nside}.csv'),
    'ede':          os.path.join(out_ede,    f'metadata_ede_nside{nside}.csv'),
    'hemispherical': os.path.join(out_hemi,  f'metadata_hemispherical_nside{nside}.csv'),
}


def _base_cfg():
    return PipelineConfig(
        nside=nside, lmax_map=lmax_map, use_lensed_spectra=USE_LENSED_SPECTRA,
        include_noise=INCLUDE_NOISE, include_galactic_mask=INCLUDE_GALACTIC_MASK,
        include_polarization=INCLUDE_POLARIZATION,
        noise_ukarcmin_t=NOISE_UKARCMIN_T, noise_ukarcmin_p=NOISE_UKARCMIN_P,
        noise_seed_offset=NOISE_SEED_OFFSET, fwhm_rad=np.radians(5.0 / 60),
        tau_fixed=tau_fixed,
    )


def build_masks():
    mask_hard_int = new_resolution_mask(mask_path_int, target_nside=nside, threshold=0.9)
    mask_hard_pol = new_resolution_mask(mask_path_pol, target_nside=nside, threshold=0.9)
    mask_int = np.clip(hp.smoothing(mask_hard_int, fwhm=np.radians(2.0)), 0.0, 1.0)
    mask_pol = np.clip(hp.smoothing(mask_hard_pol, fwhm=np.radians(2.0)), 0.0, 1.0)
    return mask_int, mask_pol


def build_pix_vec():
    """Unit vector per pixel (3, npix), computed once and shared by every
    hemispherical-set task rather than recomputed per realisation."""
    npix = hp.nside2npix(nside)
    return np.array(hp.pix2vec(nside, np.arange(npix)))


def build_planck_tasks(cfg):
    tasks = []
    tag_base = f'planck_nside{nside}'

    chain_root = os.path.join(
        '/nvme/h/lchristodoulou/data_p318',
        'COM_CosmoParams_base-plikHM-TTTEEE-lowl-lowE_R3',
        'base/plikHM_TTTEEE_lowl_lowE',
        'base_plikHM_TTTEEE_lowl_lowE',
    )
    chains = loadMCSamples(chain_root, settings={'ignore_rows': 0.3})
    p = chains.getParams()
    rng = np.random.default_rng(BASE_SEED + 10000)
    w = chains.weights / chains.weights.sum()
    idx = rng.choice(len(chains.samples), size=N_POSTERIOR_PLANCK, replace=False, p=w)

    for i in range(N_POSTERIOR_PLANCK):
        seed_i = BASE_SEED + SEED_OFFSET['planck'] + i
        H0_i, ombh2_i, omch2_i, ns_i = (float(p.H0[idx[i]]), float(p.omegabh2[idx[i]]),
                                        float(p.omegach2[idx[i]]), float(p.ns[idx[i]]))
        tau_i  = max(float(p.tau[idx[i]]), 0.01)
        lnAs_i = float(p.logA[idx[i]])
        As_i   = np.exp(lnAs_i) * 1e-10
        tag = f'{tag_base}_{i:04d}'
        tasks.append({
            'set': 'planck', 'out_dir': out_planck, 'tag': tag, 'seed': seed_i,
            'camb_kwargs': dict(H0=H0_i, ombh2=ombh2_i, omch2=omch2_i, mnu=mnu, omk=omk,
                                 tau=tau_i, As=As_i, ns=ns_i, lmax=2507),
            'meta': {'i': i, 'component': 'posterior', 'seed': seed_i,
                     'H0': H0_i, 'ombh2': ombh2_i, 'omch2': omch2_i, 'tau': tau_i,
                     'lnAs': lnAs_i, 'As': As_i, 'ns': ns_i},
        })

    lhc1 = lhc_lcdm_6d(N_BROAD_PLANCK, cfg, seed=LHC_SEED)
    for j in range(N_BROAD_PLANCK):
        i = N_POSTERIOR_PLANCK + j
        seed_i = BASE_SEED + SEED_OFFSET['planck'] + i
        As_i, tau_i = lhc1['As'][j], lhc1['tau'][j]
        tag = f'{tag_base}_{i:04d}'
        tasks.append({
            'set': 'planck', 'out_dir': out_planck, 'tag': tag, 'seed': seed_i,
            'camb_kwargs': dict(H0=lhc1['H0'][j], ombh2=lhc1['ombh2'][j], omch2=lhc1['omch2'][j],
                                 mnu=mnu, omk=omk, tau=tau_i, As=As_i, ns=lhc1['ns'][j], lmax=2507),
            'meta': {'i': i, 'component': 'broad', 'seed': seed_i,
                     'H0': lhc1['H0'][j], 'ombh2': lhc1['ombh2'][j], 'omch2': lhc1['omch2'][j],
                     'tau': tau_i, 'lnAs': np.log(As_i * 1e10), 'As': As_i, 'ns': lhc1['ns'][j]},
        })
    return tasks


def build_w0wa_tasks(cfg):
    tasks = []
    tag_base = f'w0wa_nside{nside}'

    chain_dir = (
        '/nvme/h/lchristodoulou/data_p318/'
        'base_w_wa-desi-bao-all_planck2018-lowl-TT-clik_planck2018-lowl-EE-clik_'
        'planck-NPIPE-highl-CamSpec-TTTEEE'
    )
    chains = loadMCSamples(os.path.join(chain_dir, 'chain'), settings={'ignore_rows': 0.3})
    p = chains.getParams()
    rng = np.random.default_rng(BASE_SEED + 20000)
    w = chains.weights / chains.weights.sum()
    idx = rng.choice(len(chains.samples), size=N_POSTERIOR_W0WA, replace=False, p=w)

    for i in range(N_POSTERIOR_W0WA):
        seed_i = BASE_SEED + SEED_OFFSET['w0wa'] + i
        H0_i, ombh2_i, omch2_i, ns_i = (float(p.H0[idx[i]]), float(p.ombh2[idx[i]]),
                                        float(p.omch2[idx[i]]), float(p.ns[idx[i]]))
        tau_i  = max(float(p.tau[idx[i]]), 0.01)
        lnAs_i = float(p.logA[idx[i]])
        As_i   = np.exp(lnAs_i) * 1e-10
        w0_i, wa_i = float(p.w[idx[i]]), float(p.wa[idx[i]])
        tag = f'{tag_base}_{i:04d}'
        tasks.append({
            'set': 'w0wa', 'out_dir': out_w0wa, 'tag': tag, 'seed': seed_i,
            'camb_kwargs': dict(H0=H0_i, ombh2=ombh2_i, omch2=omch2_i, mnu=mnu, omk=omk,
                                 tau=tau_i, As=As_i, ns=ns_i, lmax=2507, w0=w0_i, wa=wa_i),
            'meta': {'i': i, 'component': 'posterior', 'seed': seed_i,
                     'H0': H0_i, 'ombh2': ombh2_i, 'omch2': omch2_i, 'tau': tau_i,
                     'lnAs': lnAs_i, 'As': As_i, 'ns': ns_i, 'w0': w0_i, 'wa': wa_i},
        })

    lhc2 = lhc_w0wa_8d(N_BROAD_W0WA, cfg, seed=LHC_SEED + 1)
    for j in range(N_BROAD_W0WA):
        i = N_POSTERIOR_W0WA + j
        seed_i = BASE_SEED + SEED_OFFSET['w0wa'] + i
        As_i, tau_i = lhc2['As'][j], lhc2['tau'][j]
        tag = f'{tag_base}_{i:04d}'
        tasks.append({
            'set': 'w0wa', 'out_dir': out_w0wa, 'tag': tag, 'seed': seed_i,
            'camb_kwargs': dict(H0=lhc2['H0'][j], ombh2=lhc2['ombh2'][j], omch2=lhc2['omch2'][j],
                                 mnu=mnu, omk=omk, tau=tau_i, As=As_i, ns=lhc2['ns'][j], lmax=2507,
                                 w0=lhc2['w0'][j], wa=lhc2['wa'][j]),
            'meta': {'i': i, 'component': 'broad', 'seed': seed_i,
                     'H0': lhc2['H0'][j], 'ombh2': lhc2['ombh2'][j], 'omch2': lhc2['omch2'][j],
                     'tau': tau_i, 'lnAs': np.log(As_i * 1e10), 'As': As_i, 'ns': lhc2['ns'][j],
                     'w0': lhc2['w0'][j], 'wa': lhc2['wa'][j]},
        })
    return tasks


def build_osc_tasks(cfg):
    tasks = []
    tag_base = f'oscillations_nside{nside}'
    lhc3 = lhc_osc_8d(N_OSC, cfg, seed=LHC_SEED + 2)

    for i in range(N_OSC):
        seed_i  = BASE_SEED + SEED_OFFSET['oscillations'] + i
        As_i, tau_i = lhc3['As'][i], lhc3['tau'][i]
        freq_i, phase_i = float(lhc3['freq'][i]), float(lhc3['phase'][i])
        tag = f'{tag_base}_{i:04d}'
        tasks.append({
            'set': 'oscillations', 'out_dir': out_osc, 'tag': tag, 'seed': seed_i,
            'camb_kwargs': dict(H0=H0_osc, ombh2=lhc3['ombh2'][i], omch2=lhc3['omch2'][i],
                                 mnu=mnu, omk=omk, tau=tau_i, As=As_i, ns=lhc3['ns'][i], lmax=2507,
                                 custom_PK=True, amp=lhc3['A_lin'][i], freq=freq_i,
                                 wid=0.04, centre=0.06, phase=phase_i),
            'meta': {'i': i, 'component': 'broad', 'seed': seed_i,
                     'H0': H0_osc, 'ombh2': lhc3['ombh2'][i], 'omch2': lhc3['omch2'][i],
                     'tau': tau_i, 'lnAs': np.log(As_i * 1e10), 'As': As_i, 'ns': lhc3['ns'][i],
                     'A_lin': lhc3['A_lin'][i], 'freq': freq_i, 'phase': phase_i},
        })
    return tasks


def build_ede_tasks(cfg):
    tasks = []
    tag_base = f'ede_nside{nside}'
    lhc4 = lhc_ede_9d(N_EDE, cfg, seed=LHC_SEED + 3)

    for i in range(N_EDE):
        seed_i    = BASE_SEED + SEED_OFFSET['ede'] + i
        As_i, tau_i = lhc4['As'][i], lhc4['tau'][i]
        f_ede_i   = float(lhc4['f_ede'][i])
        theta_i_i = float(lhc4['theta_i'][i])
        logzc_i   = float(lhc4['logzc'][i])
        zc_i      = 10 ** logzc_i
        tag = f'{tag_base}_{i:04d}'
        tasks.append({
            'set': 'ede', 'out_dir': out_ede, 'tag': tag, 'seed': seed_i,
            'camb_kwargs': dict(H0=lhc4['H0'][i], ombh2=lhc4['ombh2'][i], omch2=lhc4['omch2'][i],
                                 mnu=mnu, omk=omk, tau=tau_i, As=As_i, ns=lhc4['ns'][i], lmax=2507,
                                 ede_params=(EDE_W_N, f_ede_i, zc_i, theta_i_i)),
            'meta': {'i': i, 'component': 'broad', 'seed': seed_i,
                     'H0': lhc4['H0'][i], 'ombh2': lhc4['ombh2'][i], 'omch2': lhc4['omch2'][i],
                     'tau': tau_i, 'lnAs': np.log(As_i * 1e10), 'As': As_i, 'ns': lhc4['ns'][i],
                     'f_ede': f_ede_i, 'theta_i': theta_i_i, 'zc': zc_i,
                     'logzc': logzc_i, 'w_n': EDE_W_N},
        })
    return tasks


def build_hemi_tasks(cfg):
    tasks = []
    tag_base = f'hemispherical_nside{nside}'
    lhc5 = lhc_hemispherical_8d(N_HEMI, cfg, seed=LHC_SEED + 4)

    for i in range(N_HEMI):
        seed_i = BASE_SEED + SEED_OFFSET['hemispherical'] + i
        As_i, tau_i = lhc5['As'][i], lhc5['tau'][i]
        A_i, l_i, b_i = float(lhc5['A'][i]), float(lhc5['l_deg'][i]), float(lhc5['b_deg'][i])
        tag = f'{tag_base}_{i:04d}'
        tasks.append({
            'set': 'hemispherical', 'out_dir': out_hemi, 'tag': tag, 'seed': seed_i,
            'camb_kwargs': dict(H0=H0_hemi, ombh2=lhc5['ombh2'][i], omch2=lhc5['omch2'][i],
                                 mnu=mnu, omk=omk, tau=tau_i, As=As_i, ns=lhc5['ns'][i], lmax=2507),
            'asymmetry_params': (A_i, l_i, b_i),
            'meta': {'i': i, 'component': 'broad', 'seed': seed_i,
                     'H0': H0_hemi, 'ombh2': lhc5['ombh2'][i], 'omch2': lhc5['omch2'][i],
                     'tau': tau_i, 'lnAs': np.log(As_i * 1e10), 'As': As_i, 'ns': lhc5['ns'][i],
                     'A': A_i, 'l_deg': l_i, 'b_deg': b_i},
        })
    return tasks


def _init_worker(cfg):
    global _CFG
    _CFG = cfg
    os.environ['OMP_NUM_THREADS'] = '1'


def _worker(task):
    return task['set'], task['tag'], run_task(_CFG, task)


def main():
    for d in [out_planck, out_w0wa, out_osc, out_ede, out_hemi]:
        os.makedirs(d, exist_ok=True)

    print('Loading masks...')
    mask_int, mask_pol = build_masks()
    cfg = _base_cfg()
    cfg.mask_int, cfg.mask_pol = mask_int, mask_pol
    cfg.pix_vec = build_pix_vec()

    print('Building task queue (loads Planck & DESI chains, draws all LHCs)...')
    tasks = (build_planck_tasks(cfg) + build_w0wa_tasks(cfg)
             + build_osc_tasks(cfg) + build_ede_tasks(cfg) + build_hemi_tasks(cfg))
    print(f'{len(tasks)} realisations queued across 5 sets, {N_WORKERS} workers '
          f'(1 OMP thread each).')

    # Write CSV headers only for files that don't exist yet — resuming an
    # interrupted run must not truncate metadata already written.
    for set_name, path in META_PATHS.items():
        if not os.path.exists(path):
            with open(path, 'w', newline='') as f:
                csv.DictWriter(f, fieldnames=FIELDNAMES[set_name]).writeheader()

    n_done = n_skipped = n_failed = 0
    t_start = time.time()

    with ProcessPoolExecutor(max_workers=N_WORKERS, initializer=_init_worker,
                              initargs=(cfg,)) as ex:
        futures = {ex.submit(_worker, task): task for task in tasks}
        try:
            for fut in tqdm(as_completed(futures), total=len(futures), desc='Realisations'):
                task = futures[fut]
                try:
                    set_name, tag, meta = fut.result()
                except Exception as e:
                    n_failed += 1
                    print(f"\n[FAILED] {task['set']}/{task['tag']}: {e}")
                    continue
                if meta is None:
                    n_skipped += 1
                    continue
                with open(META_PATHS[set_name], 'a', newline='') as f:
                    csv.DictWriter(f, fieldnames=FIELDNAMES[set_name]).writerow(meta)
                n_done += 1
        except _cf_process.BrokenProcessPool:
            print('\n[FATAL] A worker process died (likely a native CAMB crash on an '
                  'extreme LHC draw, not a catchable Python exception). Completed '
                  'realisations are safely on disk. Re-run this script to resume — '
                  'finished realisations are skipped automatically (idempotent on '
                  'the T FITS file), only the remaining queue re-runs.')
            raise

    elapsed = time.time() - t_start
    print(f'\nDone in {elapsed/3600:.2f} h: {n_done} computed, {n_skipped} already '
          f'present (skipped), {n_failed} failed.')


if __name__ == '__main__':
    main()

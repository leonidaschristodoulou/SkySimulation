# Contrastive CMB-anomaly proof of concept — simulations

Simulation side of a proof of concept for detecting CMB anomalies with a contrastive
encoder trained on ΛCDM skies only. The encoder, training and scoring live in
[`cmb-latent`](../../../cmb-latent) (`contrastive_*.py`, `contrastive_results.ipynb`).

All physics is in [`skysimulation/lensed_sky.py`](../../src/skysimulation/lensed_sky.py);
the scripts here only drive it. Start with
[`lensed_sky_contrastive_views.ipynb`](lensed_sky_contrastive_views.ipynb), a
walkthrough of every step with plots.

## Order of operations

| # | Step | Frame |
|---|---|---|
| 1 | Draw unlensed T and lensing potential φ jointly (keeps the ISW–lensing correlation) | sky |
| 2 | Anomaly (test sets) or modulation T(1 + εM) (hard negatives) on the **primordial, unlensed** T | sky |
| 3 | Lens with lenspyx | sky |
| 4 | *Training only:* rescale to the fiducial C_ℓ | sky |
| 5 | Random rotation, 160′ beam, anisotropic noise, Galactic mask (last) | instrument |

Steps 1–3 run here (stored as alms); steps 4–5 run on the fly in `cmb-latent`.
Temperature only, nside 64 (lmax 191), Planck 2018 best-fit cosmology.

## Scripts (run from the `SkySimulation/` root)

| Script | Output (in `data_p318/simulations/contrastive_poc_nside64/`) |
|---|---|
| `generate_contrastive_poc_skies.py` | ΛCDM training (4000), null (500) and reference (1500) skies; anomaly test sets (5 types × 4 amplitudes × 300, paired with their null twins); hard-negative modulation terms `train_modulation.npz` |
| `calibrate_modulation_eps.py` | `modulation_eps_calibration.json`: the ε prior (ε at 50% and 90% power of a targeted matched filter, per L_dev) |
| `generate_extra_anomaly_sets.py` | `anomalies_extra.npz`: hemispherical asymmetry along the Planck axis, and combinations of several anomalies |
| `generate_fpr_check_skies.py` | `anomalies_fpr.npz`: fresh ΛCDM skies (fixed cosmology and Planck-posterior draws) for the false-alarm check |
| `fisher_anomaly_ceilings.py` | `fisher_auc.csv`: Fisher-optimal AUC ceilings per anomaly |
| `fetch_planck_T.py` | Planck PR3 temperature maps and FFP10 SMICA noise sims at nside 256 (on hold; for the realism step) |

Typical order, on the cluster (the `camb` conda env has lenspyx and ducc0):

```bash
cd SkySimulation
sbatch -p cpu -A p318 -c 32 --mem=64G -t 04:00:00 -o examples/contrastive_poc/logs/%x_%j.out \
    --wrap "OMP_NUM_THREADS=1 ~/.conda/envs/camb/bin/python -u examples/contrastive_poc/generate_contrastive_poc_skies.py --workers 32"
# then calibrate_modulation_eps.py, and optionally the extra / fpr sets and the Fisher ceilings
```

Every script skips outputs that already exist, so it can be relaunched after a crash.
Then train and score in `cmb-latent` with `submit_contrastive_poc.slurm`.

## Look-elsewhere rules

The anomaly sets are only used to measure power. Every score is calibrated on ΛCDM
null skies, and the real sky is scored once, after the pipeline is frozen.

`lensed_sky_contrastive_views_v1_tqu256.ipynb` is the first version of the walkthrough
(T/Q/U at nside 256, anomalies after lensing), kept for reference. `logs/` holds the
Slurm logs of the runs so far.

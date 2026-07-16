"""
demo_recovery_planck.py — validate latent estimates against the Fisher bound,
using the real LambdaCDM ("planck") simulation suite (component='broad', 2500 sims).

Truth params : data_p318/simulations/planck/metadata_planck_nside256.csv
Spectra      : data_p318/simulations/planck/spectra/*.npz  (not needed here --
               the Fisher bound is analytic via CAMB; only the truth columns
               from the metadata are used for the recovery check)

Note: the 'broad' truths span a much wider range than the Fisher 1-sigma
(e.g. ombh2 in [0.018, 0.028] vs sigma ~1e-4), so z_std here is an ensemble
check of your estimator's typical error against the local Fisher scale, not
a claim that the bound is exact that far from the fiducial.

TODO: point LATENT_PATH at your latent estimates and fill in
load_latent_estimates() to match their format once you have them.
"""
import numpy as np
import pandas as pd
import cmb_baseline as cb

META_CSV = "/nvme/h/lchristodoulou/data_p318/simulations/planck/metadata_planck_nside256.csv"
COMPONENT = "broad"                # which metadata subset counts as "the sims"
LATENT_PATH = None                 # TODO: set once you have latent estimates

LMIN, LMAX = 2, 767
NAMES = ['thetastar', 'ombh2', 'omch2', 'ns', 'logAse2tau']
FID = cb.FID
STEPS = {n: cb._DEFAULT_STEP[n] for n in NAMES}

# 1. Truth values in the Fisher stiff basis, for the chosen sim subset --------
df = pd.read_csv(META_CSV)
df = df[df.component == COMPONENT].reset_index(drop=True)
theta_true = np.column_stack([
    df['theta_star'].to_numpy() / 100.0,      # metadata stores 100*thetastar
    df['ombh2'].to_numpy(),
    df['omch2'].to_numpy(),
    df['ns'].to_numpy(),
    df['lnAs'].to_numpy() - 2 * df['tau'].to_numpy(),
])
print(f"Loaded {len(df)} LambdaCDM ('{COMPONENT}') truths from {META_CSV}")

# 2. Fisher bound at the fiducial cosmology -----------------------------------
names, F = cb.fisher(FID, STEPS, LMIN, LMAX, lensed=True, pol=False)
con = cb.constraints(names, F)
print("\nFisher 1-sigma bound (T-only, CV-limited, lmax=767):")
for n in names:
    print(f"   {n:11s} {con['sigma'][n]:.4g}")

# 3. Latent estimates ----------------------------------------------------------
def load_latent_estimates(path, names):
    """TODO: load your network's latent estimates and return an (N, len(names))
    array in the SAME order as `names`, aligned row-for-row with `theta_true`
    (i.e. filtered/sorted the same way as the metadata, component == COMPONENT).
    """
    raise NotImplementedError(
        "Point LATENT_PATH at your latent estimates and implement load_latent_estimates()."
    )

if LATENT_PATH is None:
    print(f"\nNo latent estimates configured yet (LATENT_PATH is None) -- "
          f"set it and implement load_latent_estimates() to run the recovery check.")
else:
    theta_est = load_latent_estimates(LATENT_PATH, NAMES)
    rep = cb.recovery_report(theta_true, theta_est, con['sigma'], NAMES)
    print("\nRecovery vs Fisher bound (z_std ~1 good | >1 leaves info | <1 leakage):")
    for n in NAMES:
        r = rep[n]
        print(f"   {n:11s} bias={r['bias']:+.3g}  rms={r['rms']:.3g}  "
              f"rms/sigma={r['rms_over_fisher']:.2f}  z_std={r['z_std']:.2f}")

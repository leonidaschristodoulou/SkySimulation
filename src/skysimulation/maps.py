import numpy as np
import healpy as hp

def generate_cmb_temperature_map(cmb_cls, nside, seed0):
    rng = np.random.default_rng(seed0)
    # healpy uses global RNG internally in some places; setting seed via np.random.seed is sometimes needed.
    np.random.seed(seed0)
    cmb_map = hp.synfast(cmb_cls, nside=nside, new=True, sigma=None, pol=False)
    hp.mollview(cmb_map, title=f"CMB Map, Nside={nside}", unit="K")
    return cmb_map

def save_cmb_temperature_map(cmb_cls, nside, n_map, seed0, output_dir="./", file_prefix="cmb_map", custom_Pk=False):
    np.random.seed(seed0)
    cmb_temp_map = hp.synfast(cmb_cls, nside=nside, new=True, pol=False, sigma=None)
    suffix = "_feature" if custom_Pk else ""
    output_file = f"{output_dir}{file_prefix}{suffix}_{n_map}.fits"
    hp.write_map(output_file, cmb_temp_map)

def generate_cmb_polarization_maps(
    cl_tt, cl_ee, cl_bb, cl_te, cl_eb, cl_tb,
    nside, seed0,
    custom_smooth=False, fwhm_arcmin=5.0,
):
    np.random.seed(seed0)
    T_map, Q_map, U_map = hp.synfast([cl_tt, cl_ee, cl_bb, cl_te, cl_eb, cl_tb], nside=nside, new=True, pol=True)

    if custom_smooth:
        fwhm_rad = np.radians(fwhm_arcmin / 60)
        Q_smooth = hp.smoothing(Q_map, fwhm=fwhm_rad)
        U_smooth = hp.smoothing(U_map, fwhm=fwhm_rad)
    else:
        Q_smooth, U_smooth = Q_map, U_map

    hp.mollview(Q_smooth, title="CMB Polarization (Q component)", cmap="RdBu", unit="μK")
    hp.mollview(U_smooth, title="CMB Polarization (U component)", cmap="RdBu", unit="μK")
    return Q_smooth, U_smooth

# def save_cmb_polarization_maps(
#     cl_tt, cl_ee, cl_bb, cl_te, cl_eb, cl_tb,
#     nside, n_map, seed0,
#     output_dir="./", file_prefix="cmb_pol_map",
#     custom_smooth=False, fwhm_arcmin=0.0,
#     custom_Pk=False,
# ):
#     np.random.seed(seed0)
#     T_map, Q_map, U_map = hp.synfast([cl_tt, cl_ee, cl_bb, cl_te, cl_eb, cl_tb], nside=nside, new=True, pol=True)

#     if custom_smooth:
#         fwhm_rad = np.radians(fwhm_arcmin / 60)
#         T_map = hp.smoothing(T_map, fwhm=fwhm_rad)
#         Q_map = hp.smoothing(Q_map, fwhm=fwhm_rad)
#         U_map = hp.smoothing(U_map, fwhm=fwhm_rad)

#     suffix = "_feature" if custom_Pk else ""
#     output_file0 = f"{output_dir}{file_prefix}_T{suffix}_{n_map}.fits"
#     output_file1 = f"{output_dir}{file_prefix}_Q{suffix}_{n_map}.fits"
#     output_file2 = f"{output_dir}{file_prefix}_U{suffix}_{n_map}.fits"

#     hp.write_map(output_file0, T_map)
#     hp.write_map(output_file1, Q_map)
#     hp.write_map(output_file2, U_map)

def apply_galactic_mask(T_map, Q_map, U_map, nside, mask_path=None, lat_cut_deg=None):
    """
    Applies a galactic mask to Temperature and Polarization maps.
    
    Parameters:
    - T_map, Q_map, U_map: The input HEALPix maps.
    - nside: The resolution of the maps.
    - mask_path: Path to a .fits file containing the mask (0s and 1s).
    - lat_cut_deg: If no file is provided, cuts a strip of +/- lat_cut_deg around the equator.
    
    Returns:
    - Masked T, Q, U maps.
    """
    
    # 1. Load or Generate the Mask
    if mask_path:
        # Load mask and force it to match the simulation nside
        mask = hp.read_map(mask_path)
        if hp.npix2nside(len(mask)) != nside:
            mask = hp.ud_grade(mask, nside)
    elif lat_cut_deg:
        # create a simple binary strip mask
        # 0 at equator (galactic plane), 1 elsewhere
        npix = hp.nside2npix(nside)
        theta, phi = hp.pix2ang(nside, np.arange(npix))
        # theta is 0 at North Pole, pi at South Pole. Equator is pi/2.
        # lat = 90 - rad2deg(theta)
        lat_deg = 90.0 - np.degrees(theta)
        
        mask = np.ones(npix)
        # Mask out the band within +/- lat_cut_deg
        mask[np.abs(lat_deg) < lat_cut_deg] = 0.0
    else:
        raise ValueError("Must provide either mask_path or lat_cut_deg")

    # 2. Apply Apodization (Optional but recommended for C_ell, less critical for CNN but good practice)
    # If using a binary mask, you might want to smooth the edges slightly to avoid ringing,
    # but for a CNN, a hard binary cut (multiplying by 0) is often fine.
    
    # 3. Apply the mask to maps
    # Note: For polarization, we simply multiply Q and U by the intensity mask.
    T_masked = T_map * mask
    Q_masked = Q_map * mask
    U_masked = U_map * mask
    
    return T_masked, Q_masked, U_masked

#Downgrade the resolution of planck mask
def new_resolution_mask(mask_path, target_nside=256, threshold=0.9):
    """
    Reads a high-res Planck mask, downgrades it, and re-binarizes it.
    """
    print(f"Loading mask from {mask_path}...")
    
    # 1. Read the high-res mask (Planck is usually nside=2048)
    # field=0 is usually the intensity mask. If using a Pol mask, check field indices.
    mask_high_res = hp.read_map(mask_path) 
    
    # 2. Downgrade to your simulation nside
    # This creates values between 0.0 and 1.0 at the edges
    mask_low_res = hp.ud_grade(mask_high_res, nside_out=target_nside)
    
    # 3. Apply Threshold (Re-binarize)
    # threshold=0.9 means: "If the pixel is not at least 90% clean, throw it away."
    # This expands the mask slightly, which is safer for removing contamination.
    binary_mask = np.zeros_like(mask_low_res)
    binary_mask[mask_low_res > threshold] = 1.0
    
    # 4. (Optional) Check how much sky is left
    fsky = np.sum(binary_mask) / len(binary_mask)
    print(f"Mask downgraded to nside={target_nside}. Sky fraction (f_sky): {fsky:.2%}")
    
    return binary_mask

def save_cmb_polarization_maps(
    cl_tt, cl_ee, cl_bb, cl_te, cl_eb, cl_tb,
    nside, n_map, seed0,
    output_dir="./", file_prefix="cmb_pol_map",
    custom_smooth=False, fwhm_arcmin=0.0,
    custom_Pk=False,
    apply_mask=False,       
    mask_path=None          #Path to mask file
):
    np.random.seed(seed0)
    # 1. Generate Maps
    # Note: synfast expects inputs in the order: TT, EE, BB, TE, EB, TB
    T_map, Q_map, U_map = hp.synfast(
        [cl_tt, cl_ee, cl_bb, cl_te, cl_eb, cl_tb], 
        nside=nside, new=True, pol=True
    )

    # 2. Smooth (Beam convolution)
    if custom_smooth:
        np.random.seed(seed0)
        fwhm_rad = np.radians(fwhm_arcmin / 60)
        T_map = hp.smoothing(T_map, fwhm=fwhm_rad)
        Q_map = hp.smoothing(Q_map, fwhm=fwhm_rad)
        U_map = hp.smoothing(U_map, fwhm=fwhm_rad)

    # 3. Apply Mask (NEW STEP)
    # We apply the mask *after* smoothing to simulate real observation 
    # (the sky is smooth, then we observe it and cut the galaxy)
    if apply_mask:
        # You can use a specific file, or a simple cut for testing (e.g. 20 degrees)
        T_map, Q_map, U_map = apply_galactic_mask(
            T_map, Q_map, U_map, nside, 
            mask_path=mask_path, lat_cut_deg=20
        )

    # 4. Save
    suffix = "_feature" if custom_Pk else ""
    output_file0 = f"{output_dir}{file_prefix}_T{suffix}_{n_map}.fits"
    output_file1 = f"{output_dir}{file_prefix}_Q{suffix}_{n_map}.fits"
    output_file2 = f"{output_dir}{file_prefix}_U{suffix}_{n_map}.fits"

    hp.write_map(output_file0, T_map, overwrite=True)
    hp.write_map(output_file1, Q_map, overwrite=True)
    hp.write_map(output_file2, U_map, overwrite=True)

def deconvolve_gaussian_beam(ells, Cl_smooth_map, fwhm_arcmin):
    fwhm_rad = np.radians(fwhm_arcmin / 60.0)
    sigma = fwhm_rad / np.sqrt(8.0 * np.log(2.0))
    B_ell = np.exp(-0.5 * ells * (ells + 1) * sigma**2)
    return Cl_smooth_map / (B_ell**2)

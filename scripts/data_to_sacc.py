r"""
.. module:: fgmarge.data_to_sacc

:Synopsis: Compress the data into a sacc file.
:Authors: Hidde Jense.
"""
import numpy as np
import sacc
from soliket.mflike.ForegroundMarginalizer import ForegroundMarginalizer

"""
You can modify the code below here!
"""

outdir = "fgmarge"

fgmarge = ForegroundMarginalizer({
    # Put your nondefault settings in here.
    "input_file": "LAT_simu_sacc_00000.fits",
    "cov_Bbl_file": "data_sacc_w_covar_and_Bbl.fits",
    "data_folder": "ForegroundMarginalizer/v0.8",
    "lmax_extract": {
        "tt": 7000,
        "te": 7000,
        "ee": 7000
    },
    "foregrounds": None,
    "theoryforge": None,
    "bandpass": None
})

fixed_params = {
    "T_d": 9.6,
    "cal_LAT_93": 1,
    "cal_LAT_145": 1,
    "cal_LAT_225": 1,
    "calT_LAT_93": 1,
    "calT_LAT_145": 1,
    "calT_LAT_225": 1,
    "calE_LAT_93": 1,
    "calE_LAT_145": 1,
    "calE_LAT_225": 1,
    "alpha_LAT_93": 0,
    "alpha_LAT_145": 0,
    "alpha_LAT_225": 0,
    "bandint_shift_LAT_93": 0,
    "bandint_shift_LAT_145": 0,
    "bandint_shift_LAT_225": 0,
    "calG_all": 1
}

fgmarge.make_mapping_matrix(**fixed_params)

cmb_ls = np.loadtxt(f"{outdir}/leff.txt")
cmb_samples = np.loadtxt(f"{outdir}/extracted.txt")

n = int(0.3 * cmb_samples.shape[0])

cmb_mean = np.nanmean(cmb_samples[n:, :], axis=0)
cmb_cov = np.cov(cmb_samples[n:, :].T)

s = sacc.Sacc()

s.add_tracer("misc", "LAT_cmb_s0")
s.add_tracer("misc", "LAT_cmb_s2")

datatypes = {
    "tt": "cl_00",
    "te": "cl_0e",
    "ee": "cl_ee",
}

tracers = {
    "tt": ("LAT_cmb_s0", "LAT_cmb_s0"),
    "te": ("LAT_cmb_s0", "LAT_cmb_s2"),
    "ee": ("LAT_cmb_s2", "LAT_cmb_s2"),
}

indices = []

for m in fgmarge.spec_meta:
    if m["t1"] == "LAT_145" and m["t2"] == "LAT_145":
        p = m["pol"]
        bin0 = fgmarge.extract_zero[p]
        bin1 = bin0 + fgmarge.extract_bins[p]
        win = m["bpw"]

        win = sacc.windows.BandpowerWindow(2 + np.arange(fgmarge.lmax_extract[p] + 1),
                                           win.weight[:fgmarge.lmax_extract[p] + 1,
                                                      :fgmarge.extract_bins[p]])

        t1, t2 = tracers[p]
        s.add_ell_cl(datatypes.get(p), t1, t2, cmb_ls[bin0:bin1],
                     cmb_mean[bin0:bin1], window=win)

        _, _, ind = s.get_ell_cl(datatypes.get(p), t1, t2, return_ind=True)
        indices.append(ind)

sorted_indices = np.concatenate(indices)

s.add_covariance(cmb_cov)

s.save_fits(f"{outdir}/so_cmb_sacc_00000.fits", overwrite=True)

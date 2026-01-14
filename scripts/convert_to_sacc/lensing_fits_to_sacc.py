import os

import numpy as np
import sacc
from astropy.io import fits
from cobaya.model import get_model
from cobaya.tools import resolve_packages_path

# Set custom_packages_path to override the default cobaya packages path
custom_packages_path = None
packages_path = custom_packages_path or resolve_packages_path()

fname = os.path.join(
    packages_path, "data", "LensingLikelihood", "clkk_reconstruction_sim.fits"
)
hdul = fits.open(fname)

data_hdu = hdul[8]
data = data_hdu.data

win_bp_hdu = hdul["window:Bandpower"]
win_bp_tab = win_bp_hdu.data

ells_large = np.array([elem[0] for elem in win_bp_tab])
windows = np.array([elem[1] for elem in win_bp_tab]).T
beam = np.ones_like(data["ell"])

s = sacc.Sacc()
s.metadata['info'] = 'CMB lensing power spectra from reconstruction simulations'

s.add_tracer(
    tracer_type="Map",
    name="ck",
    quantity="cmb_convergence",
    spin=0,
    map_unit='uK_CMB',
    ell=data["ell"],
    beam=beam,
)

wins = sacc.BandpowerWindow(ells_large, windows.T)

s.add_ell_cl(
    "cl_00",  # Data type
    "ck",  # 1st tracer's name
    "ck",  # 2nd tracer's name
    data["ell"],  # Effective multipole
    data["value"],  # Power spectrum values
    window=wins,  # Bandpower windows
)

cov_hdu = hdul[9]
cov = sacc.covariance.FullCovariance.from_hdu(cov_hdu)
s.add_covariance(cov, overwrite=False)

filename = "lensing.sacc.fits"
s.save_fits(
    os.path.join(packages_path, "data/LensingLikelihood", filename), overwrite=True
)


# FIDUCIAL SACC

fid_s = sacc.Sacc()
fid_s.metadata['info'] = 'Fiducial CMB lensing power spectra computed with CAMB'

fid_s.add_tracer(
    tracer_type="Map",
    name="ct",
    quantity="cmb_temperature",
    spin=0,
    map_unit='uK_CMB',
    ell=data["ell"],
    beam=beam,
)

fid_s.add_tracer(
    tracer_type="Map",
    name="ce",
    quantity="cmb_polarization",
    spin=2,
    map_unit='uK_CMB',
    ell=data["ell"],
    beam=beam,
)

fid_s.add_tracer(
    tracer_type="Map",
    name="cb",
    quantity="cmb_polarization",
    spin=2,
    map_unit='uK_CMB',
    ell=data["ell"],
    beam=beam,
)

fid_s.add_tracer(
    tracer_type="Map",
    name="cp",
    quantity="cmb_lens_potential",
    spin=0,
    map_unit='uK_CMB',
    ell=data["ell"],
    beam=beam,
)

fid_s.add_tracer(
    tracer_type="Map",
    name="ck",
    quantity="cmb_convergence",
    spin=0,
    map_unit='uK_CMB',
    ell=data["ell"],
    beam=beam,
)

fiducial_params = {
    "ombh2": 0.02219218,
    "omch2": 0.1203058,
    "H0": 67.02393,
    "tau": 0.6574325e-01,
    "nnu": 3.046,
    "As": 2.15086031154146e-9,
    "ns": 0.9625356e00,
}

theory_lmax = 10000

info_fiducial = {
    "params": fiducial_params,
    "likelihood": {"soliket.utils.OneWithCls": {"lmax": theory_lmax}},
    "theory": {"camb": {"extra_args": {"kmax": 0.9}}},
    # "modules": modules_path,
}
model_fiducial = get_model(info_fiducial)
model_fiducial.logposterior({})
Cls = model_fiducial.provider.get_Cl(ell_factor=False)

lmax = 3000
ls = np.arange(0, lmax, dtype=np.longlong)

unbinned_wins = sacc.BandpowerWindow(ls, np.eye(len(ls)))

fid_s.add_ell_cl(
    "cl_00",  # Data type
    "ct",  # 1st tracer's name
    "ct",  # 2nd tracer's name
    ls,  # Effective multipole
    Cls["tt"][:lmax],  # Power spectrum values
    window=unbinned_wins,  # Bandpower windows
)

fid_s.add_ell_cl(
    "cl_0e",  # Data type
    "ct",  # 1st tracer's name
    "ce",  # 2nd tracer's name
    ls,  # Effective multipole
    Cls["te"][:lmax],  # Power spectrum values
    window=unbinned_wins,  # Bandpower windows
)

fid_s.add_ell_cl(
    "cl_0b",  # Data type
    "ct",  # 1st tracer's name
    "cb",  # 2nd tracer's name
    ls,  # Effective multipole
    np.zeros_like(Cls["te"][:lmax]),  # Power spectrum values
    window=unbinned_wins,  # Bandpower windows
)

fid_s.add_ell_cl(
    "cl_ee",  # Data type
    "ce",  # 1st tracer's name
    "ce",  # 2nd tracer's name
    ls,  # Effective multipole
    Cls["ee"][:lmax],  # Power spectrum values
    window=unbinned_wins,  # Bandpower windows
)

fid_s.add_ell_cl(
    "cl_bb",  # Data type
    "cb",  # 1st tracer's name
    "cb",  # 2nd tracer's name
    ls,  # Effective multipole
    Cls["bb"][:lmax],  # Power spectrum values
    window=unbinned_wins,  # Bandpower windows
)

fid_s.add_ell_cl(
    "cl_eb",  # Data type
    "ce",  # 1st tracer's name
    "cb",  # 2nd tracer's name
    ls,  # Effective multipole
    np.zeros_like(Cls["ee"][:lmax]),  # Power spectrum values
    window=unbinned_wins,  # Bandpower windows
)

fid_s.add_ell_cl(
    "cl_00",  # Data type
    "cp",  # 1st tracer's name
    "cp",  # 2nd tracer's name
    ls,  # Effective multipole
    Cls["pp"][:lmax],  # Power spectrum values
    window=unbinned_wins,  # Bandpower windows
)

fid_s.add_ell_cl(
    "cl_00",  # Data type
    "ck",  # 1st tracer's name
    "ck",  # 2nd tracer's name
    ls,  # Effective multipole
    Cls["pp"][:lmax] * (ls * (ls + 1)) ** 2 * 0.25,  # Power spectrum values
    window=unbinned_wins,  # Bandpower windows
)

filename = "fiducial_lensing.sacc.fits"
fid_s.save_fits(
    os.path.join(packages_path, "data/LensingLikelihood", filename), overwrite=True
)


# CORRECTION SACC

corr_s = sacc.Sacc()
corr_s.metadata['info'] = 'CMB lensing reconstruction noise corrections'

corr_s.add_tracer(
    tracer_type="Map",
    name="ct",
    quantity="cmb_temperature",
    spin=0,
    map_unit='uK_CMB',
    ell=data["ell"],
    beam=beam,
)

corr_s.add_tracer(
    tracer_type="Map",
    name="ce",
    quantity="cmb_polarization",
    spin=2,
    map_unit='uK_CMB',
    ell=data["ell"],
    beam=beam,
)

corr_s.add_tracer(
    tracer_type="Map",
    name="cb",
    quantity="cmb_polarization",
    spin=2,
    map_unit='uK_CMB',
    ell=data["ell"],
    beam=beam,
)

corr_s.add_tracer(
    tracer_type="Map",
    name="cp",
    quantity="cmb_lens_potential",
    spin=0,
    map_unit='uK_CMB',
    ell=data["ell"],
    beam=beam,
)

corr_s.add_tracer(
    tracer_type="Map",
    name="ck",
    quantity="cmb_convergence",
    spin=0,
    map_unit='uK_CMB',
    ell=data["ell"],
    beam=beam,
)

corr_s.add_tracer(
    tracer_type="Map",
    name="n0",
    quantity="cmb_convergence",
    spin=0,
    map_unit='uK_CMB',
    ell=data["ell"],
    beam=beam,
)

data_folder = packages_path

N0cltt = np.loadtxt(
    os.path.join(packages_path, "data/LensingLikelihood", "n0mvdcltt1.txt")
).T
N0clte = np.loadtxt(
    os.path.join(packages_path, "data/LensingLikelihood", "n0mvdclte1.txt")
).T
N0clee = np.loadtxt(
    os.path.join(packages_path, "data/LensingLikelihood", "n0mvdclee1.txt")
).T
N0clbb = np.loadtxt(
    os.path.join(packages_path, "data/LensingLikelihood", "n0mvdclbb1.txt")
).T
N1clpp = np.loadtxt(
    os.path.join(packages_path, "data/LensingLikelihood", "n1mvdclkk1.txt")
).T
N1cltt = np.loadtxt(
    os.path.join(packages_path, "data/LensingLikelihood", "n1mvdcltte1.txt")
).T
N1clte = np.loadtxt(
    os.path.join(packages_path, "data/LensingLikelihood", "n1mvdcltee1.txt")
).T
N1clee = np.loadtxt(
    os.path.join(packages_path, "data/LensingLikelihood", "n1mvdcleee1.txt")
).T
N1clbb = np.loadtxt(
    os.path.join(packages_path, "data/LensingLikelihood", "n1mvdclbbe1.txt")
).T
n0 = np.loadtxt(os.path.join(packages_path, "data/LensingLikelihood", "n0mv.txt"))
n0 = np.tile(n0, (len(ls), 1))

corr_s.add_ell_cl(
    "N0_00",  # Data type
    "ct",  # 1st tracer's name
    "ct",  # 2nd tracer's name
    ls,  # Effective multipole
    N0cltt,  # Power spectrum values
    window=unbinned_wins,  # Bandpower windows
)

corr_s.add_ell_cl(
    "N0_ee",  # Data type
    "ce",  # 1st tracer's name
    "ce",  # 2nd tracer's name
    ls,  # Effective multipole
    N0clee,  # Power spectrum values
    window=unbinned_wins,  # Bandpower windows
)

corr_s.add_ell_cl(
    "N0_bb",  # Data type
    "cb",  # 1st tracer's name
    "cb",  # 2nd tracer's name
    ls,  # Effective multipole
    N0clbb,  # Power spectrum values
    window=unbinned_wins,  # Bandpower windows
)

corr_s.add_ell_cl(
    "N0_0e",  # Data type
    "ct",  # 1st tracer's name
    "ce",  # 2nd tracer's name
    ls,  # Effective multipole
    N0clte,  # Power spectrum values
    window=unbinned_wins,  # Bandpower windows
)

corr_s.add_ell_cl(
    "N1_00",  # Data type
    "ct",  # 1st tracer's name
    "ct",  # 2nd tracer's name
    ls,  # Effective multipole
    N1cltt,  # Power spectrum values
    window=unbinned_wins,  # Bandpower windows
)

corr_s.add_ell_cl(
    "N1_ee",  # Data type
    "ce",  # 1st tracer's name
    "ce",  # 2nd tracer's name
    ls,  # Effective multipole
    N1clee,  # Power spectrum values
    window=unbinned_wins,  # Bandpower windows
)

corr_s.add_ell_cl(
    "N1_bb",  # Data type
    "cb",  # 1st tracer's name
    "cb",  # 2nd tracer's name
    ls,  # Effective multipole
    N1clbb,  # Power spectrum values
    window=unbinned_wins,  # Bandpower windows
)

corr_s.add_ell_cl(
    "N1_0e",  # Data type
    "ct",  # 1st tracer's name
    "ce",  # 2nd tracer's name
    ls,  # Effective multipole
    N1clte,  # Power spectrum values
    window=unbinned_wins,  # Bandpower windows
)

corr_s.add_ell_cl(
    "N1_00",  # Data type
    "cp",  # 1st tracer's name
    "cp",  # 2nd tracer's name
    ls,  # Effective multipole
    N1clpp,  # Power spectrum values
    window=unbinned_wins,  # Bandpower windows
)

corr_s.add_ell_cl(
    "N0_00",  # Data type
    "n0",  # 1st tracer's name
    "n0",  # 2nd tracer's name
    ls,  # Effective multipole
    np.array(n0),  # Power spectrum values
    window=unbinned_wins,  # Bandpower windows
)

filename = "corrections_lensing.sacc.fits"
corr_s.save_fits(
    os.path.join(packages_path, "data/LensingLikelihood", filename), overwrite=True
)

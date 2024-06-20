import cobaya
import numpy as np
from astropy.io import fits

from cobaya.yaml import yaml_load_file
from cobaya.model import get_model

# read in the cobaya info
info = yaml_load_file('run_lensing.yaml')

model = get_model(info)
model.loglikes({})

fname = "../../clkk_reconstruction_sim.fits"
hdul = fits.open(fname)
hdul[8].data["value"][:] = model.likelihood["soliket.LensingLikelihood"]._get_theory()

ndir = "data/clkk_smooth.fits"
hdul.writeto(ndir, overwrite=True)
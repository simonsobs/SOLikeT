import pytest
import numpy as np

from cobaya.model import get_model
from cobaya.run import run

info = {"params": {
                   "H0": 70.,
                   "ombh2": 0.0245,
                   "omch2": 0.1225,
                   "ns": 0.96,
                   "As": 2.2e-9,
                   "tau": 0.05
                   },
        "likelihood": {"one": None},
        "sampler": {"evaluate": None},
        "debug": True
       }

def test_halomodel_import():
    from soliket.halo_model import HaloModel

def test_pyhalomodel_import():
    from soliket.halo_model import HaloModel_pyhm

def test_pyhalomodel_model():
    
    from soliket.halo_model import HaloModel_pyhm

    info["theory"] = {
                        "camb": None,
                        "halo_model" : {"external": HaloModel_pyhm}
    }

    model = get_model(info)  # noqa F841


def test_pyhalomodel_compute_mm_grid():

    from soliket.halo_model import HaloModel_pyhm

    info["theory"] = {
                        "camb": None,
                        "halo_model" : {"external": HaloModel_pyhm}
    }

    model = get_model(info)  # noqa F841
    model.add_requirements({"Pk_grid": {"z": 0., "k_max": 10.,
                                "nonlinear": False,
                                "vars_pairs": ('delta_tot', 'delta_tot')
                                },
                     "Pk_mm_grid": None,
                    })

    model.logposterior(info['params'])  # force computation of model

    lhood = model.likelihood['one']

    Pk_mm_hm = lhood.provider.get_Pk_mm_grid()

    assert np.isfinite(Pk_mm_hm)
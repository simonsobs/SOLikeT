import numpy as np

from cobaya.model import get_model

info = {"params": {
                   "H0": 70.,
                   "ombh2": 0.05 * 0.7 * 0.7,
                   "omch2": 0.25 * 0.7 * 0.7,
                   "ns": 0.96,
                   "As": 2.e-9,
                   "mnu": 0.0,
                   "tau": 0.05
                   },
        "likelihood": {"one": None},
        "sampler": {"evaluate": None},
        "debug": True
       }


def test_halomodel_import():
    from soliket.halo_model import HaloModel # noqa F401


def test_pyhalomodel_import():
    from soliket.halo_model import HaloModel_pyhm # noqa F401


def test_pyhalomodel_model():
    
    from soliket.halo_model import HaloModel_pyhm

    info["theory"] = {
                        "camb": {'stop_at_error': True},
                        "halo_model": {"external": HaloModel_pyhm,
                                        "stop_at_error": True}
    }

    model = get_model(info)  # noqa F841


def test_pyhalomodel_compute_mm_grid():

    from soliket.halo_model import HaloModel_pyhm

    info["theory"] = {
                        "camb": None,
                        "halo_model": {"external": HaloModel_pyhm}
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
    k, z, Pk_mm_lin = lhood.provider.get_Pk_grid(var_pair=('delta_tot', 'delta_tot'),
                                                 nonlinear=False)

    assert np.all(np.isfinite(Pk_mm_hm))
    # this number derives from the Pk[m-m]
    # calculated in demo-basic.ipynb of the pyhalomodel repo
    assert np.isclose(Pk_mm_hm[0, k > 1.e-4][0], 3273.591586683341, rtol=1.e-3)
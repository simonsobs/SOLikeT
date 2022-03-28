import numpy as np
from soliket.szlike import SZLikelihood
from cobaya.model import get_model


def test_szlike():
    cosmo_params = {
        "h": 0.673,
        "n_s": 0.964
    }

    #come back to all of these parameters
    info = {"params": {#"omch2": cosmo_params['Omega_c'] * cosmo_params['h'] ** 2.,
                       "omch2": 0.1200,
                       "ombh2": 0.0223,
                       "H0": cosmo_params['h'] * 100,
                       "ns": cosmo_params['n_s'],
                       "As": 2.1e-9,
                       "tau": 0.054,
                       "A_IA": 0.0},
            "likelihood": {"SZLikelihood":
                            {"external": SZlikelihood,
                             "beam_file": "soliket/tests/data/beam_f150_daynight.txt" ###come back to this
                             }
                          },
            "theory": {
                "camb": None, #idk what these are for??
            },
            "debug": False, "stop_at_error": True}

    model = get_model(info)
    loglikes, derived = model.loglikes()

    lhood = model.likelihood['SZLikelihood']

    ksz_data, tsz_data = np.loadtxt('soliket/tests/data/TNG_data.txt', usecols=(1,2),unpack=True) #should this be in tests for folder for szlike?

    lhood.ksz_data = ksz_data
    lhood.tsz_data = tsz_data

    #not sure what these are for or how to finish the test
    ell_obs_kappagamma = ell_load[n_ell:]
    # ell_obs_gammagamma = ell_load[:n_ell]

    cl_theory = lhood._get_theory(**info["params"])
    cl_kappagamma = cl_theory[n_ell:]

    assert np.allclose(ell_obs_kappagamma * cl_kappagamma * 1.e6,
                       ellcl_paper,
                       atol=.2, rtol=0.)

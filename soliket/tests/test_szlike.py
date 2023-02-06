import numpy as np
import os
from soliket.szlike import KSZLikelihood, TSZLikelihood
from cobaya.model import get_model


sz_data_file = "soliket/szlike/gnfw_test_projections.txt"
cov_tsz_file = "soliket/szlike/cov_diskring_tsz_varweight_bootstrap.txt"
cov_ksz_file = "soliket/szlike/cov_diskring_ksz_varweight_bootstrap.txt"
twohalo_term = "soliket/szlike/twohalo_cmass_average.txt"
beam_file = "soliket/szlike/beam_f150_daynight.txt"
beam_response = "soliket/szlike/act_planck_s08_s18_cmb_f150_daynight_response_tsz.txt"
# thta_arc,ksz_gnfw,tsz_gnfw = np.loadtxt(sz_data_file,usecols=(0,1,2),unpack=True) #muK*sqarcmin


def test_ksz(request):
    info = {
        "params": {
            "gnfw_rho0": 3.186441,
            "gnfw_bt_ksz": 3.45493977635,
            "gnfw_A2h_ksz": 1.0,
        },
        "likelihood": {
            "KSZLikelihood": {
                "external": KSZLikelihood,
                "sz_data_file": os.path.join(request.config.rootdir, sz_data_file),
                "cov_ksz_file": os.path.join(request.config.rootdir, cov_ksz_file),
                "twohalo_term": os.path.join(request.config.rootdir, twohalo_term),
                "beam_file": os.path.join(request.config.rootdir, beam_file),
            }
        },
    }

    thta_arc, ksz_gnfw, tsz_gnfw = np.loadtxt(
        os.path.join(request.config.rootdir, sz_data_file),
        usecols=(0, 1, 2),
        unpack=True,
    )

    model = get_model(info)
    loglikes, derived = model.loglikes()

    lhood = model.likelihood["KSZLikelihood"]

    ksz_theory = lhood._get_theory(**info["params"])
    ksz_theory *= 3282.8 * 60.0**2
    print("ksz_theory", ksz_theory)
    print("ksz_gnfw", ksz_gnfw)

    assert np.allclose(ksz_gnfw, ksz_theory, atol=1.0e-1, rtol=0.1)


"""
def test_tsz(request):
    info = {"params": {"gnfw_P0":9.10766709141,
                       "gnfw_bt_tsz":4.76625102519,
                       "gnfw_A2h_tsz":1.0},
            "likelihood": {"TSZLikelihood":
                            {"external": TSZLikelihood,
                             "cov_tsz_file":os.path.join(request.config.rootdir, cov_tsz_file),
                             "beam_file":os.path.join(request.config.rootdir, beam_file),
                             "beam_response":os.path.join(request.config.rootdir, beam_response)
                             }
                          }}

    model = get_model(info)
    loglikes, derived = model.loglikes()

    lhood = model.likelihood['TSZLikelihood']

    tsz_theory = lhood._get_theory(**info["params"])
    tsz_theory *= 3282.8 * 60.**2
    print("tsz_theory",tsz_theory)
    print("tsz_gnfw",tsz_gnfw)
    
    assert np.allclose(tsz_gnfw,tsz_theory,atol=1.e-1,rtol=0.01)
"""

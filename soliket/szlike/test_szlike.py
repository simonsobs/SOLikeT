import numpy as np
from soliket.szlike import SZLikelihood
from cobaya.model import get_model


previously_calculated_differences_ksz = [ksz_data - gnfw_I_calculate_outside_SOliket]
previously_calculated_differences_tsz = [tsz_data - gnfw_I_calculate_outside_SOliket] 

def test_ksz():
    #I'm not sure what is necessary here, check others for examples. Do we need yaml?
    info = {"likelihood": {"SZLikelihood":
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

    thta_arc, ksz_data = np.loadtxt('soliket/tests/data/TNG_data.txt', usecols=(0,1),unpack=True) #should this be in tests for folder for szlike?
    cov_ksz = np.loadtxt('soliket/tests/data/....txt')

    lhood.thta_arc = thta_arc
    lhood.ksz_data = ksz_data
    lhood.cov_ksz = cov_ksz #check what we call this in szlike? might be dy? check units

    ksz_theory = lhood._get_theory(**info["params"])

    diff_theory = ksz_data - ksz_theory 

    assert np.array_equal(previously_calculated_differences_ksz, diff_theory)


def test_tsz():
    #I'm not sure what is necessary here, check others for examples. Do we need yaml?
    info = {"likelihood": {"SZLikelihood":
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

    thta_arc, tsz_data = np.loadtxt('soliket/tests/data/TNG_data.txt', usecols=(0,2),unpack=True) #should this be in tests for folder for szlike?
    cov_tsz = np.loadtxt('soliket/tests/data/....txt')

    lhood.thta_arc = thta_arc
    lhood.tsz_data = tsz_data
    lhood.cov_tsz = cov_tsz #check what we call this in szlike? might be dy? check units

    tsz_theory = lhood._get_theory(**info["params"])

    diff_theory = tsz_data - tsz_theory 

    assert np.array_equal(previously_calculated_differences_tsz, diff_theory)



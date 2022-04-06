import numpy as np
from soliket.szlike import KSZLikelihood, TSZLikelihood
from cobaya.model import get_model


sz_data_file = '/Users/emilymoser/Desktop/SOLikeT/soliket/szlike/TNG_data.txt'
gnfw_test_file = '/Users/emilymoser/Desktop/SOLikeT/soliket/szlike/gnfw_test_projections.txt'
cov_ksz_file= '/Users/emilymoser/Desktop/SOLikeT/soliket/szlike/cov_diskring_ksz_varweight_bootstrap.txt'
cov_tsz_file= '/Users/emilymoser/Desktop/SOLikeT/soliket/szlike/cov_diskring_tsz_varweight_bootstrap.txt'

ksz_data,tsz_data = np.loadtxt(sz_data_file,usecols=(1,2),unpack=True)
ksz_gnfw,tsz_gnfw = np.loadtxt(gnfw_test_file,usecols=(1,2),unpack=True)

ksz_differences = ksz_data - ksz_gnfw
tsz_differences = tsz_data - tsz_gnfw 

def test_ksz():
    info = {"params": {"Omega_m": 0.25,
                       "Omega_b": 0.044,
                       "hh": 0.7,
                       "Omega_L": 0.75,
                       "rhoc_0": 2.77525e2,
                       "C_OVER_HUBBLE": 2997.9,
                       "XH": 0.76,
                       "v_rms": 1.06e-3,
                       "gnfw_rho0":1536.17633526,
                       "gnfw_al_ksz":1.02481552231,
                       "gnfw_bt_ksz":3.45493977635,
                       "gnfw_A2h_ksz":1.0},
            "likelihood": {"KSZLikelihood":
                            {"external": KSZLikelihood,
                             }
                          }}

    model = get_model(info)
    loglikes, derived = model.loglikes()

    lhood = model.likelihood['KSZLikelihood']

    thta_arc, ksz_data = np.loadtxt(sz_data_file, usecols=(0,1),unpack=True)
    cov_ksz = np.loadtxt(cov_ksz_file)

    lhood.thta_arc = thta_arc
    lhood.ksz_data = ksz_data
    lhood.dy_ksz = np.sqrt(np.diag(cov_ksz)) * 3282.8 * 60.**2

    ksz_theory = lhood._get_theory(**info["params"])

    diff_theory = ksz_data - ksz_theory 

    #print("my gnfw",ksz_gnfw)
    #print("SO calc gnfw",ksz_theory)
    assert np.array_equal(ksz_differences, diff_theory)

def test_tsz():
    info = {"params": {"Omega_m": 0.25,
                       "Omega_b": 0.044,
                       "hh": 0.7,
                       "Omega_L": 0.75,
                       "rhoc_0": 2.77525e2,
                       "C_OVER_HUBBLE": 2997.9,
                       "XH": 0.76,
                       "v_rms": 1.06e-3,
                       "gnfw_P0":9.10766709141,
                       "gnfw_xc_tsz":0.698455690636,
                       "gnfw_bt_tsz":4.76625102519,
                       "gnfw_A2h_tsz":1.0},
            "likelihood": {"TSZLikelihood":
                            {"external": TSZLikelihood,
                             }
                          }}

    model = get_model(info)
    loglikes, derived = model.loglikes()

    lhood = model.likelihood['TSZLikelihood']

    thta_arc, tsz_data = np.loadtxt(sz_data_file, usecols=(0,2),unpack=True)
    cov_tsz = np.loadtxt(cov_tsz_file)

    lhood.thta_arc = thta_arc
    lhood.tsz_data = tsz_data
    lhood.dy_tsz = np.sqrt(np.diag(cov_tsz)) * 3282.8 * 60.**2

    tsz_theory = lhood._get_theory(**info["params"])

    diff_theory = tsz_data - tsz_theory 

    #print("my gnfw",tsz_gnfw)
    #print("SO calc gnfw",tsz_theory)
    assert np.array_equal(tsz_differences, diff_theory)


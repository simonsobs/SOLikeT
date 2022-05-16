import numpy as np
import pytest


@pytest.mark.skip(reason="Under development")
def test_clusters():
    fiducial_params = {
        "ombh2": 0.02225,
        "omch2": 0.1198,
        "H0": 67.3,
        "tau": 0.06,
        "As": 2.2e-9,
        "ns": 0.96,
        "mnu": 0.06,
        "nnu": 3.046,
    }

    info_fiducial = {
        "params": fiducial_params,
        "likelihood": {"soliket.ClusterLikelihood": {"stop_at_error": True}},
        "theory": {
            "camb": {
                "extra_args": {
                    "accurate_massive_neutrino_transfers": True,
                    "num_massive_neutrinos": 1,
                    "redshifts": np.linspace(0, 2, 41),
                    "nonlinear": False,
                    "kmax": 10.0,
                    "dark_energy_model": "ppf",
                }
            },
            # "soliket.CCL": {"stop_at_error": True},
        },
    }

    from cobaya.model import get_model

    print('starting test')

    model_fiducial = get_model(info_fiducial)

    # import pdb
    # pdb.set_trace()

    lnl = model_fiducial.loglikes({})[0]

    assert np.isfinite(lnl)

    like = model_fiducial.likelihood["soliket.ClusterLikelihood"]
    print(like,lnl)
    print('ntot:',like._get_n_expected())
    print('nzexpected:',like._get_nz_expected())

    assert like._get_n_expected() > 40
test_clusters()

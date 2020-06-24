import numpy as np


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
        "likelihood": {"solt.ClusterLikelihood": {"stop_at_error": True}},
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
            }
        },
    }

    from cobaya.model import get_model

    model_fiducial = get_model(info_fiducial)

    lnl = model_fiducial.loglikes({})[0]

    assert np.isfinite(lnl)

    like = model_fiducial.likelihood["solt.ClusterLikelihood"]

    assert like._get_n_expected() > 40

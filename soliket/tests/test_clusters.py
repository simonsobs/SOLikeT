import numpy as np
import pytest
from cobaya.model import get_model

pytestmark = pytest.mark.require_ccl

clusters_like_and_theory = {
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
                "bbn_predictor": "PArthENoPE_880.2_standard.dat"
            }
        },
    },
}


def test_clusters_model(evaluate_one_info, test_cosmology_params):
    evaluate_one_info["params"] = test_cosmology_params
    evaluate_one_info.update(clusters_like_and_theory)

    model_fiducial = get_model(evaluate_one_info)  # noqa F841


def test_clusters_loglike(evaluate_one_info, test_cosmology_params):
    evaluate_one_info["params"] = test_cosmology_params
    evaluate_one_info.update(clusters_like_and_theory)

    model_fiducial = get_model(evaluate_one_info)

    lnl = model_fiducial.loglikes({})[0]

    assert np.isclose(lnl, -847.22462272, rtol=1.e-3, atol=1.e-5)


def test_clusters_n_expected(evaluate_one_info, test_cosmology_params):
    evaluate_one_info["params"] = test_cosmology_params
    evaluate_one_info.update(clusters_like_and_theory)

    model_fiducial = get_model(evaluate_one_info)

    lnl = model_fiducial.loglikes({})[0]

    like = model_fiducial.likelihood["soliket.ClusterLikelihood"]

    assert np.isclose(lnl, -847.22462272, rtol=1.e-3, atol=1.e-5)
    assert like._get_n_expected() > 40

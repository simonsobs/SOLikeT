import numpy as np
import pytest
from solike.tests.test_mflike import cosmo_params, nuisance_params


# @pytest.mark.skip(reason="still in development")
def test_multi():

    lensing_options = {"sim_number": 1}

    pre = "data_sacc_"
    mflike_options = {
        "input_file": pre + "00000.fits",
        "cov_Bbl_file": pre + "w_covar_and_Bbl.fits",
        "stop_at_error": True,
    }

    camb_options = {"extra_args": {"lens_potential_accuracy": 1}}

    fg_params = {"a_tSZ": {"prior": {"min": 3.0, "max": 3.6}}, "a_kSZ": {"prior": {"min": 1.4, "max": 1.8}}}
    mflike_params = {**cosmo_params, **nuisance_params}
    mflike_params.update(fg_params)

    lensing_params = {**cosmo_params}

    info = {
        "likelihood": {
            "solike.gaussian.MultiGaussianLikelihood": {
                "components": ["solike.mflike.MFLike", "solike.SimulatedLensingLikelihood"],
                "options": [mflike_options, lensing_options],
                "stop_at_error": True,
            }
        },
        "theory": {"camb": camb_options},
        "params": {**mflike_params},
    }

    info1 = {
        "likelihood": {"solike.mflike.MFLike": mflike_options},
        "theory": {"camb": camb_options},
        "params": {**mflike_params},
    }

    info2 = {
        "likelihood": {"solike.SimulatedLensingLikelihood": lensing_options},
        "theory": {"camb": camb_options},
        "params": {**lensing_params},
    }

    from cobaya.model import get_model

    model = get_model(info)
    model1 = get_model(info1)
    model2 = get_model(info2)

    fg_values = {"a_tSZ": nuisance_params["a_tSZ"], "a_kSZ": nuisance_params["a_kSZ"]}

    logp = model.loglikes(fg_values, cached=False)[0].sum()
    assert np.isfinite(logp)

    model1_logp = model1.loglikes(fg_values, cached=False)[0].sum()
    model2_logp = model2.loglikes({}, cached=False)[0].sum()
    logp_sum = model1_logp + model2_logp

    assert np.isclose(logp, logp_sum)

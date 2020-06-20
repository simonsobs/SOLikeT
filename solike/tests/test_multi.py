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

    # To test here, the absolute values of the logps are not identical
    # to the sum of components when combined (probably due to numerical issues of
    # computing inv_cov); so here we test to make sure
    # that the change in logp between two different sets of params is identical

    fg_values_a = {"a_tSZ": nuisance_params["a_tSZ"], "a_kSZ": nuisance_params["a_kSZ"]}
    fg_values_b = {k: v * 1.1 for k, v in fg_values_a.items()}

    logp_a = model.loglikes(fg_values_a, cached=False)[0].sum()
    logp_b = model.loglikes(fg_values_b, cached=False)[0].sum()
    d_logp = logp_b - logp_a
    assert np.isfinite(d_logp)

    model1_logp_a = model1.loglikes(fg_values_a, cached=False)[0].sum()
    model2_logp_a = model2.loglikes({}, cached=False)[0].sum()

    model1_logp_b = model1.loglikes(fg_values_b, cached=False)[0].sum()
    model2_logp_b = model2.loglikes({}, cached=False)[0].sum()

    d_logp1 = model1_logp_b - model1_logp_a
    d_logp2 = model2_logp_b - model2_logp_a
    d_logp_sum = d_logp1 + d_logp2

    assert np.isclose(d_logp, d_logp_sum)

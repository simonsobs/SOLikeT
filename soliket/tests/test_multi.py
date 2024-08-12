import numpy as np
from cobaya.tools import resolve_packages_path

from soliket.tests.test_mflike import nuisance_params

packages_path = resolve_packages_path()


def test_multi(test_cosmology_params):
    lensing_options = {"theory_lmax": 5000}

    pre = "test_data_sacc_"
    mflike_options = {
        "input_file": pre + "00000.fits",
        "data_folder": "TestMFLike",
        "stop_at_error": True,
    }

    camb_options = {"extra_args": {"lens_potential_accuracy": 1}}

    fg_params = {
        "a_tSZ": {"prior": {"min": 3.0, "max": 3.6}},
        "a_kSZ": {"prior": {"min": 1.4, "max": 1.8}},
    }
    mflike_params = {**test_cosmology_params, **nuisance_params}
    mflike_params.update(fg_params)

    lensing_params = {**test_cosmology_params}

    info = {
        "likelihood": {
            "soliket.gaussian.MultiGaussianLikelihood": {
                "components": [
                    "soliket.mflike.TestMFLike",
                    "soliket.LensingLikelihood",
                ],
                "options": [mflike_options, lensing_options],
                "stop_at_error": True,
            }
        },
        "theory": {
            "camb": camb_options,
            "soliket.TheoryForge_MFLike": {"stop_at_error": True},
            "soliket.Foreground": {"stop_at_error": True},
            "soliket.BandPass": {"stop_at_error": True},
        },
        "params": {**mflike_params},
    }

    info1 = {
        "likelihood": {"soliket.mflike.TestMFLike": mflike_options},
        "theory": {
            "camb": camb_options,
            "soliket.TheoryForge_MFLike": {"stop_at_error": True},
            "soliket.Foreground": {"stop_at_error": True},
            "soliket.BandPass": {"stop_at_error": True},
        },
        "params": {**mflike_params},
    }

    info2 = {
        "likelihood": {"soliket.LensingLikelihood": lensing_options},
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
    assert np.isclose(d_logp, 0.0052275, rtol=1e-5)

    model1_logp_a = model1.loglikes(fg_values_a, cached=False)[0].sum()
    model2_logp_a = model2.loglikes({}, cached=False)[0].sum()

    model1_logp_b = model1.loglikes(fg_values_b, cached=False)[0].sum()
    model2_logp_b = model2.loglikes({}, cached=False)[0].sum()

    d_logp1 = model1_logp_b - model1_logp_a
    d_logp2 = model2_logp_b - model2_logp_a
    d_logp_sum = d_logp1 + d_logp2

    assert np.isclose(d_logp, d_logp_sum, rtol=1e-5)

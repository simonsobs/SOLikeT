import numpy as np

from solike.tests.test_mflike import cosmo_params, nuisance_params


def test_multi():
    info = {
        "likelihood": {
            "solike.gaussian.MultiGaussianLikelihood": {
                "components": ["solike.mflike.MFLike", "solike.SimulatedLensingLikelihood"],
                "options": [{"sim_id": 0}, {"sim_number": 1}],
                "stop_at_error": True,
            }
        },
        "theory": {"camb": {"extra_args": {"lens_potential_accuracy": 1}}},
        "params": {**cosmo_params, **nuisance_params},
    }

    info1 = {
        "likelihood": {"solike.mflike.MFLike": {"sim_id": 0},},
        "theory": {"camb": {"extra_args": {"lens_potential_accuracy": 1}}},
        "params": {**cosmo_params, **nuisance_params},
    }

    info2 = {
        "likelihood": {"solike.SimulatedLensingLikelihood": {"sim_number": 1},},
        "theory": {"camb": {"extra_args": {"lens_potential_accuracy": 1}}},
        "params": {**cosmo_params},
    }

    from cobaya.model import get_model

    model = get_model(info)
    model1 = get_model(info1)
    model2 = get_model(info2)

    logp = np.float(model.loglikes({}, cached=False)[0])
    assert np.isfinite(logp)

    assert np.isclose(
        logp, float(model1.loglikes({}, cached=False)[0] + model2.loglikes({}, cached=False)[0])
    )

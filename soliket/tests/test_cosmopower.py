"""
Check that CosmoPower gives the correct Planck CMB power spectrum.
"""

import os

import numpy as np
from cobaya.model import get_model

fiducial_params = {
    "ombh2": 0.0224,
    "omch2": 0.122,
    "h": 0.67,
    "tau": 0.065,
    "ns": 0.9645,
    "logA": 3.07,
    "A_planck": 1.0,
    # derived params
    "As": {"value": "lambda logA: 1e-10 * np.exp(logA)"},
    "H0": {"value": "lambda h: h * 100.0"},
}

info_dict = {
    "params": fiducial_params,
    "likelihood": {
        # This should be installed, otherwise one should install it via cobaya.
        "planck_2018_highl_plik.TTTEEE_lite_native": {"stop_at_error": True}
    },
    "theory": {
        "soliket.CosmoPower": {
            "stop_at_error": True,
            "network_settings": {
                "tt": {
                    "type": "NN",
                    "log": True,
                    "filename": "cmb_TT_NN",
                    # If your network has been trained on (l (l+1) / 2 pi) C_l,
                    # this flag needs to be set.
                    "has_ell_factor": False,
                },
                "ee": {
                    "type": "NN",
                    "log": True,
                    "filename": "cmb_EE_NN",
                    "has_ell_factor": False,
                },
                "te": {
                    "type": "PCAplusNN",
                    # Trained on Cl, not log(Cl)
                    "log": False,
                    "filename": "cmb_TE_PCAplusNN",
                    "has_ell_factor": False,
                },
            },
            "renames": {
                "ombh2": "omega_b",
                "omch2": "omega_cdm",
                "ns": "n_s",
                "logA": "ln10^{10}A_s",
                "tau": "tau_reio",
            },
        }
    },
}


def test_cosmopower_import(check_skip_cosmopower):
    from soliket.cosmopower import CosmoPower  # noqa F401


def test_cosmopower_theory(request, check_skip_cosmopower, install_planck_lite):
    info_dict["theory"]["soliket.CosmoPower"]["network_path"] = os.path.join(
        request.config.rootdir, "soliket/cosmopower/data/CP_paper"
    )
    model_fiducial = get_model(info_dict)  # noqa F841


def test_cosmopower_loglike(request, check_skip_cosmopower, install_planck_lite):
    info_dict["theory"]["soliket.CosmoPower"]["network_path"] = os.path.join(
        request.config.rootdir, "soliket/cosmopower/data/CP_paper"
    )
    model_cp = get_model(info_dict)

    logL_cp = float(model_cp.loglikes({})[0])

    assert np.isclose(logL_cp, -295.139)


def test_cosmopower_against_camb(request, check_skip_cosmopower, install_planck_lite):
    info_dict["theory"] = {"camb": {"stop_at_error": True}}
    model_camb = get_model(info_dict)
    logL_camb = float(model_camb.loglikes({})[0])
    camb_cls = model_camb.theory["camb"].get_Cl()

    info_dict["theory"] = {
        "soliket.CosmoPower": {
            "stop_at_error": True,
            "extra_args": {"lmax": camb_cls["ell"].max()},
            "network_path": os.path.join(
                request.config.rootdir, "soliket/cosmopower/data/CP_paper"
            ),
            "network_settings": {
                "tt": {"type": "NN", "log": True, "filename": "cmb_TT_NN"},
                "ee": {"type": "NN", "log": True, "filename": "cmb_EE_NN"},
                "te": {"type": "PCAplusNN", "log": False, "filename": "cmb_TE_PCAplusNN"},
            },
            "renames": {
                "ombh2": "omega_b",
                "omch2": "omega_cdm",
                "ns": "n_s",
                "logA": "ln10^{10}A_s",
                "tau": "tau_reio",
            },
        }
    }

    model_cp = get_model(info_dict)
    logL_cp = float(model_cp.loglikes({})[0])
    cp_cls = model_cp.theory["soliket.CosmoPower"].get_Cl()

    nanmask = ~np.isnan(cp_cls["tt"])

    assert np.allclose(cp_cls["tt"][nanmask], camb_cls["tt"][nanmask], rtol=1.0e-2)
    assert np.isclose(logL_camb, logL_cp, rtol=1.0e-1)

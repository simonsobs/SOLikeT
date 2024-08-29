import copy
import os

import numpy as np
from cobaya.model import get_model
import pytest

foreground_params = {
    "a_tSZ": 3.3044404448917724,
    "a_kSZ": 1.6646620740058649,
    "a_p": 6.912474322461401,
    "beta_p": 2.077474196171309,
    "a_c": 4.88617700670901,
    "beta_c": 2.2030316332596014,
    "a_s": 3.099214100532393,
    "T_d": 9.60,
    "a_gtt": 0,
    "a_gte": 0,
    "a_gee": 0,
    "a_psee": 0,
    "a_pste": 0,
    "xi": 0,
}


def test_foreground_import():
    from soliket.foreground import Foreground  # noqa F401


def test_wrong_types():
    from soliket.foreground import Foreground

    base_case = {"spectra": {
        "polarizations": ["a", "b", "c"],
        "lmin": 1,
        "lmax": 2,
        "exp_ch": ["A", "B", "C"],
        "eff_freqs": [1, 2, 3],
    }, "foregrounds": {
        "components": {"a": ["b", "c"], "d": ["e", "f"]},
        "normalisation": {"nu_0": 1.0, "ell_0": 2, "T_CMB": 3.0},
    }, "params": {}}

    wrong_main_cases = {
        "spectra": "not_a_dict",
        "foregrounds": "not_a_dict",
        "params": "not_a_dict",
    }

    wrong_spectra_cases = {
        "polarizations": "not_a_list",
        "lmin": "not_an_int",
        "lmax": "not_an_int",
        "exp_ch": "not_a_list",
        "eff_freqs": "not_a_list",
    }

    wrong_foregrounds_cases = {
        "components": "not_a_dict",
        "normalisation": "not_a_dict",
    }

    wrong_normalization_cases = {
        "nu_0": "not_a_float",
        "ell_0": "not_an_int",
        "T_CMB": "not_a_float",
    }

    for key, wrong_value in wrong_main_cases.items():
        case = copy.deepcopy(base_case)
        case[key] = wrong_value
        with pytest.raises(TypeError):
            _ = Foreground(**case)

    for key, wrong_value in wrong_spectra_cases.items():
        case = copy.deepcopy(base_case)
        case["spectra"][key] = wrong_value
        with pytest.raises(TypeError):
            _ = Foreground(**case)

    for key, wrong_value in wrong_foregrounds_cases.items():
        case = copy.deepcopy(base_case)
        case["foregrounds"][key] = wrong_value
        with pytest.raises(TypeError):
            _ = Foreground(**case)

    for key, wrong_value in wrong_normalization_cases.items():
        case = copy.deepcopy(base_case)
        case["foregrounds"]["normalisation"][key] = wrong_value
        with pytest.raises(TypeError):
            _ = Foreground(**case)


def test_foreground_model(evaluate_one_info):
    from soliket.foreground import Foreground

    evaluate_one_info["params"] = foreground_params
    evaluate_one_info["theory"] = {
        "foreground": {"external": Foreground},
    }
    model = get_model(evaluate_one_info)  # noqa F841


def test_foreground_compute(evaluate_one_info):
    from soliket.bandpass import BandPass
    from soliket.foreground import Foreground

    evaluate_one_info["params"] = foreground_params

    evaluate_one_info["theory"] = {
        "foreground": {"external": Foreground},
        "bandpass": {"external": BandPass},
    }

    evaluate_one_info["foregrounds"] = {
        "normalisation": {"nu_0": 150.0, "ell_0": 3000, "T_CMB": 2.725},
        "components": {
            "tt": ["kSZ", "tSZ_and_CIB", "cibp", "dust", "radio"],
            "te": ["radio", "dust"],
            "ee": ["radio", "dust"],
        },
    }

    evaluate_one_info["spectra"] = {
        "polarizations": ["tt", "te", "ee"],
        "lmin": 2,
        "lmax": 9000,
        "exp_ch": ["LAT_93", "LAT_145", "LAT_225"],
        "eff_freqs": [93, 145, 225],
    }

    nu_0 = evaluate_one_info["foregrounds"]["normalisation"]["nu_0"]
    ell_0 = evaluate_one_info["foregrounds"]["normalisation"]["ell_0"]
    ell = np.arange(
        evaluate_one_info["spectra"]["lmin"], evaluate_one_info["spectra"]["lmax"] + 1
    )
    requested_cls = evaluate_one_info["spectra"]["polarizations"]
    components = evaluate_one_info["foregrounds"]["components"]
    exp_ch = evaluate_one_info["spectra"]["exp_ch"]
    eff_freqs = np.asarray(evaluate_one_info["spectra"]["eff_freqs"])
    bands = {
        f"{expc}_s0": {"nu": [eff_freqs[iexpc]], "bandpass": [1.0]}
        for iexpc, expc in enumerate(exp_ch)
    }

    model = get_model(evaluate_one_info)
    model.add_requirements(
        {
            "fg_dict": {
                "requested_cls": requested_cls,
                "ell": ell,
                "exp_ch": exp_ch,
                "bands": bands,
            },
        }
    )

    model.logposterior(evaluate_one_info["params"])  # force computation of model

    lhood = model.likelihood["one"]

    fg_model = lhood.provider.get_fg_dict()
    fg_model_test = get_fg(
        exp_ch, eff_freqs, ell, ell_0, nu_0, requested_cls, components, evaluate_one_info
    )

    for k in fg_model_test.keys():
        assert np.allclose(fg_model[k], fg_model_test[k])


def get_fg(
    freqs, bandint_freqs, ell, ell_0, nu_0, requested_cls, components, evaluate_one_info
):
    from fgspectra import cross as fgc
    from fgspectra import frequency as fgf
    from fgspectra import power as fgp

    template_path = os.path.join(os.path.dirname(os.path.abspath(fgp.__file__)), "data")
    cibc_file = os.path.join(template_path, "cl_cib_Choi2020.dat")

    ksz = fgc.FactorizedCrossSpectrum(fgf.ConstantSED(), fgp.kSZ_bat())
    cibp = fgc.FactorizedCrossSpectrum(fgf.ModifiedBlackBody(), fgp.PowerLaw())
    radio = fgc.FactorizedCrossSpectrum(fgf.PowerLaw(), fgp.PowerLaw())
    tsz = fgc.FactorizedCrossSpectrum(fgf.ThermalSZ(), fgp.tSZ_150_bat())
    cibc = fgc.FactorizedCrossSpectrum(fgf.CIB(), fgp.PowerSpectrumFromFile(cibc_file))
    dust = fgc.FactorizedCrossSpectrum(fgf.ModifiedBlackBody(), fgp.PowerLaw())
    tSZ_and_CIB = fgc.SZxCIB_Choi2020()

    ell_clp = ell * (ell + 1.0)
    ell_0clp = ell_0 * (ell_0 + 1.0)
    fg_component_list = {s: components[s] for s in requested_cls}

    model = {}
    model["tt", "kSZ"] = evaluate_one_info["params"]["a_kSZ"] * ksz(
        {"nu": bandint_freqs}, {"ell": ell, "ell_0": ell_0}
    )

    model["tt", "cibp"] = evaluate_one_info["params"]["a_p"] * cibp(
        {
            "nu": bandint_freqs,
            "nu_0": nu_0,
            "temp": evaluate_one_info["params"]["T_d"],
            "beta": evaluate_one_info["params"]["beta_p"],
        },
        {"ell": ell_clp, "ell_0": ell_0clp, "alpha": 1},
    )

    model["tt", "radio"] = evaluate_one_info["params"]["a_s"] * radio(
        {"nu": bandint_freqs, "nu_0": nu_0, "beta": -0.5 - 2.0},
        {"ell": ell_clp, "ell_0": ell_0clp, "alpha": 1},
    )

    model["tt", "tSZ"] = evaluate_one_info["params"]["a_tSZ"] * tsz(
        {"nu": bandint_freqs, "nu_0": nu_0}, {"ell": ell, "ell_0": ell_0}
    )

    model["tt", "cibc"] = evaluate_one_info["params"]["a_c"] * cibc(
        {
            "nu": bandint_freqs,
            "nu_0": nu_0,
            "temp": evaluate_one_info["params"]["T_d"],
            "beta": evaluate_one_info["params"]["beta_c"],
        },
        {"ell": ell, "ell_0": ell_0},
    )

    model["tt", "dust"] = evaluate_one_info["params"]["a_gtt"] * dust(
        {"nu": bandint_freqs, "nu_0": nu_0, "temp": 19.6, "beta": 1.5},
        {"ell": ell, "ell_0": 500.0, "alpha": -0.6},
    )

    model["tt", "tSZ_and_CIB"] = tSZ_and_CIB(
        {
            "kwseq": (
                {"nu": bandint_freqs, "nu_0": nu_0},
                {
                    "nu": bandint_freqs,
                    "nu_0": nu_0,
                    "temp": evaluate_one_info["params"]["T_d"],
                    "beta": evaluate_one_info["params"]["beta_c"],
                },
            )
        },
        {
            "kwseq": (
                {
                    "ell": ell,
                    "ell_0": ell_0,
                    "amp": evaluate_one_info["params"]["a_tSZ"],
                },
                {"ell": ell, "ell_0": ell_0, "amp": evaluate_one_info["params"]["a_c"]},
                {
                    "ell": ell,
                    "ell_0": ell_0,
                    "amp": -evaluate_one_info["params"]["xi"]
                    * np.sqrt(
                        evaluate_one_info["params"]["a_tSZ"]
                        * evaluate_one_info["params"]["a_c"]
                    ),
                },
            )
        },
    )

    model["ee", "radio"] = evaluate_one_info["params"]["a_psee"] * radio(
        {"nu": bandint_freqs, "nu_0": nu_0, "beta": -0.5 - 2.0},
        {"ell": ell_clp, "ell_0": ell_0clp, "alpha": 1},
    )

    model["ee", "dust"] = evaluate_one_info["params"]["a_gee"] * dust(
        {"nu": bandint_freqs, "nu_0": nu_0, "temp": 19.6, "beta": 1.5},
        {"ell": ell, "ell_0": 500.0, "alpha": -0.4},
    )

    model["te", "radio"] = evaluate_one_info["params"]["a_pste"] * radio(
        {"nu": bandint_freqs, "nu_0": nu_0, "beta": -0.5 - 2.0},
        {"ell": ell_clp, "ell_0": ell_0clp, "alpha": 1},
    )

    model["te", "dust"] = evaluate_one_info["params"]["a_gte"] * dust(
        {"nu": bandint_freqs, "nu_0": nu_0, "temp": 19.6, "beta": 1.5},
        {"ell": ell, "ell_0": 500.0, "alpha": -0.4},
    )

    fg_dict = {}
    for c1, f1 in enumerate(freqs):
        for c2, f2 in enumerate(freqs):
            for s in requested_cls:
                fg_dict[s, "all", f1, f2] = np.zeros(len(ell))
                for comp in fg_component_list[s]:
                    if comp == "tSZ_and_CIB":
                        fg_dict[s, "tSZ", f1, f2] = model[s, "tSZ"][c1, c2]
                        fg_dict[s, "cibc", f1, f2] = model[s, "cibc"][c1, c2]
                        fg_dict[s, "tSZxCIB", f1, f2] = (
                            model[s, comp][c1, c2]
                            - model[s, "tSZ"][c1, c2]
                            - model[s, "cibc"][c1, c2]
                        )
                        fg_dict[s, "all", f1, f2] += model[s, comp][c1, c2]
                    else:
                        fg_dict[s, comp, f1, f2] = model[s, comp][c1, c2]
                        fg_dict[s, "all", f1, f2] += fg_dict[s, comp, f1, f2]

    return fg_dict

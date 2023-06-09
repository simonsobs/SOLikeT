# pytest -k bandpass -v .

import pytest
import numpy as np
import os

from cobaya.model import get_model
from cobaya.run import run

info = {"params": {
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
                   },
        "likelihood": {"one": None},
        "sampler": {"evaluate": None},
        "debug": True
       }


def test_foreground_import():
    from soliket.foreground import Foreground


def test_foreground_model():
    from soliket.foreground import Foreground

    info["theory"] = {"foreground": {"external": Foreground},
                     }
    model = get_model(info)  # noqa F841


def test_foreground_compute():

    from soliket.foreground import Foreground
    from soliket.bandpass import BandPass

    info["theory"] = {
                      "foreground": {"external": Foreground},
                      "bandpass": {"external": BandPass},
                      }

    info["foregrounds"] = {
                           "normalisation": {"nu_0": 150.0,
                                             "ell_0": 3000,
                                             "T_CMB": 2.725
                                             },
    
                           "components": {"tt": ["kSZ", "tSZ_and_CIB", 
                                                 "cibp", "dust", "radio"], 
                                          "te": ["radio", "dust"],
                                          "ee": ["radio", "dust"]
                                          },
                            }

    info["spectra"] = {
                       "polarizations": ["tt", "te", "ee"],
                       "lmin": 2,
                       "lmax": 9000,
                       "exp_ch": ["LAT_93", "LAT_145", "LAT_225"],
                       "eff_freqs": [93, 145, 225]
                       }

    nu_0 = info["foregrounds"]["normalisation"]["nu_0"]
    ell_0 = info["foregrounds"]["normalisation"]["ell_0"]
    ell = np.arange(info["spectra"]["lmin"], info["spectra"]["lmax"] + 1)
    requested_cls = info["spectra"]["polarizations"]
    components = info["foregrounds"]["components"]
    exp_ch = info["spectra"]["exp_ch"]
    eff_freqs = np.asarray(info["spectra"]["eff_freqs"])
    bands = {f"{expc}_s0": {'nu': [eff_freqs[iexpc]], 'bandpass': [1.]} 
                for iexpc, expc in enumerate(exp_ch)}

    model = get_model(info)  # noqa F841
    model.add_requirements({"fg_dict": {
                                        "requested_cls": requested_cls,
                                        "ell": ell,
                                        "exp_ch": exp_ch,
                                        "bands": bands},
                            })

    model.logposterior(info['params'])  # force computation of model

    lhood = model.likelihood['one']

    fg_model = lhood.provider.get_fg_dict()
    fg_model_test = get_fg(exp_ch, eff_freqs, ell, ell_0, nu_0, requested_cls, components)

    for k in fg_model_test.keys():
        assert np.allclose(fg_model[k], fg_model_test[k])
        

def get_fg(freqs, bandint_freqs, ell, ell_0, nu_0, requested_cls, components):

    from fgspectra import cross as fgc
    from fgspectra import frequency as fgf
    from fgspectra import power as fgp

    template_path = os.path.join(os.path.dirname(os.path.abspath(fgp.__file__)),
                                     'data')
    cibc_file = os.path.join(template_path, 'cl_cib_Choi2020.dat')

    ksz = fgc.FactorizedCrossSpectrum(fgf.ConstantSED(), fgp.kSZ_bat())
    cibp = fgc.FactorizedCrossSpectrum(fgf.ModifiedBlackBody(), fgp.PowerLaw())
    radio = fgc.FactorizedCrossSpectrum(fgf.PowerLaw(), fgp.PowerLaw())
    tsz = fgc.FactorizedCrossSpectrum(fgf.ThermalSZ(), fgp.tSZ_150_bat())
    cibc = fgc.FactorizedCrossSpectrum(fgf.CIB(),
                                           fgp.PowerSpectrumFromFile(cibc_file))
    dust = fgc.FactorizedCrossSpectrum(fgf.ModifiedBlackBody(), fgp.PowerLaw())
    tSZ_and_CIB = fgc.SZxCIB_Choi2020()

    ell_clp = ell * (ell + 1.)
    ell_0clp = ell_0 * (ell_0 + 1.)
    fg_component_list = {s: components[s] for s in requested_cls}

    model = {}
    model["tt", "kSZ"] = info["params"]["a_kSZ"] * ksz({"nu": bandint_freqs},
                                                           {"ell": ell,
                                                            "ell_0": ell_0})

    model["tt", "cibp"] = info["params"]["a_p"] * cibp({"nu": bandint_freqs,
                                                        "nu_0": nu_0,
                                                        "temp": info["params"]["T_d"],
                                                        "beta": info["params"]["beta_p"]},
                                                       {"ell": ell_clp,
                                                        "ell_0": ell_0clp,
                                                        "alpha": 1})

    model["tt", "radio"] = info["params"]["a_s"] * radio({"nu": bandint_freqs,
                                                          "nu_0": nu_0,
                                                          "beta": -0.5 - 2.},
                                                         {"ell": ell_clp,
                                                          "ell_0": ell_0clp,
                                                          "alpha": 1})

    model["tt", "tSZ"] = info["params"]["a_tSZ"] * tsz({"nu": bandint_freqs,
                                                        "nu_0": nu_0},
                                                       {"ell": ell,
                                                        "ell_0": ell_0})

    model["tt", "cibc"] = info["params"]["a_c"] * cibc({"nu": bandint_freqs,
                                                        "nu_0": nu_0,
                                                        "temp": info["params"]["T_d"],
                                                        "beta": info["params"]["beta_c"]},
                                                       {"ell": ell,
                                                        "ell_0": ell_0})

    model["tt", "dust"] = info["params"]["a_gtt"] * dust({"nu": bandint_freqs,
                                                          "nu_0": nu_0,
                                                          "temp": 19.6,
                                                          "beta": 1.5},
                                                         {"ell": ell,
                                                          "ell_0": 500.,
                                                          "alpha": -0.6})

    model["tt", "tSZ_and_CIB"] = \
            tSZ_and_CIB({'kwseq': ({'nu': bandint_freqs, 'nu_0': nu_0},
                                        {'nu': bandint_freqs, 'nu_0': nu_0,
                                         'temp': info["params"]['T_d'],
                                         'beta': info["params"]["beta_c"]})},
                             {'kwseq': ({'ell': ell, 'ell_0': ell_0,
                                         'amp': info["params"]['a_tSZ']},
                                        {'ell': ell, 'ell_0': ell_0,
                                         'amp': info["params"]['a_c']},
                                        {'ell': ell, 'ell_0': ell_0,
                                         'amp': - info["params"]['xi'] \
                                                    * np.sqrt(info["params"]['a_tSZ'] *
                                                              info["params"]['a_c'])})})

    model["ee", "radio"] = info["params"]["a_psee"] * radio({"nu": bandint_freqs,
                                                                 "nu_0": nu_0,
                                                                 "beta": -0.5 - 2.},
                                                                {"ell": ell_clp,
                                                                "ell_0": ell_0clp,
                                                                 "alpha": 1})

    model["ee", "dust"] = info["params"]["a_gee"] * dust({"nu": bandint_freqs,
                                                              "nu_0": nu_0,
                                                              "temp": 19.6,
                                                              "beta": 1.5},
                                                             {"ell": ell,
                                                              "ell_0": 500.,
                                                              "alpha": -0.4})

    model["te", "radio"] = info["params"]["a_pste"] * radio({"nu": bandint_freqs,
                                                                 "nu_0": nu_0,
                                                                 "beta": -0.5 - 2.},
                                                                {"ell": ell_clp,
                                                                 "ell_0": ell_0clp,
                                                                 "alpha": 1})

    model["te", "dust"] = info["params"]["a_gte"] * dust({"nu": bandint_freqs,
                                                              "nu_0": nu_0,
                                                              "temp": 19.6,
                                                              "beta": 1.5},
                                                             {"ell": ell,
                                                              "ell_0": 500.,
                                                              "alpha": -0.4})

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

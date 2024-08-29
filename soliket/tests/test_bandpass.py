import numpy as np
from cobaya.model import get_model

from soliket.constants import T_CMB, h_Planck, k_Boltzmann

bandpass_params = {
    "bandint_shift_LAT_93": 0.0,
    "bandint_shift_LAT_145": 0.0,
    "bandint_shift_LAT_225": 0.0
}

bands = {"LAT_93_s0": {"nu": [93], "bandpass": [1.]},
         "LAT_145_s0": {"nu": [145], "bandpass": [1.]},
         "LAT_225_s0": {"nu": [225], "bandpass": [1.]}}
exp_ch = [k.replace("_s0", "") for k in bands.keys()]


def _cmb2bb(nu):
    # NB: numerical factors not included
    x = nu * h_Planck * 1e9 / k_Boltzmann / T_CMB
    return np.exp(x) * (nu * x / np.expm1(x)) ** 2


# noinspection PyUnresolvedReferences
def test_bandpass_import():
    from soliket.bandpass import BandPass  # noqa F401


def test_bandpass_model(evaluate_one_info):
    from soliket.bandpass import BandPass

    evaluate_one_info["params"] = bandpass_params
    evaluate_one_info["theory"] = {"bandpass": {
        "external": BandPass,
    },
    }
    model = get_model(evaluate_one_info)  # noqa F841


def test_bandpass_read_from_sacc(evaluate_one_info):
    from soliket.bandpass import BandPass

    # testing the default read_from_sacc
    evaluate_one_info["params"] = bandpass_params
    evaluate_one_info["theory"] = {
        "bandpass": {"external": BandPass},
    }

    model = get_model(evaluate_one_info)
    model.add_requirements({"bandint_freqs": {"bands": bands}
                            })

    model.logposterior(evaluate_one_info['params'])  # force computation of model

    lhood = model.likelihood['one']

    bandpass = lhood.provider.get_bandint_freqs()

    bandint_freqs = np.empty_like(exp_ch, dtype=float)
    for ifr, fr in enumerate(exp_ch):
        bandpar = 'bandint_shift_' + fr
        bandint_freqs[ifr] = (
            np.asarray(bands[fr + "_s0"]["nu"]) + evaluate_one_info["params"][bandpar]
        )

    assert np.allclose(bandint_freqs, bandpass)


def test_bandpass_top_hat(evaluate_one_info):
    from soliket.bandpass import BandPass

    # now testing top-hat construction
    evaluate_one_info["params"] = bandpass_params
    evaluate_one_info["theory"] = {
        "bandpass": {"external": BandPass,
                     "top_hat_band": {
                         "nsteps": 3,
                         "bandwidth": 0.5},
                     "external_bandpass": {},
                     "read_from_sacc": False,
                     },
    }

    model = get_model(evaluate_one_info)
    model.add_requirements({"bandint_freqs": {"bands": bands}
                            })
    model.logposterior(evaluate_one_info['params'])  # force computation of model

    lhood = model.likelihood['one']

    bandpass = lhood.provider.get_bandint_freqs()

    bandint_freqs = []
    nsteps = evaluate_one_info["theory"]["bandpass"]["top_hat_band"]["nsteps"]
    bandwidth = evaluate_one_info["theory"]["bandpass"]["top_hat_band"]["bandwidth"]
    for ifr, fr in enumerate(exp_ch):
        bandpar = 'bandint_shift_' + fr
        bd = bands[f"{fr}_s0"]
        nu_ghz, bp = np.asarray(bd["nu"]), np.asarray(bd["bandpass"])
        fr = nu_ghz @ bp / bp.sum()
        bandlow = fr * (1 - bandwidth * .5)
        bandhigh = fr * (1 + bandwidth * .5)
        nub = np.linspace(bandlow + evaluate_one_info["params"][bandpar],
                          bandhigh + evaluate_one_info["params"][bandpar],
                          nsteps, dtype=float)
        tranb = _cmb2bb(nub)
        tranb_norm = np.trapz(_cmb2bb(nub), nub)
        bandint_freqs.append([nub, tranb / tranb_norm])

    assert np.allclose(bandint_freqs, bandpass)


def test_bandpass_external_file(request, evaluate_one_info):
    import os

    from soliket.bandpass import BandPass

    filepath = os.path.join(request.config.rootdir,
                            "soliket/tests/data/")
    # now testing reading from external file
    evaluate_one_info["params"] = bandpass_params
    evaluate_one_info["theory"] = {
        "bandpass": {"external": BandPass,
                     "data_folder": f"{filepath}",
                     "top_hat_band": {},
                     "external_bandpass": {
                         "path": "test_bandpass"},
                     "read_from_sacc": False,
                     },
    }

    model = get_model(evaluate_one_info)
    model.add_requirements({"bandint_freqs": {"bands": bands}
                            })

    model.logposterior(evaluate_one_info['params'])  # force computation of model

    lhood = model.likelihood['one']

    bandpass = lhood.provider.get_bandint_freqs()

    path = os.path.normpath(os.path.join(
        evaluate_one_info["theory"]["bandpass"]["data_folder"],
        evaluate_one_info["theory"]["bandpass"]["external_bandpass"]["path"]))

    arrays = os.listdir(path)
    external_bandpass = []
    for a in arrays:
        nu_ghz, bp = np.loadtxt(path + "/" + a, usecols=(0, 1), unpack=True)
        external_bandpass.append([a, nu_ghz, bp])

    bandint_freqs = []
    for expc, nu_ghz, bp in external_bandpass:
        bandpar = "bandint_shift_" + expc
        nub = nu_ghz + evaluate_one_info["params"][bandpar]
        if not hasattr(bp, "__len__"):
            bandint_freqs.append(nub)
            bandint_freqs = np.asarray(bandint_freqs)
        else:
            trans_norm = np.trapz(bp * _cmb2bb(nub), nub)
            trans = bp / trans_norm * _cmb2bb(nub)
            bandint_freqs.append([nub, trans])

    assert np.allclose(bandint_freqs, bandpass)

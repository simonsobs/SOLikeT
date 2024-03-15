import numpy as np

from cobaya.model import get_model
from cobaya.tools import resolve_packages_path
from ..constants import T_CMB, h_Planck, k_Boltzmann

packages_path = resolve_packages_path()

info = {"params": {
    "bandint_shift_LAT_93": 0.0,
    "bandint_shift_LAT_145": 0.0,
    "bandint_shift_LAT_225": 0.0
},
    "likelihood": {"one": None},
    "sampler": {"evaluate": None},
    "debug": True
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


def test_bandpass_model():
    from soliket.bandpass import BandPass

    info["theory"] = {"bandpass": {
        "external": BandPass,
    },
    }
    model = get_model(info)  # noqa F841


def test_bandpass_read_from_sacc():
    from soliket.bandpass import BandPass

    # testing the default read_from_sacc
    info["theory"] = {
        "bandpass": {"external": BandPass},
    }

    model = get_model(info)  # noqa F841
    model.add_requirements({"bandint_freqs": {"bands": bands}
                            })

    model.logposterior(info['params'])  # force computation of model

    lhood = model.likelihood['one']

    bandpass = lhood.provider.get_bandint_freqs()

    bandint_freqs = np.empty_like(exp_ch, dtype=float)
    for ifr, fr in enumerate(exp_ch):
        bandpar = 'bandint_shift_' + fr
        bandint_freqs[ifr] = np.asarray(bands[fr + "_s0"]["nu"]) + info["params"][bandpar]

    assert np.allclose(bandint_freqs, bandpass)


def test_bandpass_top_hat():
    from soliket.bandpass import BandPass
    # now testing top-hat construction
    info["theory"].update({
        "bandpass": {"external": BandPass,
                     "top_hat_band": {
                         "nsteps": 3,
                         "bandwidth": 0.5},
                     "external_bandpass": {},
                     "read_from_sacc": {},
                     },
    })

    model = get_model(info)
    model.add_requirements({"bandint_freqs": {"bands": bands}
                            })
    model.logposterior(info['params'])  # force computation of model

    lhood = model.likelihood['one']

    bandpass = lhood.provider.get_bandint_freqs()

    bandint_freqs = []
    nsteps = info["theory"]["bandpass"]["top_hat_band"]["nsteps"]
    bandwidth = info["theory"]["bandpass"]["top_hat_band"]["bandwidth"]
    for ifr, fr in enumerate(exp_ch):
        bandpar = 'bandint_shift_' + fr
        bd = bands[f"{fr}_s0"]
        nu_ghz, bp = np.asarray(bd["nu"]), np.asarray(bd["bandpass"])
        fr = nu_ghz @ bp / bp.sum()
        bandlow = fr * (1 - bandwidth * .5)
        bandhigh = fr * (1 + bandwidth * .5)
        nub = np.linspace(bandlow + info["params"][bandpar],
                          bandhigh + info["params"][bandpar],
                          nsteps, dtype=float)
        tranb = _cmb2bb(nub)
        tranb_norm = np.trapz(_cmb2bb(nub), nub)
        bandint_freqs.append([nub, tranb / tranb_norm])

    assert np.allclose(bandint_freqs, bandpass)


def test_bandpass_external_file():
    from soliket.bandpass import BandPass
    import os

    filepath = os.path.join(packages_path,
                            "../../../soliket/tests/data/")
    # now testing reading from external file
    info["theory"].update({
        "bandpass": {"external": BandPass,
                     "data_folder": f"{filepath}",
                     "top_hat_band": {},
                     "external_bandpass": {
                         "path": "test_bandpass"},
                     "read_from_sacc": {},
                     },
    })

    model = get_model(info)
    model.add_requirements({"bandint_freqs": {"bands": bands}
                            })

    model.logposterior(info['params'])  # force computation of model

    lhood = model.likelihood['one']

    bandpass = lhood.provider.get_bandint_freqs()

    path = os.path.normpath(os.path.join(
        info["theory"]["bandpass"]["data_folder"],
        info["theory"]["bandpass"]["external_bandpass"]["path"]))

    arrays = os.listdir(path)
    external_bandpass = []
    for a in arrays:
        nu_ghz, bp = np.loadtxt(path + "/" + a, usecols=(0, 1), unpack=True)
        external_bandpass.append([a, nu_ghz, bp])

    bandint_freqs = []
    for expc, nu_ghz, bp in external_bandpass:
        bandpar = "bandint_shift_" + expc
        nub = nu_ghz + info["params"][bandpar]
        if not hasattr(bp, "__len__"):
            bandint_freqs.append(nub)
            bandint_freqs = np.asarray(bandint_freqs)
        else:
            trans_norm = np.trapz(bp * _cmb2bb(nub), nub)
            trans = bp / trans_norm * _cmb2bb(nub)
            bandint_freqs.append([nub, trans])

    assert np.allclose(bandint_freqs, bandpass)

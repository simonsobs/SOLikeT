# pytest -k bandpass -v .

import pytest
import numpy as np

from cobaya.model import get_model
from cobaya.run import run

info = {"params": {
                   "bandint_shift_LAT_93": 0.0,
                   "bandint_shift_LAT_145": 0.0,
                   "bandint_shift_LAT_225": 0.0
                   },
        "likelihood": {"one": None},
        "sampler": {"evaluate": None},
        "debug": True
       }

freqs = np.array([93, 145, 225])


def test_bandpass_import():
    from soliket.bandpass import BandPass


def test_bandpass_model():
    from soliket.bandpass import BandPass

    info["theory"] = {"bandpass": {
                                   "external": BandPass,
                                   },
                     }
    model = get_model(info)  # noqa F841


def test_bandpass_compute():

    from soliket.bandpass import BandPass

    info["theory"] = {
               "bandpass": {"external": BandPass},
               }

    model = get_model(info)  # noqa F841
    model.add_requirements({"bandint_freqs": {"freqs": freqs}
                            })

    model.logposterior(info['params'])  # force computation of model

    lhood = model.likelihood['one']

    bandpass = lhood.provider.get_bandint_freqs()

    bandint_freqs = np.empty_like(freqs, dtype=float)
    for ifr, fr in enumerate(freqs):
        bandpar = 'bandint_shift_' + str(fr)
        bandint_freqs[ifr] = fr + info["params"][bandpar]

    assert np.allclose(bandint_freqs, bandpass)

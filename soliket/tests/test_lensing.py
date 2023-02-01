import os
import tempfile

import pytest
import numpy as np

from cobaya.yaml import yaml_load
from cobaya.model import get_model

try:
    import classy  # noqa F401
except ImportError:
    boltzmann_codes = ["camb"]
else:
    boltzmann_codes = ["camb", "classy"]

packages_path = os.environ.get("COBAYA_PACKAGES_PATH") or os.path.join(
    tempfile.gettempdir(), "lensing_packages"
)


def get_demo_lensing_model(theory):
    if theory == "camb":
        info_yaml = r"""
        likelihood:
            soliket.LensingLikelihood:
                stop_at_error: True

        theory:
            camb:
                extra_args:
                    lens_potential_accuracy: 1

        params:
            ns:
                prior:
                  min: 0.8
                  max: 1.2
            H0:
                prior:
                  min: 40
                  max: 100
        """
    elif theory == "classy":
        info_yaml = r"""
        likelihood:
            soliket.LensingLikelihood:
                stop_at_error: True

        theory:
            classy:
                extra_args:
                    output: lCl, tCl
                path: global

        params:
            n_s:
                prior:
                  min: 0.8
                  max: 1.2
            H0:
                prior:
                  min: 40
                  max: 100

        """

    info = yaml_load(info_yaml)

    from cobaya.install import install
    install(info, path=packages_path, skip_global=True)

    test_point = {}
    for par, pdict in info["params"].items():
        if not isinstance(pdict, dict):
            continue

        if "ref" in pdict:
            try:
                value = float(pdict["ref"])
            except TypeError:
                value = (pdict["ref"]["min"] + pdict["ref"]["max"]) / 2
            test_point[par] = value
        elif "prior" in pdict:
            value = (pdict["prior"]["min"] + pdict["prior"]["max"]) / 2
            test_point[par] = value

    model = get_model(info)
    return model, test_point


@pytest.mark.parametrize("theory", boltzmann_codes)
def test_lensing(theory):
    model, test_point = get_demo_lensing_model(theory)
    lnl = model.loglike(test_point)[0]

    assert np.isfinite(lnl)

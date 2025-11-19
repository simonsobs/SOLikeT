import numpy as np
from cobaya.likelihood import Likelihood

from soliket.utils import OneWithCls, binner, get_likelihood


def naive_binner(bmin, bmax, x, tobin):
    binned = list()
    bcent = list()
    # All but the last bins are open to the right
    for bm, bmx in zip(bmin[:-1], bmax[:-1]):
        bcent.append(0.5 * (bmx + bm))
        binned.append(np.mean(tobin[np.where((x >= bm) & (x < bmx))[0]]))
    # The last bin is closed to the right
    bcent.append(0.5 * (bmax[-1] + bmin[-1]))
    binned.append(np.mean(tobin[np.where((x >= bmin[-1]) & (x <= bmax[-1]))[0]]))

    return np.array(bcent), np.array(binned)


def test_binning():
    # bmin = np.arange(10, step=3)
    # bmax = np.array([2, 5, 8, 12])
    binedge = np.arange(13, step=3)
    bmin = binedge[:-1]
    bmax = binedge[1:]
    ell = np.arange(13)
    cell = np.arange(13)

    centers_test, values_test = naive_binner(bmin, bmax, ell, cell)

    bincent, binval = binner(ell, cell, binedge)

    assert np.allclose(bincent, centers_test)
    assert np.allclose(binval, values_test)

def test_get_likelihood_and_onewithcls(tmp_path, monkeypatch):
    # create a dummy module with a Likelihood subclass
    module_code = """
from cobaya.likelihood import Likelihood

class DummyLikelihood(Likelihood):
    def __init__(self, options=None):
        super().__init__()
        self.options = options or {}

    def initialize(self):
        pass

    def get_requirements(self):
        return {}
"""
    mod_path = tmp_path / "modtest.py"
    mod_path.write_text(module_code)

    # insert tmp_path into sys.path and import by name
    import sys

    sys.path.insert(0, str(tmp_path))

    try:
        ll = get_likelihood("modtest.DummyLikelihood", options={"a": 1})
        assert isinstance(ll, Likelihood)
        assert ll.options["a"] == 1

        # test OneWithCls returns expected requirements
        one = OneWithCls({})
        req = one.get_requirements()
        assert "Cl" in req
        assert all(k in req["Cl"] for k in ["pp", "tt", "te", "ee", "bb"])
    finally:
        sys.path.pop(0)


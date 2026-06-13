"""Unit tests for the decoupled lensing N0/N1 bias correction.

These exercise ``LensingCorrections.compute`` in isolation -- no cobaya, no data
download -- using small hand-computable inputs.
"""

import numpy as np

from soliket.lensing import LensingCorrections

_SPECS = ("tt", "te", "ee", "bb")


def _corrections(
    n,
    *,
    fiducial=None,
    n0_response=None,
    n1_response=None,
    n1_clpp=None,
    thetaclkk=None,
    n0=None,
):
    """A LensingCorrections defaulting to zeros, with the given pieces overridden.

    ``fiducial``/``n0_response``/``n1_response`` are merged onto all-zero defaults
    so a test only specifies the spectra it cares about.
    """
    fid = {k: np.zeros(n) for k in _SPECS}
    fid.update(fiducial or {})
    n0r = {k: np.zeros((n, n)) for k in _SPECS}
    n0r.update(n0_response or {})
    n1r = {k: np.zeros((n, n)) for k in _SPECS}
    n1r.update(n1_response or {})
    return LensingCorrections(
        fiducial=fid,
        n0_response=n0r,
        n1_response=n1r,
        n1_clpp=np.zeros((n, n)) if n1_clpp is None else n1_clpp,
        thetaclkk=np.zeros(n) if thetaclkk is None else thetaclkk,
        n0=np.ones(n) if n0 is None else n0,
    )


def _cls(n, **overrides):
    cls = {k: np.zeros(n) for k in _SPECS}
    cls.update(overrides)
    return cls


def test_n0_term_scaling_and_response():
    """The N0 response is scaled by 2 * thetaclkk / n0 applied to (Cl - fiducial)."""
    n = 3
    corr = _corrections(
        n,
        fiducial={"tt": np.array([1.0, 1.0, 1.0])},
        thetaclkk=np.array([2.0, 2.0, 2.0]),
        n0=np.array([1.0, 1.0, 1.0]),
        n0_response={"tt": np.eye(n)},
    )
    # delta_tt = [1, 2, 3]; scale = 2 * 2 / 1 = 4
    out = corr.compute(
        cls=_cls(n, tt=np.array([2.0, 3.0, 4.0])),
        clkk_theo=np.zeros(n),
        binning_matrix=np.eye(n),
    )

    np.testing.assert_allclose(out, [4.0, 8.0, 12.0])


def test_n1_clpp_term():
    """The N1 lensing term applies n1_clpp to (Clkk_theo - thetaclkk) with no scaling."""
    n = 3
    corr = _corrections(
        n,
        thetaclkk=np.array([1.0, 1.0, 1.0]),
        n0=np.array([1.0, 1.0, 1.0]),
        n1_clpp=np.eye(n),
    )
    clkk_theo = np.array([3.0, 4.0, 5.0])  # delta = [2, 3, 4]

    out = corr.compute(
        cls=_cls(n),
        clkk_theo=clkk_theo,
        binning_matrix=np.eye(n),
    )

    np.testing.assert_allclose(out, [2.0, 3.0, 4.0])


def test_binning_matrix_is_applied_last():
    """The full correction vector is projected through the binning matrix."""
    n = 3
    corr = _corrections(
        n,
        fiducial={"tt": np.array([1.0, 1.0, 1.0])},
        thetaclkk=np.array([2.0, 2.0, 2.0]),
        n0=np.array([1.0, 1.0, 1.0]),
        n0_response={"tt": np.eye(n)},
    )
    binning = np.array([[1.0, 1.0, 0.0], [0.0, 0.0, 1.0]])  # 2 bins x 3

    out = corr.compute(
        cls=_cls(n, tt=np.array([2.0, 3.0, 4.0])),  # unbinned correction = [4, 8, 12]
        clkk_theo=np.zeros(n),
        binning_matrix=binning,
    )

    np.testing.assert_allclose(out, [12.0, 12.0])  # [4+8, 12]

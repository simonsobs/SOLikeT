"""Tests for soliket.sacc_tools dataset-creation utilities."""

import os

import numpy as np
import pytest
import sacc
from cobaya.tools import resolve_packages_path

from soliket.sacc_tools import (
    gaussian_covariance,
    smooth_mflike_sacc,
    smooth_twin_sacc,
    top_hat_windows,
)


def _mflike_data_available():
    path = resolve_packages_path()
    return bool(path) and os.path.isfile(
        os.path.join(path, "data", "MFLike", "v0.8", "LAT_simu_sacc_00044.fits")
    )


def test_top_hat_windows_shapes_and_partition():
    ells, window = top_hat_windows(ell_max=600, n_bins=20)

    assert ells.shape == (20,)
    weights = window.weight  # (n_support, n_bins)
    assert weights.shape[1] == 20
    assert weights.shape[0] == 601  # every multipole 0..ell_max has a support slot
    # top-hat partition: every multipole, including ell_max, lands in exactly one bin
    covered = weights.sum(axis=1)
    np.testing.assert_array_equal(covered, np.ones(601))


def test_top_hat_windows_covers_tail_when_not_divisible():
    # ell_max + 1 = 101 is not divisible by 3; the old floor-division binning left
    # the high-ell tail (and ell_max itself) unbinned. The partition must be total.
    ells, window = top_hat_windows(ell_max=100, n_bins=3)

    assert ells.shape == (3,)
    covered = window.weight.sum(axis=1)
    np.testing.assert_array_equal(covered, np.ones(101))


def test_gaussian_covariance_is_symmetric_positive_diagonal():
    n_maps, n_ell = 2, 5
    ells = (np.arange(n_ell) + 0.5) * 30
    rng = np.random.default_rng(0)
    cls = rng.random((n_maps, n_maps, n_ell)) + 1.0
    cls = (cls + cls.transpose(1, 0, 2)) / 2  # symmetric in the map indices

    cov = gaussian_covariance(cls, ells, delta_ell=30, fsky=0.4)

    n_cross = n_maps * (n_maps + 1) // 2
    assert cov.shape == (n_cross * n_ell, n_cross * n_ell)
    np.testing.assert_allclose(cov, cov.T)
    assert np.all(np.diag(cov) > 0)


def test_gaussian_covariance_scales_inversely_with_fsky():
    n_maps, n_ell = 2, 4
    ells = (np.arange(n_ell) + 0.5) * 30
    rng = np.random.default_rng(1)
    cls = rng.random((n_maps, n_maps, n_ell)) + 1.0
    cls = (cls + cls.transpose(1, 0, 2)) / 2

    half = gaussian_covariance(cls, ells, 30, fsky=0.4)
    quarter = gaussian_covariance(cls, ells, 30, fsky=0.2)

    np.testing.assert_allclose(quarter, 2 * half)


def _tiny_lensing_sacc(values):
    """A one-spectrum (cl_00, ck x ck) SACC with windows and a covariance."""
    n = len(values)
    s = sacc.Sacc()
    s.add_tracer("Map", "ck", quantity="cmb_convergence", spin=0,
                 ell=np.arange(100), beam=np.ones(100))
    support = np.arange(2, 2 + 3 * n)
    weight = np.zeros((len(support), n))
    for b, idx in enumerate(np.array_split(np.arange(len(support)), n)):
        weight[idx, b] = 1.0 / len(idx)
    s.add_ell_cl("cl_00", "ck", "ck", np.arange(n), values,
                 window=sacc.BandpowerWindow(support, weight))
    s.add_covariance(np.diag(np.full(n, 4.0)))
    return s


def test_smooth_twin_sacc_replaces_mean_and_reuses_windows_and_cov(tmp_path):
    # A smooth twin keeps the source tracers/windows/covariance but swaps the
    # measured bandpowers for the theory we pass in.
    src = _tiny_lensing_sacc(np.array([1.0, 2.0, 3.0, 4.0]))
    theory = np.array([10.0, 20.0, 30.0, 40.0])
    out = tmp_path / "twin.fits"

    twin = smooth_twin_sacc(src, "cl_00", "ck", "ck", theory, out_path=out)

    np.testing.assert_array_equal(twin.mean, theory)            # data == theory
    np.testing.assert_array_equal(twin.covariance.covmat, src.covariance.covmat)
    assert "ck" in twin.tracers                                 # tracer carried over
    reloaded = sacc.Sacc.load_fits(str(out))
    np.testing.assert_array_equal(reloaded.mean, theory)        # survives round-trip
    assert reloaded.get_bandpower_windows(np.arange(len(theory))) is not None


def _numeric_fiducial(info):
    """Flat {name: value} of the numeric fixed fiducial params (skip lambdas)."""
    return {
        name: spec["value"]
        for name, spec in info["params"].items()
        if isinstance(spec, dict)
        and "value" in spec
        and not isinstance(spec["value"], str)
    }


@pytest.mark.skipif(not _mflike_data_available(), reason="MFLike data not installed")
def test_smooth_mflike_sacc_gives_zero_chi2_round_trip(tmp_path, check_skip_mflike):
    # A smooth MFLike dataset is theory binned through MFLike's own windows; the
    # MFLike likelihood evaluated on it at the same fiducial must give chi^2 = 0.
    from cobaya.model import get_model

    from soliket.presets import build_info, resolve_aliases

    info = build_info("mflike")
    info["packages_path"] = resolve_packages_path()
    model = get_model(info)
    roles = resolve_aliases(model)
    params = _numeric_fiducial(info)
    model.loglikes(params)  # populate the provider at the fiducial point

    dls = model.provider.get_Cl(ell_factor=True)
    fg_totals = roles.foreground.get_fg_totals()
    out = tmp_path / "data_sacc_smooth.fits"

    sacc_obj = smooth_mflike_sacc(roles.mflike, dls, fg_totals, params, out_path=out)

    assert out.is_file()
    assert len(sacc_obj.mean) > 0

    # Rebuild MFLike on the smooth data (absolute input_file; reuse shipped cov/Bbl).
    info2 = build_info("mflike")
    info2["packages_path"] = resolve_packages_path()
    info2["likelihood"]["mflike.TTTEEE"]["input_file"] = str(out)
    model2 = get_model(info2)
    mflike2 = model2.likelihood["mflike.TTTEEE"]
    loglike = float(model2.loglikes(params)[0].sum())
    chi2 = -2 * (loglike - mflike2.logp_const)

    assert abs(chi2) < 1e-6

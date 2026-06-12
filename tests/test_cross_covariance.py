"""Tests for soliket.cross_covariance kernels.

The kernels are pure: they take the CAMB lensing derivative and fiducial spectra
as arrays and return covariance blocks. These structural tests pin shape,
fsky-scaling and spectrum-scaling without a (slow) CAMB recompute; an end-to-end
regression against the committed reference products lives separately and is
gated on data availability.
"""

import os
from types import SimpleNamespace

import numpy as np
import pytest

from soliket.cross_covariance import (
    camb_lensing_derivatives,
    cmb_combs_from_sacc,
    cmb_combs_from_spec_meta,
    cmb_lensing_block,
    lensing_induced_block,
    lensing_induced_cov,
    n1_crosscov_block,
    shear_kappa_block,
)


def _mflike_cov_sacc_path():
    from cobaya.tools import resolve_packages_path

    path = resolve_packages_path()
    if not path:
        return None
    f = os.path.join(path, "data", "MFLike", "v0.8", "data_sacc_w_covar_and_Bbl.fits")
    return f if os.path.isfile(f) else None


def _tiny_mflike_sacc():
    """A minimal MFLike-like SACC: one TT spectrum with bandpower windows."""
    import sacc

    s = sacc.Sacc()
    s.add_tracer("Misc", "LAT_93_s0", quantity="cmb_temperature", spin=0)
    support = np.arange(2, 30)
    n_bins = 3
    weight = np.zeros((len(support), n_bins))
    centers = []
    for b, idx in enumerate(np.array_split(np.arange(len(support)), n_bins)):
        weight[idx, b] = 1.0 / len(idx)
        centers.append(support[idx].mean())
    window = sacc.BandpowerWindow(support, weight)
    s.add_ell_cl(
        "cl_00",
        "LAT_93_s0",
        "LAT_93_s0",
        np.array(centers),
        np.zeros(n_bins),
        window=window,
    )
    s.metadata["f_sky_LAT"] = 0.4
    s.metadata["cosmo_params"] = repr(
        {
            "cosmomc_theta": 0.0104,
            "logA": 3.05,
            "ombh2": 0.0224,
            "omch2": 0.1202,
            "ns": 0.9649,
            "Alens": 1.0,
            "tau": 0.0544,
        }
    )
    s.metadata["accuracy_params"] = repr({})
    s.metadata["lmax"] = 60
    return s


def test_cmb_lensing_crosscov_runs_end_to_end_on_tiny_data():
    """The full glue (extract -> CAMB derivative -> kernel) at small lmax.

    Confirms the code path works without the full-accuracy (multi-GB) derivative;
    exact reproduction of the published reference is a separate, heavy check.
    """
    from soliket.cross_covariance import cmb_lensing_crosscov

    lmax_kk = 25
    binning = np.zeros((2, lmax_kk))
    binning[0, 2:12] = 0.1
    binning[1, 12:lmax_kk] = 0.1
    lensing = SimpleNamespace(
        binning_matrix=binning,
        provider=SimpleNamespace(
            get_Cl=lambda ell_factor=True: {"pp": np.ones(lmax_kk) * 1e-8}
        ),
    )

    block = cmb_lensing_crosscov(_tiny_mflike_sacc(), lensing)

    assert block.shape == (3, 2)  # 3 CMB bins x 2 kappa bins
    assert np.all(np.isfinite(block))


def test_shared_derivatives_match_per_block_computation():
    """A precomputed derivative bundle must reproduce the per-block CAMB run.

    The joint covariance shares one ``camb_lensing_derivatives`` run across blocks
    via ``derivatives=``; passing the bundle must give bit-identical results to
    letting each builder compute its own (the redundant-CAMB path).
    """
    from soliket.cross_covariance import (
        camb_lensing_derivatives_from_sacc,
        cmb_lensing_crosscov,
        lensing_induced_cov,
    )

    lmax_kk = 25
    binning = np.zeros((2, lmax_kk))
    binning[0, 2:12] = 0.1
    binning[1, 12:lmax_kk] = 0.1
    lensing = SimpleNamespace(
        binning_matrix=binning,
        provider=SimpleNamespace(
            get_Cl=lambda ell_factor=True: {"pp": np.ones(lmax_kk) * 1e-8}
        ),
    )

    sacc_data = _tiny_mflike_sacc()
    derivs = camb_lensing_derivatives_from_sacc(sacc_data)

    np.testing.assert_array_equal(
        cmb_lensing_crosscov(sacc_data, lensing, derivatives=derivs),
        cmb_lensing_crosscov(sacc_data, lensing),
    )
    np.testing.assert_array_equal(
        lensing_induced_cov(sacc_data, derivatives=derivs),
        lensing_induced_cov(sacc_data),
    )


def test_from_cmb_lensing_roundtrips_into_multigaussian(tmp_path):
    """``CrossCov.from_cmb_lensing`` must produce a file that loads back usably.

    Two regressions are pinned: the cross block has to be keyed by the components'
    real ``GaussianData`` names (else ``MultiGaussianData`` silently drops it), and
    the component auto-covariances have to be carried (else the saved file's zero
    auto-blocks make the joint covariance singular on load).
    """
    from soliket.gaussian.gaussian_data import (
        CrossCov,
        GaussianData,
        MultiGaussianData,
    )

    sacc_data = _tiny_mflike_sacc()
    sacc_path = tmp_path / "mflike_cov.fits"
    sacc_data.save_fits(str(sacc_path), overwrite=True)

    n_cmb, n_kk = 3, 2
    # auto-cov labels (x) must be the SACC's bandpower ells: from_cmb_lensing checks
    # the cross block lines up with the auto-cov by bandpower identity.
    cmb_ell = sacc_data.get_ell_cl("cl_00", "LAT_93_s0", "LAT_93_s0")[0]
    mf_data = GaussianData("mflike", cmb_ell, np.zeros(n_cmb), np.eye(n_cmb))
    le_data = GaussianData(
        "CMB Lensing", np.arange(n_kk), np.zeros(n_kk), np.eye(n_kk) * 1e-15
    )

    lmax_kk = 25
    binning = np.zeros((n_kk, lmax_kk))
    binning[0, 2:12] = 0.1
    binning[1, 12:lmax_kk] = 0.1
    lensing = SimpleNamespace(
        binning_matrix=binning,
        provider=SimpleNamespace(
            get_Cl=lambda ell_factor=True: {"pp": np.ones(lmax_kk) * 1e-8}
        ),
        _get_gauss_data=lambda: le_data,
    )
    mflike = SimpleNamespace(
        data_folder=str(tmp_path),
        cov_Bbl_file="mflike_cov.fits",
        input_file="mflike_cov.fits",
        _get_gauss_data=lambda: mf_data,
    )
    xcov = CrossCov.from_cmb_lensing(mflike, lensing)

    # cross block keyed by the real component names, with its auto-covariances
    assert ("mflike", "CMB Lensing") in xcov
    assert xcov[("mflike", "CMB Lensing")].shape == (n_cmb, n_kk)
    np.testing.assert_array_equal(xcov[("mflike", "mflike")], mf_data.cov)
    np.testing.assert_array_equal(xcov[("CMB Lensing", "CMB Lensing")], le_data.cov)

    # round-trips through SACC and into a non-singular joint covariance
    out = tmp_path / "xcov.fits"
    xcov.save(str(out))
    multi = MultiGaussianData([mf_data, le_data], CrossCov.load(str(out)))
    assert np.all(np.isfinite(multi.inv_cov))


def test_from_cmb_lensing_trims_block_to_used_bandpowers(tmp_path):
    """The cross block spans every SACC bandpower, but the likelihood uses only the
    bins surviving its scale cuts. ``from_cmb_lensing`` must trim the block to those
    used bins (via each component's ``indices``) so it matches the auto-covariances
    and ``save`` does not raise a broadcast error.
    """
    from soliket.gaussian.gaussian_data import (
        CrossCov,
        GaussianData,
        MultiGaussianData,
    )

    sacc_data = _tiny_mflike_sacc()  # 3 CMB bandpowers
    sacc_data.save_fits(str(tmp_path / "mflike_cov.fits"), overwrite=True)

    # MFLike keeps only 2 of the 3 bandpowers; ``indices`` maps full -> used.
    used = np.array([True, True, False])
    n_used, n_kk = 2, 2
    cmb_ell = sacc_data.get_ell_cl("cl_00", "LAT_93_s0", "LAT_93_s0")[0]
    mf_data = GaussianData(
        "mflike", cmb_ell[used], np.zeros(n_used), np.eye(n_used), indices=used
    )
    le_data = GaussianData(
        "CMB Lensing", np.arange(n_kk), np.zeros(n_kk), np.eye(n_kk) * 1e-15
    )

    lmax_kk = 25
    binning = np.zeros((n_kk, lmax_kk))
    binning[0, 2:12] = 0.1
    binning[1, 12:lmax_kk] = 0.1
    lensing = SimpleNamespace(
        binning_matrix=binning,
        provider=SimpleNamespace(
            get_Cl=lambda ell_factor=True: {"pp": np.ones(lmax_kk) * 1e-8}
        ),
        _get_gauss_data=lambda: le_data,
    )
    mflike = SimpleNamespace(
        data_folder=str(tmp_path),
        cov_Bbl_file="mflike_cov.fits",
        input_file="mflike_cov.fits",
        _get_gauss_data=lambda: mf_data,
    )
    xcov = CrossCov.from_cmb_lensing(mflike, lensing)

    # block trimmed from the 3 SACC bandpowers down to the 2 the likelihood uses
    assert xcov[("mflike", "CMB Lensing")].shape == (n_used, n_kk)

    out = tmp_path / "xcov.fits"
    xcov.save(str(out))  # must not raise a broadcast error
    multi = MultiGaussianData([mf_data, le_data], CrossCov.load(str(out)))
    assert np.all(np.isfinite(multi.inv_cov))


def test_from_cmb_lensing_rejects_auto_cov_order_mismatch(tmp_path):
    # Robustness guard: if mflike's auto-covariance is in a different bandpower order
    # than the SACC's natural order (a reordered cov_Bbl file, or TE/ET
    # symmetrization), the positional trim would silently mis-order the cross-cov.
    # from_cmb_lensing must detect the bandpower-ell mismatch and refuse.
    from soliket.gaussian.gaussian_data import CrossCov, GaussianData

    sacc_data = _tiny_mflike_sacc()
    sacc_data.save_fits(str(tmp_path / "mflike_cov.fits"), overwrite=True)

    n_cmb, n_kk = 3, 2
    cmb_ell = sacc_data.get_ell_cl("cl_00", "LAT_93_s0", "LAT_93_s0")[0]
    # auto-cov labelled in a DIFFERENT order than the SACC's natural bandpowers
    mf_data = GaussianData("mflike", cmb_ell[::-1], np.zeros(n_cmb), np.eye(n_cmb))
    le_data = GaussianData(
        "CMB Lensing", np.arange(n_kk), np.zeros(n_kk), np.eye(n_kk) * 1e-15
    )

    lmax_kk = 25
    binning = np.zeros((n_kk, lmax_kk))
    binning[0, 2:12] = 0.1
    binning[1, 12:lmax_kk] = 0.1
    lensing = SimpleNamespace(
        binning_matrix=binning,
        provider=SimpleNamespace(
            get_Cl=lambda ell_factor=True: {"pp": np.ones(lmax_kk) * 1e-8}
        ),
        _get_gauss_data=lambda: le_data,
    )
    mflike = SimpleNamespace(
        data_folder=str(tmp_path),
        cov_Bbl_file="mflike_cov.fits",
        input_file="mflike_cov.fits",
        _get_gauss_data=lambda: mf_data,
    )
    with pytest.raises(ValueError, match="row order"):
        CrossCov.from_cmb_lensing(mflike, lensing)


def test_cmb_combs_from_spec_meta_follows_spec_meta_order_and_pol():
    # The block builder driven by spec_meta yields one triple per spectrum, in
    # spec_meta (= auto-cov) order, mapping pol -> CAMB row and passing the window
    # through unchanged. This is what aligns the cross-cov with the auto-cov.
    import sacc

    support = np.arange(2, 8)
    weight = np.ones((len(support), 2)) / len(support)
    bpw = sacc.BandpowerWindow(support, weight)
    # deliberately TE before TT (an order no get_tracer_combinations() would invent)
    spec_meta = [
        {"pol": "te", "bpw": bpw, "leff": np.array([3.0, 6.0]), "ids": np.array([0, 1])},
        {"pol": "tt", "bpw": bpw, "leff": np.array([3.0, 6.0]), "ids": np.array([2, 3])},
    ]

    combs = cmb_combs_from_spec_meta(spec_meta)

    assert [ind_camb for ind_camb, _, _ in combs] == [3, 0]  # te->3, tt->0, IN order
    for (_, support_out, weight_out), m in zip(combs, spec_meta):
        np.testing.assert_array_equal(support_out, m["bpw"].values)
        assert weight_out.shape == (m["bpw"].weight.shape[1], m["bpw"].weight.shape[0])


def test_from_cmb_lensing_uses_spec_meta_when_available(tmp_path):
    # When the likelihood exposes spec_meta, the block is built from it (MFLike's own
    # order/windows) and needs no positional row trim -- aligned with the auto-cov by
    # construction. We check the path runs end-to-end and is shaped from spec_meta.
    from soliket.gaussian.gaussian_data import CrossCov, GaussianData

    sacc_data = _tiny_mflike_sacc()
    sacc_data.save_fits(str(tmp_path / "mflike_cov.fits"), overwrite=True)
    ell, _, ind = sacc_data.get_ell_cl(
        "cl_00", "LAT_93_s0", "LAT_93_s0", return_ind=True
    )
    bpw = sacc_data.get_bandpower_windows(ind)
    spec_meta = [{"pol": "tt", "bpw": bpw, "leff": ell, "ids": np.arange(len(ell))}]

    n_cmb, n_kk = len(ell), 2
    mf_data = GaussianData("mflike", ell, np.zeros(n_cmb), np.eye(n_cmb))
    le_data = GaussianData(
        "CMB Lensing", np.arange(n_kk), np.zeros(n_kk), np.eye(n_kk) * 1e-15
    )
    lmax_kk = 25
    binning = np.zeros((n_kk, lmax_kk))
    binning[0, 2:12] = 0.1
    binning[1, 12:lmax_kk] = 0.1
    lensing = SimpleNamespace(
        binning_matrix=binning,
        provider=SimpleNamespace(
            get_Cl=lambda ell_factor=True: {"pp": np.ones(lmax_kk) * 1e-8}
        ),
        _get_gauss_data=lambda: le_data,
    )
    mflike = SimpleNamespace(
        data_folder=str(tmp_path),
        cov_Bbl_file="mflike_cov.fits",
        input_file="mflike_cov.fits",
        spec_meta=spec_meta,
        _get_gauss_data=lambda: mf_data,
    )

    xcov = CrossCov.from_cmb_lensing(mflike, lensing)
    block = xcov[("mflike", "CMB Lensing")]

    assert block.shape == (n_cmb, n_kk)  # rows from spec_meta, no positional row trim
    assert np.all(np.isfinite(block))


@pytest.mark.skipif(_mflike_cov_sacc_path() is None, reason="MFLike data not installed")
def test_cmb_combs_from_sacc_extracts_well_formed_triples():
    import sacc

    s = sacc.Sacc.load_fits(_mflike_cov_sacc_path())
    combs = cmb_combs_from_sacc(s)

    assert len(combs) == len(s.get_tracer_combinations())
    for ind_camb, support, weight in combs:
        assert ind_camb in (0, 1, 3)  # TT, EE, TE
        assert support.ndim == 1
        assert weight.shape[1] == support.shape[0]  # weight is (n_bins, n_support)


def _tt_sacc(*, interleave):
    """Two TT spectra (two channels). With ``interleave=True`` the first pair's
    bandpowers are split across two ``add_ell_cl`` calls with the second pair in
    between, so its data points are non-contiguous in the flat data vector."""
    import sacc

    s = sacc.Sacc()
    for name in ("LAT_93_s0", "LAT_145_s0"):
        s.add_tracer("Misc", name, quantity="cmb_temperature", spin=0)
    support = np.arange(2, 8)
    weight = np.ones((len(support), 2)) / len(support)
    bpw = sacc.BandpowerWindow(support, weight)
    lo, hi = np.array([3.0, 6.0]), np.array([9.0, 12.0])
    if interleave:
        s.add_ell_cl("cl_00", "LAT_93_s0", "LAT_93_s0", lo, np.zeros(2), window=bpw)
        s.add_ell_cl("cl_00", "LAT_145_s0", "LAT_145_s0", lo, np.zeros(2), window=bpw)
        s.add_ell_cl("cl_00", "LAT_93_s0", "LAT_93_s0", hi, np.zeros(2), window=bpw)
    else:
        for t in ("LAT_93_s0", "LAT_145_s0"):
            s.add_ell_cl("cl_00", t, t, lo, np.zeros(2), window=bpw)
    return s


def test_cmb_combs_from_sacc_accepts_contiguous_multi_spectrum():
    # Two contiguously-stored spectra: per-combination order equals the natural
    # flat order, so the builder accepts it and yields one triple per spectrum.
    combs = cmb_combs_from_sacc(_tt_sacc(interleave=False))

    assert len(combs) == 2
    assert [ind_camb for ind_camb, _, _ in combs] == [0, 0]  # both TT -> CAMB row 0


def test_cmb_combs_from_sacc_rejects_non_contiguous_spectra():
    # A spectrum stored non-contiguously breaks the assumption that the block rows
    # follow the SACC's natural bandpower order, so the cross-covariance would be
    # silently mis-ordered. The builder must reject it loudly.
    with pytest.raises(ValueError, match="contiguous"):
        cmb_combs_from_sacc(_tt_sacc(interleave=True))




def test_camb_lensing_derivatives_returns_consistent_shapes():
    cosmo = {
        "H0": 67.7,
        "ombh2": 0.0224,
        "omch2": 0.1202,
        "ns": 0.9649,
        "tau": 0.0544,
        "As": 2.1e-9,
    }
    lmax = 150

    cls, clp, dCllens = camb_lensing_derivatives(cosmo, accuracy={}, lmax=lmax)

    assert cls.shape[0] == lmax + 1
    assert clp.shape[0] == lmax + 1
    assert dCllens.shape[0] == 4  # TT, EE, BB, TE
    assert dCllens.shape[1] == lmax + 1


def _toy_cmb_combs():
    # one CMB tracer combination (TT), 1 bandpower bin over ell support {2,3,4}
    return [(0, np.array([2, 3, 4]), np.ones((1, 3)))]


def test_cmb_lensing_block_shape():
    L, lmax_kk = 8, 5
    rng = np.random.default_rng(0)
    dCllens = rng.random((4, L, L))
    clp = rng.random(L) + 1.0
    cl_kk = rng.random(lmax_kk) + 1.0
    binning = np.zeros((2, lmax_kk))
    binning[0, 2] = 1.0
    binning[1, 3] = 1.0

    block = cmb_lensing_block(dCllens, clp, cl_kk, 0.5, _toy_cmb_combs(), binning)

    assert block.shape == (1, 2)  # (n_cmb_data, n_kk_bins)


def test_cmb_lensing_block_scales_inversely_with_fsky():
    L, lmax_kk = 8, 5
    rng = np.random.default_rng(1)
    dCllens = rng.random((4, L, L))
    clp = rng.random(L) + 1.0
    cl_kk = rng.random(lmax_kk) + 1.0
    binning = np.zeros((2, lmax_kk))
    binning[0, 2] = 1.0
    binning[1, 3] = 1.0
    combs = _toy_cmb_combs()

    half = cmb_lensing_block(dCllens, clp, cl_kk, 0.5, combs, binning)
    quarter = cmb_lensing_block(dCllens, clp, cl_kk, 0.25, combs, binning)

    np.testing.assert_allclose(quarter, 2 * half)


def test_cmb_lensing_block_scales_with_cl_kk_squared():
    L, lmax_kk = 8, 5
    rng = np.random.default_rng(2)
    dCllens = rng.random((4, L, L))
    clp = rng.random(L) + 1.0
    cl_kk = rng.random(lmax_kk) + 1.0
    binning = np.zeros((2, lmax_kk))
    binning[0, 2] = 1.0
    binning[1, 3] = 1.0
    combs = _toy_cmb_combs()

    base = cmb_lensing_block(dCllens, clp, cl_kk, 0.5, combs, binning)
    doubled = cmb_lensing_block(dCllens, clp, 2 * cl_kk, 0.5, combs, binning)

    np.testing.assert_allclose(doubled, 4 * base)


def _two_cmb_combs():
    return [
        (0, np.array([2, 3, 4]), np.ones((2, 3))),  # TT, 2 bins
        (1, np.array([3, 4, 5]), np.ones((2, 3))),  # EE, 2 bins
    ]


def test_lensing_induced_block_shape_and_symmetry():
    L = 8
    dCllens = np.random.default_rng(3).random((4, L, L))

    cov = lensing_induced_block(dCllens, 0.5, _two_cmb_combs())

    assert cov.shape == (4, 4)  # 2 combs x 2 bins each
    np.testing.assert_allclose(cov, cov.T)


def test_lensing_induced_block_scales_inversely_with_fsky():
    L = 8
    dCllens = np.random.default_rng(4).random((4, L, L))
    combs = _two_cmb_combs()

    half = lensing_induced_block(dCllens, 0.5, combs)
    quarter = lensing_induced_block(dCllens, 0.25, combs)

    np.testing.assert_allclose(quarter, 2 * half)


def test_n1_crosscov_block_shape_and_fsky_scaling():
    L, n_ell, lmax_kk = 8, 6, 5
    rng = np.random.default_rng(7)
    dCllens = rng.random((4, L, L))
    clp = rng.random(L) + 1.0
    n1_mat = rng.random((lmax_kk, n_ell))
    binning = np.zeros((2, lmax_kk))
    binning[0, 2], binning[1, 3] = 1.0, 1.0
    combs = [(0, np.array([2, 3, 4]), np.ones((1, 3)))]

    half = n1_crosscov_block(dCllens, clp, n1_mat, 0.5, combs, binning)
    quarter = n1_crosscov_block(dCllens, clp, n1_mat, 0.25, combs, binning)

    assert half.shape == (1, 2)
    np.testing.assert_allclose(quarter, 2 * half)


def test_n1_crosscov_block_pins_ell_weighting():
    # With uniform dCllens/clp, an N1 matrix nonzero at a single multipole j
    # isolates factor[j] = 2/(2j+1)/fsky * 2pi/(j(j+1))**2, so the ratio between
    # two columns pins the bespoke ell weighting exactly.
    L, n_ell, lmax_kk = 10, 8, 4
    dCllens = np.ones((4, L, L))
    clp = np.ones(L)
    binning = np.zeros((1, lmax_kk))
    binning[0, 1] = 1.0
    combs = [(0, np.array([2]), np.ones((1, 1)))]

    def block_at_column(j):
        n1 = np.zeros((lmax_kk, n_ell))
        n1[1, j] = 1.0
        return n1_crosscov_block(dCllens, clp, n1, 0.5, combs, binning)[0, 0]

    ratio = block_at_column(2) / block_at_column(3)
    expected = (2 / 5 * 2 * np.pi / (2 * 3) ** 2) / (2 / 7 * 2 * np.pi / (3 * 4) ** 2)
    np.testing.assert_allclose(ratio, expected)


def test_shear_kappa_block_concatenates_per_tracer_cmb_lensing_blocks():
    L, lmax_lss = 8, 5
    rng = np.random.default_rng(5)
    dCllens = rng.random((4, L, L))
    clp = rng.random(L) + 1.0
    cl1 = rng.random(lmax_lss) + 1.0
    cl2 = rng.random(lmax_lss) + 1.0
    bin1 = np.zeros((2, lmax_lss))
    bin1[0, 2], bin1[1, 3] = 1.0, 1.0
    bin2 = np.zeros((3, lmax_lss))
    bin2[0, 2], bin2[1, 3], bin2[2, 4] = 1.0, 1.0, 1.0
    combs = [(0, np.array([2, 3, 4]), np.ones((1, 3)))]

    block = shear_kappa_block(dCllens, clp, [cl1, cl2], [bin1, bin2], 0.5, combs)

    # one CMB datum, columns = tracer1 bins (2) + tracer2 bins (3)
    assert block.shape == (1, 5)
    expected = np.hstack(
        [
            cmb_lensing_block(dCllens, clp, cl1, 0.5, combs, bin1),
            cmb_lensing_block(dCllens, clp, cl2, 0.5, combs, bin2),
        ]
    )
    np.testing.assert_allclose(block, expected)


def test_lensing_induced_cov_honours_explicit_combs():
    # The combs= override must drive the block, so a caller can pass
    # cmb_combs_from_spec_meta to align it with the MFLike auto-covariance order
    # instead of the SACC's tracer-combination order (the from_cmb_lensing fix,
    # applied to the other builders too).
    sacc_data = _tiny_mflike_sacc()  # provides f_sky_LAT metadata
    fsky = float(sacc_data.metadata["f_sky_LAT"])
    L = 8
    dCllens = np.random.default_rng(0).random((4, L, L))
    combs = _two_cmb_combs()  # support max 5 < L

    out = lensing_induced_cov(
        sacc_data, derivatives=(None, None, dCllens), combs=combs
    )

    np.testing.assert_allclose(out, lensing_induced_block(dCllens, fsky, combs))
    assert out.shape == (4, 4)  # 2 combs x 2 bins, from the given combs not the SACC

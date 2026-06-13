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


def _tiny_spec_meta(sacc_data):
    """A one-TT-spectrum ``spec_meta`` for ``_tiny_mflike_sacc``, in mflike's shape
    (``pol``, ``hasYX_xsp``, ``t1``, ``t2``, ``bpw``, ``leff``, ``ids``)."""
    ell, _, ind = sacc_data.get_ell_cl(
        "cl_00", "LAT_93_s0", "LAT_93_s0", return_ind=True
    )
    bpw = sacc_data.get_bandpower_windows(ind)
    return [
        {
            "pol": "tt",
            "hasYX_xsp": False,
            "t1": "LAT_93",
            "t2": "LAT_93",
            "bpw": bpw,
            "leff": ell,
            "ids": np.arange(len(ell)),
        }
    ]


def _mflike_ids(spec_meta):
    """The per-row bandpower identities for a ``spec_meta``, matching
    ``gaussian.bandpower_ids`` and ``from_cmb_lensing``'s row labels."""
    return [
        (m["pol"], bool(m["hasYX_xsp"]), (m["t1"], m["t2"]), float(leff))
        for m in spec_meta
        for leff in np.asarray(m["leff"])
    ]


def test_from_cmb_lensing_roundtrips_into_multigaussian(tmp_path):
    """``CrossCov.from_cmb_lensing`` must produce a file that loads back usably.

    The cross block is keyed by the components' real ``GaussianData`` names (else
    ``MultiGaussianData`` silently drops it), and the component auto-covariances are
    carried alongside it so the saved file is a self-contained, non-singular joint
    covariance. Every block carries bandpower ids, so it realigns to each component's
    data by identity on load.
    """
    from soliket.gaussian.gaussian_data import (
        CrossCov,
        GaussianData,
        MultiGaussianData,
    )

    sacc_data = _tiny_mflike_sacc()
    sacc_data.save_fits(str(tmp_path / "mflike_cov.fits"), overwrite=True)
    spec_meta = _tiny_spec_meta(sacc_data)

    n_cmb, n_kk = 3, 2
    le_ids = [("pp", ("kappa", "kappa"), float(i)) for i in range(n_kk)]
    # the assembled components carry the same ids the saved block is labelled with
    mf_data = GaussianData(
        "mflike", np.arange(n_cmb), np.zeros(n_cmb), np.eye(n_cmb),
        ids=_mflike_ids(spec_meta),
    )
    le_data = GaussianData(
        "CMB Lensing", np.arange(n_kk), np.zeros(n_kk), np.eye(n_kk) * 1e-15,
        ids=le_ids,
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

    # the off-diagonal block plus both auto-covariances, keyed by the real names
    assert ("mflike", "CMB Lensing") in xcov
    assert xcov[("mflike", "CMB Lensing")].shape == (n_cmb, n_kk)
    np.testing.assert_array_equal(xcov[("mflike", "mflike")], mf_data.cov)
    np.testing.assert_array_equal(xcov[("CMB Lensing", "CMB Lensing")], le_data.cov)

    # round-trips through SACC and into a non-singular joint covariance
    out = tmp_path / "xcov.fits"
    xcov.save(str(out))
    multi = MultiGaussianData([mf_data, le_data], CrossCov.load(str(out)))
    assert np.all(np.isfinite(multi.inv_cov))


def test_from_cmb_lensing_trims_lensing_axis_to_used_bandpowers(tmp_path):
    """The cross block is computed on the full kappa range, but the lensing
    likelihood may keep only the bins surviving its scale cuts. ``from_cmb_lensing``
    trims the columns to those kept bins (via the lensing data's ``indices``) and
    labels them with the kept ids, so the cross block and the lensing auto share one
    column order and the joint covariance assembles cleanly.
    """
    from soliket.gaussian.gaussian_data import (
        CrossCov,
        GaussianData,
        MultiGaussianData,
    )

    sacc_data = _tiny_mflike_sacc()  # 3 CMB bandpowers
    sacc_data.save_fits(str(tmp_path / "mflike_cov.fits"), overwrite=True)
    spec_meta = _tiny_spec_meta(sacc_data)
    n_cmb = 3

    # Lensing has 2 full kappa bins but keeps only 1 (a scale cut): ids span the
    # full range, indices is the kept mask, and the data vector is the kept bin.
    n_kk_full, n_kept = 2, 1
    kk_ids = [("pp", ("kappa", "kappa"), float(i)) for i in range(n_kk_full)]
    kept_mask = np.array([True, False])
    mf_data = GaussianData(
        "mflike", np.arange(n_cmb), np.zeros(n_cmb), np.eye(n_cmb),
        ids=_mflike_ids(spec_meta),
    )
    le_data = GaussianData(
        "CMB Lensing", np.arange(n_kept), np.zeros(n_kept), np.eye(n_kept) * 1e-15,
        indices=kept_mask, ids=kk_ids,
    )

    lmax_kk = 25
    binning = np.zeros((n_kk_full, lmax_kk))
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

    # the stored block is trimmed to the kept kappa bin, matching the lensing auto
    assert xcov[("mflike", "CMB Lensing")].shape == (n_cmb, n_kept)

    out = tmp_path / "xcov.fits"
    xcov.save(str(out))  # must not raise a broadcast error
    multi = MultiGaussianData([mf_data, le_data], CrossCov.load(str(out)))
    assert np.all(np.isfinite(multi.inv_cov))
    # the assembled joint covariance keeps the cross block at the kept granularity
    assert multi.cov[:n_cmb, n_cmb:].shape == (n_cmb, n_kept)


def test_from_cmb_lensing_aligns_reordered_block_in_joint_cov(tmp_path):
    # The whole point of labelling: when the data is in a DIFFERENT order than the
    # block was built in, assembly must realign the cross block by identity (not
    # position). Here the mflike data carries the bandpower ids in reversed order,
    # so the assembled cross block rows must come out reversed relative to the block
    # as built. With the old positional cross-cov this either mis-orders silently or
    # raises (data has ids, block does not).
    from soliket.gaussian.gaussian_data import CrossCov, GaussianData, MultiGaussianData

    sacc_data = _tiny_mflike_sacc()
    sacc_data.save_fits(str(tmp_path / "mflike_cov.fits"), overwrite=True)
    spec_meta = _tiny_spec_meta(sacc_data)
    n_cmb, n_kk = 3, 2

    le_ids = [("pp", ("kappa", "kappa"), float(i)) for i in range(n_kk)]
    # mflike data ids REVERSED relative to the block's spec_meta build order
    rev_ids = _mflike_ids(spec_meta)[::-1]
    mf_data = GaussianData(
        "mflike", np.arange(n_cmb), np.zeros(n_cmb), np.eye(n_cmb), ids=rev_ids
    )
    le_data = GaussianData(
        "CMB Lensing", np.arange(n_kk), np.zeros(n_kk), np.eye(n_kk) * 1e-15,
        ids=le_ids,
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
    block_as_built = np.array(xcov[("mflike", "CMB Lensing")])

    multi = MultiGaussianData([mf_data, le_data], xcov)
    cross = multi.cov[:n_cmb, n_cmb:]
    # data is reversed vs the build order, so the assembled rows are reversed
    np.testing.assert_allclose(cross, block_as_built[::-1])


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


def test_from_cmb_lensing_labels_block_by_bandpower_identity(tmp_path):
    # The cross block must carry per-row/col bandpower identities so CrossCov can
    # realign it to the data by identity (not by build order). Rows use mflike's
    # spec_meta vocabulary (pol, hasYX_xsp, (t1, t2), leff) -- exactly what
    # gaussian.bandpower_ids reconstructs -- and columns reuse the lensing data's
    # own ids. Without these labels the block is silently positional.
    from soliket.gaussian.gaussian_data import CrossCov, GaussianData

    sacc_data = _tiny_mflike_sacc()
    sacc_data.save_fits(str(tmp_path / "mflike_cov.fits"), overwrite=True)
    ell, _, ind = sacc_data.get_ell_cl(
        "cl_00", "LAT_93_s0", "LAT_93_s0", return_ind=True
    )
    bpw = sacc_data.get_bandpower_windows(ind)
    spec_meta = [
        {
            "pol": "tt",
            "hasYX_xsp": False,
            "t1": "LAT_93",
            "t2": "LAT_93",
            "bpw": bpw,
            "leff": ell,
            "ids": np.arange(len(ell)),
        }
    ]

    n_cmb, n_kk = len(ell), 2
    le_ids = [("pp", ("kappa", "kappa"), float(i)) for i in range(n_kk)]
    mf_data = GaussianData("mflike", ell, np.zeros(n_cmb), np.eye(n_cmb))
    le_data = GaussianData(
        "CMB Lensing", np.arange(n_kk), np.zeros(n_kk), np.eye(n_kk) * 1e-15,
        ids=le_ids,
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
    assert block.shape == (n_cmb, n_kk)  # rows from spec_meta, cols full kappa range
    assert np.all(np.isfinite(block))

    row_ids, col_ids = xcov._block_ids_map[("mflike", "CMB Lensing")]
    expected_rows = [("tt", False, ("LAT_93", "LAT_93"), float(leff)) for leff in ell]
    assert row_ids == expected_rows
    assert col_ids == le_ids


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


def test_cmb_combs_from_sacc_accepts_non_contiguous_spectra():
    # Storage order no longer matters: blocks are aligned to the data by bandpower
    # identity (CrossCov.to_canonical), not by position, so the builder accepts a
    # non-contiguously-stored SACC and just yields one triple per tracer combination.
    combs = cmb_combs_from_sacc(_tt_sacc(interleave=True))

    assert len(combs) == 2
    assert [ind_camb for ind_camb, _, _ in combs] == [0, 0]  # both TT -> CAMB row 0




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

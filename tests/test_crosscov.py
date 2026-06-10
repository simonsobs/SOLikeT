import os
from tempfile import gettempdir

import numpy as np
import pytest
import sacc
from sklearn.datasets import make_spd_matrix

from soliket import MultiGaussianLikelihood
from soliket.gaussian import CrossCov
from soliket.gaussian.gaussian import GaussianLikelihood
from soliket.utils import get_likelihood


def create_toy_sacc_file(name: str, n: int, cov: np.ndarray, seed: int, file_path: str):
    """Create a SACC file for ToyLikelihood with dummy data."""
    np.random.seed(seed)

    # Create a simple SACC object
    s = sacc.Sacc()

    # Add a single tracer
    s.add_tracer("Misc", name, quantity="cmb_temperature", spin=0)

    # Create dummy ell values and data
    ells = np.arange(n, dtype=float)
    data = np.random.multivariate_normal(np.zeros(n), cov)

    # Add data points
    for i, (ell, datum) in enumerate(zip(ells, data)):
        s.add_data_point("cl_00", (name, name), datum, ell=ell)

    # Add covariance
    s.add_covariance(cov)

    # Save to file
    s.save_fits(file_path, overwrite=True)
    return file_path


class ToyLikelihood(GaussianLikelihood):
    name = "toy"
    _allowable_tracers = ("cmb_temperature",)

    def _get_theory(self):
        # Get the data size from SACC data
        n = len(self.sacc_data.mean)
        return np.zeros(n)


def test_toy():
    n1, n2, n3 = [10, 20, 30]
    full_cov = make_spd_matrix(n1 + n2 + n3, random_state=1234) * 1e-1
    full_cov += np.diag(np.ones(n1 + n2 + n3))

    cov1 = full_cov[:n1, :n1]
    cov2 = full_cov[n1 : n1 + n2, n1 : n1 + n2]
    cov3 = full_cov[n1 + n2 :, n1 + n2 :]

    name1, name2, name3 = ["A", "B", "C"]

    # Create SACC files for each likelihood instance
    tempdir = gettempdir()
    sacc_path1 = os.path.join(tempdir, f"toy_{name1}.sacc.fits")
    sacc_path2 = os.path.join(tempdir, f"toy_{name2}.sacc.fits")
    sacc_path3 = os.path.join(tempdir, f"toy_{name3}.sacc.fits")

    create_toy_sacc_file(name1, n1, cov1, 123, sacc_path1)
    create_toy_sacc_file(name2, n2, cov2, 234, sacc_path2)
    create_toy_sacc_file(name3, n3, cov3, 345, sacc_path3)

    cross_cov = CrossCov()

    ids1 = [("cl_00", (name1, name1), float(i)) for i in range(n1)]
    ids2 = [("cl_00", (name2, name2), float(i)) for i in range(n2)]
    ids3 = [("cl_00", (name3, name3), float(i)) for i in range(n3)]

    cross_cov.add_cross_covariance(
        name1, name2, full_cov[:n1, n1 : n1 + n2], ids1=ids1, ids2=ids2
    )
    cross_cov.add_cross_covariance(
        name1, name3, full_cov[:n1, n1 + n2 :], ids1=ids1, ids2=ids3
    )
    cross_cov.add_cross_covariance(
        name2, name3, full_cov[n1 : n1 + n2, n1 + n2 :], ids1=ids2, ids2=ids3
    )

    cross_cov_path = os.path.join(tempdir, "toy_cross_cov.sacc.fits")
    cross_cov.save(cross_cov_path)

    info1 = {"name": name1, "datapath": sacc_path1, "use_spectra": "all"}
    info2 = {"name": name2, "datapath": sacc_path2, "use_spectra": "all"}
    info3 = {"name": name3, "datapath": sacc_path3, "use_spectra": "all"}

    lhood = "tests.test_crosscov.ToyLikelihood"
    components = [lhood] * 3
    options = [info1, info2, info3]
    multilike1 = MultiGaussianLikelihood({"components": components, "options": options})
    multilike2 = MultiGaussianLikelihood(
        {"components": components, "options": options, "cross_cov_path": cross_cov_path}
    )

    like1 = get_likelihood(lhood, info1)
    like2 = get_likelihood(lhood, info2)
    like3 = get_likelihood(lhood, info3)

    assert np.isclose(
        multilike1.logp(), sum([likex.logp() for likex in [like1, like2, like3]])
    )
    assert not np.isclose(
        multilike2.logp(), sum([likex.logp() for likex in [like1, like2, like3]])
    )

    assert np.allclose(like1.cov, cov1), "Likelihood 1 covariance mismatch"
    assert np.allclose(like2.cov, cov2), "Likelihood 2 covariance mismatch"
    assert np.allclose(like3.cov, cov3), "Likelihood 3 covariance mismatch"

    assert len(like1.y) == n1, f"Likelihood 1 data size {len(like1.y)} != {n1}"
    assert len(like2.y) == n2, f"Likelihood 2 data size {len(like2.y)} != {n2}"
    assert len(like3.y) == n3, f"Likelihood 3 data size {len(like3.y)} != {n3}"

    cross_cov_loaded = multilike2.cross_cov
    assert cross_cov_loaded is not None, "Cross-covariance should be loaded"

    orig_blocks = {
        (name1, name2): full_cov[:n1, n1 : n1 + n2],
        (name1, name3): full_cov[:n1, n1 + n2 :],
        (name2, name3): full_cov[n1 : n1 + n2, n1 + n2 :],
    }

    for key in cross_cov_loaded.keys():
        loaded_block = cross_cov_loaded[key]

        if key in orig_blocks:
            orig_block = orig_blocks[key]
        else:
            key_rev = (key[1], key[0])
            if key_rev in orig_blocks:
                orig_block = orig_blocks[key_rev].T
            else:
                continue

        assert np.allclose(loaded_block, orig_block), f"Cross-cov {key} mismatch"


def test_crosscov_add_component():
    """Test CrossCov with explicit add_component() calls."""
    from soliket.gaussian.gaussian_data import CrossCov

    n1, n2 = 5, 8
    cov1 = make_spd_matrix(n1, random_state=42)
    cov2 = make_spd_matrix(n2, random_state=43)
    cross_12 = np.random.randn(n1, n2) * 0.1

    # Test add_component
    crosscov = CrossCov()
    crosscov.add_component("A", cov1)
    crosscov.add_component("B", cov2)
    crosscov.add_cross_covariance("A", "B", cross_12)

    assert ("A", "A") in crosscov
    assert ("B", "B") in crosscov
    assert ("A", "B") in crosscov
    assert crosscov.component_names == ["A", "B"]

    # Test save/load roundtrip with explicit components
    tempdir = gettempdir()
    path = os.path.join(tempdir, "test_explicit.fits")
    crosscov.save(path)

    loaded = CrossCov.load(path)
    assert np.allclose(crosscov[("A", "A")], loaded[("A", "A")])
    assert np.allclose(crosscov[("B", "B")], loaded[("B", "B")])
    assert np.allclose(crosscov[("A", "B")], loaded[("A", "B")])


def test_crosscov_error_handling():
    """Test CrossCov error handling."""
    import pytest

    from soliket.gaussian.gaussian_data import CrossCov

    # Test invalid file format on save
    crosscov = CrossCov()
    crosscov.add_cross_covariance("A", "B", np.ones((3, 4)))
    with pytest.raises(ValueError, match="Only .fits or .sacc"):
        crosscov.save("invalid.npz")

    # Test invalid file format on load
    with pytest.raises(ValueError, match="Only .fits or .sacc"):
        CrossCov.load("invalid.npz")

    # Test dict passed as cov
    with pytest.raises(TypeError, match="must be a numpy array"):
        crosscov.add_component("C", {"not": "an array"})

    # Test inconsistent sizes
    crosscov2 = CrossCov()
    crosscov2.add_cross_covariance("A", "B", np.ones((3, 4)))
    crosscov2.add_cross_covariance("A", "C", np.ones((5, 6)))  # A size mismatch
    with pytest.raises(ValueError, match="Inconsistent sizes"):
        crosscov2._infer_component_info()


def test_multigaussiandata_properties():
    """Test MultiGaussianData properties."""
    from soliket.gaussian.gaussian_data import GaussianData, MultiGaussianData

    n1, n2 = 5, 8
    cov1 = make_spd_matrix(n1, random_state=42)
    cov2 = make_spd_matrix(n2, random_state=43)

    x1 = np.arange(n1, dtype=float)
    y1 = np.zeros(n1)
    data1 = GaussianData("A", x1, y1, cov1)

    x2 = np.arange(n2, dtype=float)
    y2 = np.zeros(n2)
    data2 = GaussianData("B", x2, y2, cov2)

    multi = MultiGaussianData([data1, data2])

    # Test properties
    assert multi.name == "A + B"
    assert multi.cov.shape == (n1 + n2, n1 + n2)
    assert multi.inv_cov.shape == (n1 + n2, n1 + n2)
    assert isinstance(multi.norm_const, float)
    assert multi.labels == ["A"] * n1 + ["B"] * n2
    assert multi.lengths == [n1, n2]
    assert multi.names == ["A", "B"]


def test_multigaussiandata_with_crosscov_modes():
    """Test MultiGaussianData with different CrossCov modes."""
    from soliket.gaussian.gaussian_data import CrossCov, GaussianData, MultiGaussianData

    n1, n2 = 5, 8
    full_cov = make_spd_matrix(n1 + n2, random_state=44)
    cov1 = full_cov[:n1, :n1]
    cov2 = full_cov[n1:, n1:]
    cross_12 = full_cov[:n1, n1:]

    x1, y1 = np.arange(n1, dtype=float), np.zeros(n1)
    x2, y2 = np.arange(n2, dtype=float), np.zeros(n2)
    data1 = GaussianData("A", x1, y1, cov1)
    data2 = GaussianData("B", x2, y2, cov2)
    data_list = [data1, data2]

    # Mode 1: Full CrossCov with add_component
    crosscov_full = CrossCov()
    crosscov_full.add_component("A", cov1)
    crosscov_full.add_component("B", cov2)
    crosscov_full.add_cross_covariance("A", "B", cross_12)
    multi_full = MultiGaussianData(data_list, crosscov_full)

    # Mode 2: Cross-only CrossCov
    crosscov_cross = CrossCov()
    crosscov_cross.add_cross_covariance("A", "B", cross_12)
    multi_cross = MultiGaussianData(data_list, crosscov_cross)

    # Mode 3: No CrossCov (auto-covs from individual data)
    multi_none = MultiGaussianData(data_list, None)

    # Verify full and cross-only produce same result
    assert np.allclose(multi_full.cov, multi_cross.cov)

    # Verify no cross-cov has zeros in off-diagonal
    assert np.allclose(multi_none.cov[:n1, n1:], 0)
    assert np.allclose(multi_none.cov[n1:, :n1], 0)


def create_multi_combo_sacc_file(
    tracers: list[str], ns: list[int], cov: np.ndarray, file_path: str
):
    """Create a SACC file with several auto-spectra (one per tracer).

    This lets us drop part of a probe's data vector via ``use_spectra`` to mimic
    scale cuts, while the cross-covariance is built over the full (uncut) range.
    """
    s = sacc.Sacc()
    for t in tracers:
        s.add_tracer("Misc", t, quantity="cmb_temperature", spin=0)
    for t, n in zip(tracers, ns):
        for i in range(n):
            s.add_data_point("cl_00", (t, t), 0.0, ell=float(i))
    s.add_covariance(cov)
    s.save_fits(file_path, overwrite=True)
    return file_path


def test_crosscov_different_probe_ranges():
    """Cross-covariance must be cut correctly when probe ranges differ.

    Reproduces the user-reported bug: when the probes entering a
    ``MultiGaussianLikelihood`` do not share the same range (i.e. at least one
    applies a scale cut so its data vector is shorter than the range used to
    build the cross-covariance), the off-diagonal (and auto) covariance blocks
    are not cut correctly.

    Root cause: ``GaussianLikelihood._get_gauss_data`` builds its
    ``GaussianData`` without passing ``indices``, so ``GaussianData.indices``
    defaults to an all-True mask of the *cut* length. ``MultiGaussianData`` then
    trims the full-range cross-covariance block with that mask
    (``cov_block[d1.indices, :][:, d2.indices]``), but the mask length (cut
    size) no longer matches the block dimension (full size), raising::

        IndexError: boolean index did not match indexed array along dimension 0

    The fix (to be implemented separately) must make the likelihood record which
    full-range bandpowers it kept so the blocks can be trimmed correctly.
    """
    tempdir = gettempdir()

    # Probe A has two auto-spectra (A0: 6 points, A1: 4 points) -> full size 10.
    # Probe B has a single auto-spectrum (8 points).
    nA0, nA1, nB = 6, 4, 8
    nA = nA0 + nA1

    full_cov = make_spd_matrix(nA + nB, random_state=11) + np.eye(nA + nB)
    cov_A = full_cov[:nA, :nA]
    cov_B = full_cov[nA:, nA:]
    cross_AB = full_cov[:nA, nA:]  # full off-diagonal block (10 x 8)

    sacc_A = create_multi_combo_sacc_file(
        ["A0", "A1"], [nA0, nA1], cov_A, os.path.join(tempdir, "rangeA.sacc.fits")
    )
    sacc_B = create_multi_combo_sacc_file(
        ["B0"], [nB], cov_B, os.path.join(tempdir, "rangeB.sacc.fits")
    )

    # Cross-covariance built over the FULL range of both probes.
    cross_cov = CrossCov()
    ids_A = [("cl_00", ("A0", "A0"), float(i)) for i in range(nA0)] + [
        ("cl_00", ("A1", "A1"), float(i)) for i in range(nA1)
    ]
    ids_B = [("cl_00", ("B0", "B0"), float(i)) for i in range(nB)]
    cross_cov.add_component("A", cov_A, ids=ids_A)
    cross_cov.add_component("B", cov_B, ids=ids_B)
    cross_cov.add_cross_covariance("A", "B", cross_AB, ids1=ids_A, ids2=ids_B)
    cross_cov_path = os.path.join(tempdir, "range_cross_cov.sacc.fits")
    cross_cov.save(cross_cov_path)

    lhood = "tests.test_crosscov.ToyLikelihood"

    # At runtime probe A keeps ONLY (A0, A0): its range now differs from the
    # full range used to build the cross-covariance (kept indices = [0..5]).
    info_A = {"name": "A", "datapath": sacc_A, "use_spectra": [("A0", "A0")]}
    info_B = {"name": "B", "datapath": sacc_B, "use_spectra": "all"}

    multilike = MultiGaussianLikelihood(
        {
            "components": [lhood, lhood],
            "options": [info_A, info_B],
            "cross_cov_path": cross_cov_path,
        }
    )

    cov = multilike.data.cov
    nA_cut = nA0  # only A0 survives the cut

    assert cov.shape == (nA_cut + nB, nA_cut + nB)

    # Auto-covariance of A must be the full block restricted to the kept points.
    assert np.allclose(cov[:nA_cut, :nA_cut], cov_A[:nA_cut, :nA_cut])
    # Auto-covariance of B is untouched.
    assert np.allclose(cov[nA_cut:, nA_cut:], cov_B)
    # Off-diagonal must be the full cross block restricted to A's kept rows.
    assert np.allclose(cov[:nA_cut, nA_cut:], cross_AB[:nA_cut, :])
    assert np.allclose(cov[nA_cut:, :nA_cut], cross_AB[:nA_cut, :].T)


def test_crosscov_realigned_by_identity():
    """Cross-covariance blocks must be aligned to the data by *identity*.

    ``CrossCov`` stores raw matrices; ``MultiGaussianData`` looks blocks up by
    component name and places them in the right block position, but it must not
    assume the block's internal row/col order matches each probe's data vector.

    Here the cross-covariance is built with component A in a *shuffled* order
    relative to the ``GaussianData`` it will be combined with. When the components
    carry per-bandpower identity keys, the blocks must be realigned by matching
    those keys to the data's keys, so the assembled covariance is correct
    regardless of the order the cross-covariance was built in.

    Without identity realignment the shuffled block is used positionally and the
    assembled auto/off-diagonal blocks are silently wrong (yet still symmetric).
    """
    from soliket.gaussian.gaussian_data import (
        CrossCov,
        GaussianData,
        MultiGaussianData,
    )

    nA, nB = 5, 4
    full = make_spd_matrix(nA + nB, random_state=5) + np.eye(nA + nB)
    cov_A = full[:nA, :nA]
    cov_B = full[nA:, nA:]
    cross_AB = full[:nA, nA:]

    # Per-bandpower identities, in the order the likelihood data uses.
    ids_A = [("cl_00", ("A", "A"), float(i)) for i in range(nA)]
    ids_B = [("cl_00", ("B", "B"), float(i)) for i in range(nB)]

    # GaussianData in the canonical (ids_A / ids_B) order.
    d_A = GaussianData("A", np.arange(nA, dtype=float), np.zeros(nA), cov_A, ids=ids_A)
    d_B = GaussianData("B", np.arange(nB, dtype=float), np.zeros(nB), cov_B, ids=ids_B)

    # Build the CrossCov with component A in a DIFFERENT (shuffled) order.
    perm = np.array([3, 4, 0, 1, 2])
    cross_cov = CrossCov()
    cross_cov.add_component("A", cov_A[np.ix_(perm, perm)], ids=[ids_A[i] for i in perm])
    cross_cov.add_component("B", cov_B, ids=ids_B)
    cross_cov.add_cross_covariance(
        "A",
        "B",
        cross_AB[perm, :],
        ids1=[ids_A[i] for i in perm],
        ids2=ids_B,
    )

    multi = MultiGaussianData([d_A, d_B], cross_cov)
    cov = multi.cov

    # Everything must be realigned back to the data (ids_A) order.
    assert np.allclose(cov[:nA, :nA], cov_A), "auto-A not realigned to data order"
    assert np.allclose(cov[:nA, nA:], cross_AB), "off-diagonal not realigned by identity"
    assert np.allclose(cov[nA:, :nA], cross_AB.T)
    assert np.allclose(cov[nA:, nA:], cov_B)


def test_crosscov_realign_then_trim():
    """Realignment by identity and trimming by scale cut must compose.

    The CrossCov is built on the FULL range (shuffled), while probe A applies a
    scale cut. ``CrossCov.to_canonical`` realigns each block to the data's order
    and trims it to the kept points in a single identity gather per axis; the
    result must reproduce the cross block restricted to A's kept points.
    """
    from soliket.gaussian.gaussian_data import (
        CrossCov,
        GaussianData,
        MultiGaussianData,
    )

    nA, nB, keep = 5, 4, 3
    full = make_spd_matrix(nA + nB, random_state=3) + np.eye(nA + nB)
    cov_A, cov_B, cross_AB = full[:nA, :nA], full[nA:, nA:], full[:nA, nA:]
    ids_A = [("cl_00", ("A", "A"), float(i)) for i in range(nA)]
    ids_B = [("cl_00", ("B", "B"), float(i)) for i in range(nB)]

    # Probe A keeps only its first `keep` points; ids stay on the full range.
    kept_mask = np.array([i < keep for i in range(nA)], dtype=bool)
    d_A = GaussianData(
        "A",
        np.arange(keep, dtype=float),
        np.zeros(keep),
        cov_A[:keep, :keep],
        indices=kept_mask,
        ids=ids_A,
    )
    d_B = GaussianData("B", np.arange(nB, dtype=float), np.zeros(nB), cov_B, ids=ids_B)

    # CrossCov built on the FULL range, with component A in a shuffled order.
    perm = np.array([3, 4, 0, 1, 2])
    cc = CrossCov()
    cc.add_component("A", cov_A[np.ix_(perm, perm)], ids=[ids_A[i] for i in perm])
    cc.add_component("B", cov_B, ids=ids_B)
    cc.add_cross_covariance(
        "A",
        "B",
        cross_AB[perm, :],
        ids1=[ids_A[i] for i in perm],
        ids2=ids_B,
    )

    cov = MultiGaussianData([d_A, d_B], cc).cov

    assert cov.shape == (keep + nB, keep + nB)
    # Realigned back to data order, then trimmed to the kept points.
    assert np.allclose(cov[:keep, :keep], cov_A[:keep, :keep])
    assert np.allclose(cov[:keep, keep:], cross_AB[:keep, :])
    assert np.allclose(cov[keep:, :keep], cross_AB[:keep, :].T)
    assert np.allclose(cov[keep:, keep:], cov_B)


def test_crosscov_ids_survive_save_load():
    """Per-bandpower identity keys must survive save/load and still realign."""
    from soliket.gaussian.gaussian_data import (
        CrossCov,
        GaussianData,
        MultiGaussianData,
    )

    nA, nB = 5, 4
    full = make_spd_matrix(nA + nB, random_state=8) + np.eye(nA + nB)
    cov_A, cov_B, cross_AB = full[:nA, :nA], full[nA:, nA:], full[:nA, nA:]
    ids_A = [("cl_00", ("A", "A"), float(i)) for i in range(nA)]
    ids_B = [("cl_00", ("B", "B"), float(i)) for i in range(nB)]

    perm = np.array([3, 4, 0, 1, 2])
    cc = CrossCov()
    cc.add_component("A", cov_A[np.ix_(perm, perm)], ids=[ids_A[i] for i in perm])
    cc.add_component("B", cov_B, ids=ids_B)
    cc.add_cross_covariance(
        "A",
        "B",
        cross_AB[perm, :],
        ids1=[ids_A[i] for i in perm],
        ids2=ids_B,
    )

    path = os.path.join(gettempdir(), "ids_roundtrip.fits")
    cc.save(path)
    loaded = CrossCov.load(path)

    # Keys round-trip as (hashable) tuples in the same order.
    assert loaded.component_ids("A") == [ids_A[i] for i in perm]
    assert loaded.component_ids("B") == ids_B

    # And realignment through the loaded object is correct.
    d_A = GaussianData("A", np.arange(nA, dtype=float), np.zeros(nA), cov_A, ids=ids_A)
    d_B = GaussianData("B", np.arange(nB, dtype=float), np.zeros(nB), cov_B, ids=ids_B)
    cov = MultiGaussianData([d_A, d_B], loaded).cov
    assert np.allclose(cov[:nA, :nA], cov_A)
    assert np.allclose(cov[:nA, nA:], cross_AB)


def test_bandpower_ids_adapter():
    """The ids adapter reconstructs keys from mflike spec_meta and from sacc."""
    from soliket.gaussian.gaussian import bandpower_ids

    # mflike-style component: spec_meta over a (compressed) data vector.
    # The two "te" entries (TE and ET) of the cross-pair share (pol, t1, t2, ell)
    # and are disambiguated only by hasYX_xsp.
    class FakeMflike:
        data_vec = np.zeros(4)
        spec_meta = [
            {
                "ids": np.array([0, 1]),
                "pol": "te",
                "hasYX_xsp": False,
                "t1": "LAT_93",
                "t2": "LAT_145",
                "leff": np.array([100.0, 200.0]),
            },
            {
                "ids": np.array([2, 3]),
                "pol": "te",
                "hasYX_xsp": True,
                "t1": "LAT_93",
                "t2": "LAT_145",
                "leff": np.array([100.0, 200.0]),
            },
        ]

    keys = bandpower_ids(FakeMflike())
    assert keys == [
        ("te", False, ("LAT_93", "LAT_145"), 100.0),
        ("te", False, ("LAT_93", "LAT_145"), 200.0),
        ("te", True, ("LAT_93", "LAT_145"), 100.0),
        ("te", True, ("LAT_93", "LAT_145"), 200.0),
    ]
    assert len(set(keys)) == len(keys), "mflike keys must be unique"

    # sacc-style component.
    s = sacc.Sacc()
    s.add_tracer("Misc", "X", quantity="cmb_temperature", spin=0)
    for ell in (30.0, 60.0, 90.0):
        s.add_data_point("cl_00", ("X", "X"), 0.0, ell=ell)

    class FakeSoliket:
        sacc_data = s

    assert bandpower_ids(FakeSoliket()) == [
        ("cl_00", ("X", "X"), 30.0),
        ("cl_00", ("X", "X"), 60.0),
        ("cl_00", ("X", "X"), 90.0),
    ]


def test_to_canonical_identity_gather_and_trim():
    """to_canonical reshuffles a shuffled block into the target order and trims."""
    from soliket.gaussian.gaussian_data import CrossCov

    nA, nB = 5, 4
    full = make_spd_matrix(nA + nB, random_state=5) + np.eye(nA + nB)
    cov_A, cov_B, cross_AB = full[:nA, :nA], full[nA:, nA:], full[:nA, nA:]
    ids_A = [("cl_00", ("A", "A"), float(i)) for i in range(nA)]
    ids_B = [("cl_00", ("B", "B"), float(i)) for i in range(nB)]

    perm = [3, 4, 0, 1, 2]
    cc = CrossCov()
    cc.add_component("A", cov_A[np.ix_(perm, perm)], ids=[ids_A[i] for i in perm])
    cc.add_component("B", cov_B, ids=ids_B)
    cc.add_cross_covariance(
        "A",
        "B",
        cross_AB[perm, :],
        ids1=[ids_A[i] for i in perm],
        ids2=ids_B,
    )

    # Full target order -> realign only.
    full_out = cc.to_canonical({"A": ids_A, "B": ids_B})
    assert np.allclose(full_out[:nA, :nA], cov_A)
    assert np.allclose(full_out[:nA, nA:], cross_AB)
    assert np.allclose(full_out[nA:, nA:], cov_B)

    # Trimmed target (keep A's first 3) -> realign + trim fused.
    keep = ids_A[:3]
    out = cc.to_canonical({"A": keep, "B": ids_B})
    assert out.shape == (3 + nB, 3 + nB)
    assert np.allclose(out[:3, :3], cov_A[:3, :3])
    assert np.allclose(out[:3, 3:], cross_AB[:3, :])


def test_to_canonical_missing_block_is_zeros():
    from soliket.gaussian.gaussian_data import CrossCov

    ids_A = [("cl_00", ("A", "A"), float(i)) for i in range(2)]
    ids_B = [("cl_00", ("B", "B"), float(i)) for i in range(3)]
    cc = CrossCov()
    cc.add_component("A", np.eye(2), ids=ids_A)
    cc.add_component("B", np.eye(3), ids=ids_B)
    # no A-B cross block
    out = cc.to_canonical({"A": ids_A, "B": ids_B})
    assert np.allclose(out[:2, 2:], 0.0)
    assert np.allclose(out[2:, :2], 0.0)


def test_to_canonical_superset_block_sliced_by_identity():
    """A block built on a WIDER range is sliced down by identity."""
    from soliket.gaussian.gaussian_data import CrossCov

    ids_full = [("cl_00", ("A", "A"), float(i)) for i in range(4)]
    cc = CrossCov()
    cc.add_component("A", np.diag([1.0, 2.0, 3.0, 4.0]), ids=ids_full)
    # Ask only for a subset, in a different order.
    target = [ids_full[2], ids_full[0]]
    out = cc.to_canonical({"A": target})
    assert np.allclose(out, np.diag([3.0, 1.0]))


def test_to_canonical_missing_bandpower_raises():
    import pytest

    from soliket.gaussian.gaussian_data import CrossCov

    ids = [("cl_00", ("A", "A"), float(i)) for i in range(3)]
    cc = CrossCov()
    cc.add_component("A", np.eye(3), ids=ids)
    bogus = ("cl_00", ("A", "A"), 99.0)
    with pytest.raises(ValueError, match="missing bandpower"):
        cc.to_canonical({"A": ids[:2] + [bogus]})


def test_to_canonical_target_ids_block_unlabelled_raises():
    """Target carries ids but the block does not -> refuse to guess."""
    import pytest

    from soliket.gaussian.gaussian_data import CrossCov

    ids = [("cl_00", ("A", "A"), float(i)) for i in range(3)]
    cc = CrossCov()
    cc.add_component("A", np.eye(3))  # NO ids on the block
    with pytest.raises(ValueError, match="does not carry bandpower identities"):
        cc.to_canonical({"A": ids})


def test_to_canonical_positional_when_no_ids_anywhere():
    """Neither target nor block has ids -> positional (size-based)."""
    from soliket.gaussian.gaussian_data import CrossCov

    cc = CrossCov()
    cc.add_component("A", np.diag([1.0, 2.0]))
    cc.add_component("B", np.diag([3.0]))
    cc.add_cross_covariance("A", "B", np.array([[0.1], [0.2]]))
    out = cc.to_canonical({"A": 2, "B": 1})  # int sizes = positional
    assert out.shape == (3, 3)
    assert np.allclose(out[:2, :2], np.diag([1.0, 2.0]))
    assert np.allclose(out[:2, 2:], np.array([[0.1], [0.2]]))


def test_to_canonical_positional_size_mismatch_raises():
    import pytest

    from soliket.gaussian.gaussian_data import CrossCov

    cc = CrossCov()
    cc.add_component("A", np.eye(3))  # block axis length 3, no ids
    with pytest.raises(ValueError, match="must already match the target size"):
        cc.to_canonical({"A": 2})  # positional target of wrong size


def test_to_canonical_fills_transpose_from_one_direction():
    """An upper-triangle-only CrossCov still assembles a symmetric matrix."""
    from soliket.gaussian.gaussian_data import CrossCov

    cc = CrossCov()
    cc.add_component("A", np.diag([1.0, 2.0]))
    cc.add_component("B", np.diag([3.0]))
    cc[("A", "B")] = np.array([[0.5], [0.7]])  # only one direction, no ("B","A")
    out = cc.to_canonical({"A": 2, "B": 1})
    assert np.allclose(out, out.T)
    assert np.allclose(out[:2, 2:], np.array([[0.5], [0.7]]))
    assert np.allclose(out[2:, :2], np.array([[0.5, 0.7]]))


def test_kept_order_uses_ids_at_data_vector_granularity():
    """mflike-like case: ids describe the data vector directly while indices is
    a longer, mismatched mask. _kept_order must use the ids, not zip-truncate."""
    from soliket.gaussian.gaussian_data import GaussianData, MultiGaussianData

    ids = [("te", float(i)) for i in range(3)]
    d = GaussianData("m", np.arange(3.0), np.zeros(3), np.eye(3), ids=ids)
    # Simulate mflike's mismatched, longer indices (len 4, not 3).
    d.indices = np.array([True, True, False, True])
    assert MultiGaussianData._kept_order(d) == ids


def test_kept_order_trims_on_full_range_mask():
    """SOLikeT scale-cut case: ids span the full range with a same-length kept
    mask; _kept_order returns the kept subset matching the data length."""
    from soliket.gaussian.gaussian_data import GaussianData, MultiGaussianData

    full_ids = [("cl_00", float(i)) for i in range(5)]
    mask = np.array([True, True, False, True, False])  # keep 3
    d = GaussianData(
        "s", np.arange(3.0), np.zeros(3), np.eye(3), indices=mask, ids=full_ids
    )
    assert MultiGaussianData._kept_order(d) == [full_ids[0], full_ids[1], full_ids[3]]


def test_arbitrary_order_blocks_not_rejected():
    """Two blocks labelling the same component in different orders are accepted
    and each canonicalises independently to the data order."""
    from soliket.gaussian.gaussian_data import CrossCov

    ids_A = [("cl_00", ("A", "A"), float(i)) for i in range(3)]
    ids_B = [("cl_00", ("B", "B"), float(i)) for i in range(2)]
    ids_C = [("cl_00", ("C", "C"), float(i)) for i in range(2)]

    cc = CrossCov()
    cc.add_component("A", np.diag([1.0, 2.0, 3.0]), ids=ids_A)
    cc.add_cross_covariance(
        "A", "B", np.arange(6.0).reshape(3, 2), ids1=ids_A, ids2=ids_B
    )
    perm = [2, 0, 1]
    cc.add_cross_covariance(
        "A",
        "C",
        np.arange(6.0).reshape(3, 2)[perm, :],
        ids1=[ids_A[i] for i in perm],
        ids2=ids_C,
    )

    out = cc.to_canonical({"A": ids_A, "B": ids_B, "C": ids_C})
    assert np.allclose(out[:3, 3:5], np.arange(6.0).reshape(3, 2))
    assert np.allclose(out[:3, 5:7], np.arange(6.0).reshape(3, 2))


def test_input_validation_ids_length():
    import pytest

    from soliket.gaussian.gaussian_data import CrossCov

    cc = CrossCov()
    with pytest.raises(ValueError, match="length"):
        cc.add_component("A", np.eye(3), ids=[("x",), ("y",)])  # 2 ids, dim 3
    with pytest.raises(ValueError, match="length"):
        cc.add_cross_covariance("A", "B", np.ones((3, 2)), ids1=[("a",)], ids2=None)


def test_save_reconciles_cross_only_component_orders():
    """A component shared by several blocks in same-set but different orders is
    reconciled on save (here the A-axis of the A-C block is permuted relative to
    A's auto block). The round-trip must place each block in the canonical order.
    """
    ids_A = [("cl_00", ("A", "A"), float(i)) for i in range(3)]
    ids_B = [("cl_00", ("B", "B"), float(i)) for i in range(2)]
    ids_C = [("cl_00", ("C", "C"), float(i)) for i in range(2)]

    cross_AB = np.array([[10.0, 11.0], [12.0, 13.0], [14.0, 15.0]])  # rows a0,a1,a2
    cross_AC = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])  # rows a0,a1,a2
    perm = [2, 0, 1]  # store A-C with A-axis as [a2, a0, a1]

    cc = CrossCov()
    cc.add_component("A", np.eye(3), ids=ids_A)
    cc.add_component("B", np.eye(2), ids=ids_B)
    cc.add_component("C", np.eye(2), ids=ids_C)
    cc.add_cross_covariance("A", "B", cross_AB, ids1=ids_A, ids2=ids_B)
    cc.add_cross_covariance(
        "A", "C", cross_AC[perm, :], ids1=[ids_A[i] for i in perm], ids2=ids_C
    )

    path = os.path.join(gettempdir(), "reconcile_multi.sacc.fits")
    cc.save(path)  # must NOT raise

    loaded = CrossCov.load(path)
    full = loaded.to_canonical({"A": ids_A, "B": ids_B, "C": ids_C})
    # A rows 0:3; B cols 3:5; C cols 5:7. Both cross blocks in canonical A order.
    assert np.allclose(full[0:3, 3:5], cross_AB)
    assert np.allclose(full[0:3, 5:7], cross_AC)


def test_load_rejects_unlabelled_file(tmp_path):
    """Clean break: a cross-cov saved without ids metadata cannot be loaded."""
    s = sacc.Sacc()
    s.add_tracer("misc", "A", quantity="generic", spin=0)
    for i in range(2):
        s.add_data_point("generic", ("A", "A"), 0.0, ell=float(i))
    s.add_covariance(np.eye(2))
    path = str(tmp_path / "old.fits")
    s.save_fits(path, overwrite=True)
    with pytest.raises(ValueError, match="regenerate"):
        CrossCov.load(path)


def test_mflike_style_component_assembles_to_dcov_untrimmed():
    """Regression for the mflike granularity case.

    mflike's GaussianData carries ``ids`` at the data-vector granularity
    (``len(ids) == len(d)``) alongside a longer, mismatched ``indices`` mask (in
    production: 3087 ids vs a 3108-long mask). Such a component must NOT be
    trimmed or reordered at assembly: its block in the joint covariance must be
    its own ``d.cov`` byte-for-byte. This pins both ``_kept_order`` (returns the
    data-vector ids, never zip-truncating against the mismatched mask) and the
    end-to-end assembly, standalone and with a cross-covariance present.
    """
    from soliket.gaussian.gaussian_data import (
        CrossCov,
        GaussianData,
        MultiGaussianData,
    )

    nM, nL = 4, 3
    full = make_spd_matrix(nM + nL, random_state=17) + np.eye(nM + nL)
    cov_M = full[:nM, :nM]
    cov_L = full[nM:, nM:]
    cross_ML = full[:nM, nM:]

    # mflike-style: ids label the data vector (length nM == len(d)).
    ids_M = [("te", False, ("LAT_93", "LAT_145"), float(i)) for i in range(nM)]
    ids_L = [("cl_00", ("CMBk", "CMBk"), float(i)) for i in range(nL)]

    d_M = GaussianData(
        "mflike", np.arange(nM, dtype=float), np.zeros(nM), cov_M, ids=ids_M
    )
    # Simulate mflike's mismatched, LONGER kept-mask (len nM+2 != len(ids_M)).
    d_M.indices = np.array([True] * nM + [False, True])
    d_L = GaussianData(
        "CMBk", np.arange(nL, dtype=float), np.zeros(nL), cov_L, ids=ids_L
    )

    # _kept_order must use the data-vector ids, not zip-truncate the mask.
    assert MultiGaussianData._kept_order(d_M) == ids_M

    # (a) Standalone (no cross-cov): the mflike block is d.cov, byte-for-byte.
    cov_a = MultiGaussianData([d_M, d_L]).cov
    assert np.array_equal(cov_a[:nM, :nM], cov_M)
    assert np.array_equal(cov_a[nM:, nM:], cov_L)
    assert np.allclose(cov_a[:nM, nM:], 0.0)

    # (b) With a labelled cross-cov: diagonals stay exactly d.cov; the off-
    # diagonal is placed by identity. No trim/reorder despite the bad mask.
    cc = CrossCov()
    cc.add_cross_covariance("mflike", "CMBk", cross_ML, ids1=ids_M, ids2=ids_L)
    cov_b = MultiGaussianData([d_M, d_L], cc).cov
    assert np.array_equal(cov_b[:nM, :nM], cov_M)
    assert np.array_equal(cov_b[nM:, nM:], cov_L)
    assert np.allclose(cov_b[:nM, nM:], cross_ML)
    assert np.allclose(cov_b[nM:, :nM], cross_ML.T)


def test_save_reconciles_conflicting_same_set_orders():
    """save() must reconcile blocks that label a component in conflicting but
    same-set orders, instead of refusing.

    A user builds a cross-covariance in their own (arbitrary) bandpower order
    and an auto-covariance in the data order. Both carry genuine identity keys
    that are permutations of the *same* set. Because every block is labelled,
    the store has all it needs to realign by identity, so save() should pick a
    canonical order, write a self-consistent file, and round-trip exactly.
    """
    from soliket.gaussian.gaussian_data import CrossCov

    a0 = ("cl_00", ("A", "A"), 0.0)
    a1 = ("cl_00", ("A", "A"), 1.0)
    a2 = ("cl_00", ("A", "A"), 2.0)
    b0 = ("cl_00", ("B", "B"), 0.0)
    b1 = ("cl_00", ("B", "B"), 1.0)

    full = make_spd_matrix(5, random_state=7) + np.eye(5)
    cov_AA = full[:3, :3]
    cov_BB = full[3:, 3:]
    cross_AB = full[:3, 3:]  # rows in [a0, a1, a2] order

    cc = CrossCov()
    cc.add_component("A", cov_AA, ids=[a0, a1, a2])  # auto in data order
    cc.add_component("B", cov_BB, ids=[b0, b1])
    # Cross block stored with A's axis PERMUTED -> [a2, a0, a1] (same set).
    perm = [2, 0, 1]
    cc.add_cross_covariance(
        "A", "B", cross_AB[perm, :], ids1=[a2, a0, a1], ids2=[b0, b1]
    )

    path = os.path.join(gettempdir(), "reconcile_cross_cov.sacc.fits")
    cc.save(path)  # must NOT raise

    loaded = CrossCov.load(path)

    # Realigning to the data order must reproduce the original full covariance,
    # i.e. the permuted cross block was correctly reconciled on save/load.
    target = {"A": [a0, a1, a2], "B": [b0, b1]}
    assert np.allclose(loaded.to_canonical(target), full)


def test_save_raises_on_genuine_set_mismatch():
    """save() must still raise when a component's blocks carry *different sets*
    of identity keys (not a mere reordering) -- that is genuinely unreconcilable.
    """
    from soliket.gaussian.gaussian_data import CrossCov

    a0 = ("cl_00", ("A", "A"), 0.0)
    a1 = ("cl_00", ("A", "A"), 1.0)
    a2 = ("cl_00", ("A", "A"), 2.0)
    aX = ("cl_00", ("A", "A"), 99.0)  # not in the auto's set
    b0 = ("cl_00", ("B", "B"), 0.0)
    b1 = ("cl_00", ("B", "B"), 1.0)

    full = make_spd_matrix(5, random_state=8) + np.eye(5)
    cc = CrossCov()
    cc.add_component("A", full[:3, :3], ids=[a0, a1, a2])
    cc.add_component("B", full[3:, 3:], ids=[b0, b1])
    cc.add_cross_covariance(
        "A", "B", full[:3, 3:], ids1=[a0, a1, aX], ids2=[b0, b1]
    )

    path = os.path.join(gettempdir(), "mismatch_cross_cov.sacc.fits")
    with pytest.raises(ValueError, match="different|bandpower|set"):
        cc.save(path)


def test_save_warns_naming_reordered_block():
    """When save() realigns a block to the canonical order, it must warn the
    user, naming the block and the component axis that was reordered.
    """
    a0 = ("cl_00", ("A", "A"), 0.0)
    a1 = ("cl_00", ("A", "A"), 1.0)
    a2 = ("cl_00", ("A", "A"), 2.0)
    b0 = ("cl_00", ("B", "B"), 0.0)
    b1 = ("cl_00", ("B", "B"), 1.0)

    full = make_spd_matrix(5, random_state=3) + np.eye(5)
    cc = CrossCov()
    cc.add_component("A", full[:3, :3], ids=[a0, a1, a2])  # canonical A order
    cc.add_component("B", full[3:, 3:], ids=[b0, b1])
    cc.add_cross_covariance(
        "A", "B", full[:3, 3:][[2, 0, 1], :], ids1=[a2, a0, a1], ids2=[b0, b1]
    )

    with pytest.warns(UserWarning) as record:
        cc.save(os.path.join(gettempdir(), "warn_reorder.sacc.fits"))

    msgs = "\n".join(str(w.message) for w in record)
    assert "('A', 'B')" in msgs  # names the block
    assert "'A'" in msgs  # names the reordered component axis
    assert ("realign" in msgs) or ("reorder" in msgs)


def test_save_no_warning_when_already_canonical():
    """No reorder => no warning."""
    import warnings

    a0 = ("cl_00", ("A", "A"), 0.0)
    a1 = ("cl_00", ("A", "A"), 1.0)
    b0 = ("cl_00", ("B", "B"), 0.0)

    cc = CrossCov()
    cc.add_component("A", np.eye(2), ids=[a0, a1])
    cc.add_component("B", np.eye(1), ids=[b0])
    cc.add_cross_covariance("A", "B", np.ones((2, 1)), ids1=[a0, a1], ids2=[b0])

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning becomes an error
        cc.save(os.path.join(gettempdir(), "warn_none.sacc.fits"))


def test_save_keeps_cross_only_components_when_mixed_with_add_component():
    """Regression: a component present only via add_cross_covariance must not be
    dropped from the saved file just because *another* component was registered
    with add_component. Previously save() inferred missing components only when
    no component had been added explicitly, silently truncating mixed stores.
    """
    from soliket.gaussian.gaussian_data import CrossCov, GaussianData, MultiGaussianData

    ids_A = [("cl_00", ("A", "A"), float(i)) for i in range(3)]
    ids_B = [("cl_00", ("B", "B"), float(i)) for i in range(2)]
    # Small off-diagonal so the assembled joint covariance stays positive definite.
    cross_AB = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])

    # A has an auto; B is cross-only (its auto will be supplied live at runtime).
    cc = CrossCov()
    cc.add_component("A", np.diag([1.0, 2.0, 3.0]), ids=ids_A)
    cc.add_cross_covariance("A", "B", cross_AB, ids1=ids_A, ids2=ids_B)

    path = os.path.join(gettempdir(), "mixed_cross_only.sacc.fits")
    cc.save(path)

    loaded = CrossCov.load(path)
    assert set(loaded.component_names) == {"A", "B"}  # B not dropped
    assert ("A", "B") in loaded  # cross term survived
    assert np.allclose(loaded[("A", "B")], cross_AB)

    # B is cross-only: assembling with a live B falls back to B's own covariance,
    # and the cross block is placed by identity.
    cov_B = np.diag([10.0, 20.0])
    d_A = GaussianData("A", np.arange(3.0), np.zeros(3), np.diag([1.0, 2.0, 3.0]),
                       ids=ids_A)
    d_B = GaussianData("B", np.arange(2.0), np.zeros(2), cov_B, ids=ids_B)
    cov = MultiGaussianData([d_A, d_B], loaded).cov
    assert np.allclose(cov[3:, 3:], cov_B)  # B auto from the live likelihood
    assert np.allclose(cov[:3, 3:], cross_AB)  # cross term placed correctly


def test_cross_only_file_reconciles_at_runtime():
    """The 'omit autos, store only the cross term' workflow (as documented in
    notebook 04): a cross-only file, saved with the cross block in a different
    order than the live data, must round-trip and then -- at MultiGaussianData
    assembly -- take both autos from the live likelihoods and realign the cross
    block to the data order by identity.
    """
    from soliket.gaussian.gaussian_data import CrossCov, GaussianData, MultiGaussianData

    ids_A = [("cl_00", ("A", "A"), float(i)) for i in range(3)]
    ids_B = [("cl_00", ("B", "B"), float(i)) for i in range(2)]
    cov_A = np.diag([1.0, 2.0, 3.0])
    cov_B = np.diag([10.0, 20.0])
    cross_data = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])  # in DATA order

    # Cross-only: NO add_component. Store the cross block in a PERMUTED A order.
    perm = [2, 0, 1]
    assert not np.allclose(cross_data[perm, :], cross_data)  # realignment matters
    cc = CrossCov()
    cc.add_cross_covariance(
        "A", "B", cross_data[perm, :], ids1=[ids_A[i] for i in perm], ids2=ids_B
    )

    path = os.path.join(gettempdir(), "cross_only_runtime.sacc.fits")
    cc.save(path)

    loaded = CrossCov.load(path)
    assert set(loaded.component_names) == {"A", "B"}
    assert ("A", "A") not in loaded and ("B", "B") not in loaded  # no autos stored

    d_A = GaussianData("A", np.arange(3.0), np.zeros(3), cov_A, ids=ids_A)
    d_B = GaussianData("B", np.arange(2.0), np.zeros(2), cov_B, ids=ids_B)
    cov = MultiGaussianData([d_A, d_B], loaded).cov

    assert np.allclose(cov[:3, :3], cov_A)  # auto A from the live likelihood
    assert np.allclose(cov[3:, 3:], cov_B)  # auto B from the live likelihood
    assert np.allclose(cov[:3, 3:], cross_data)  # cross realigned to data order
    assert np.allclose(cov[3:, :3], cross_data.T)

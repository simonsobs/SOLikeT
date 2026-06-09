import os
from tempfile import gettempdir

import numpy as np
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

    cross_cov.add_cross_covariance(name1, name2, full_cov[:n1, n1 : n1 + n2])
    cross_cov.add_cross_covariance(name1, name3, full_cov[:n1, n1 + n2 :])
    cross_cov.add_cross_covariance(name2, name3, full_cov[n1 : n1 + n2, n1 + n2 :])

    # Add required metadata for SACC format
    tracer_info = {
        name1: {"name": name1, "quantity": "cmb_temperature", "spin": 0},
        name2: {"name": name2, "quantity": "cmb_temperature", "spin": 0},
        name3: {"name": name3, "quantity": "cmb_temperature", "spin": 0},
    }

    # Add metadata for each cross-covariance block
    cross_cov.add_metadata(
        key=(name1, name2),
        tracers=((name1, name1), (name2, name2)),
        data_types=("cl_00", "cl_00"),
        tracer_info=tracer_info,
    )
    cross_cov.add_metadata(
        key=(name1, name3),
        tracers=((name1, name1), (name3, name3)),
        data_types=("cl_00", "cl_00"),
        tracer_info=tracer_info,
    )
    cross_cov.add_metadata(
        key=(name2, name3),
        tracers=((name2, name2), (name3, name3)),
        data_types=("cl_00", "cl_00"),
        tracer_info=tracer_info,
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

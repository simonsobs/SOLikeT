import os
from tempfile import gettempdir

import numpy as np
import sacc
from sklearn.datasets import make_spd_matrix

from soliket import MultiGaussianLikelihood, PSLikelihood
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
    _allowable_tracers = "cmb_temperature"

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

    cross_cov = CrossCov(
        {
            (name1, name2): full_cov[:n1, n1 : n1 + n2],
            (name1, name3): full_cov[:n1, n1 + n2 :],
            (name2, name3): full_cov[n1 : n1 + n2, n1 + n2 :],
        }
    )

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

    lhood = "tests.test_ps.ToyLikelihood"
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


class DummyProviderCl:
    def __init__(self, lmax):
        self.lmax = lmax

    def get_Cl(self, ell_factor=True):
        # return small arrays for pp, tt, ee, te, bb
        size = self.lmax
        return {
            "pp": np.arange(size, dtype=float) + 1.0,
            "tt": (np.arange(size, dtype=float) + 2.0),
            "ee": (np.arange(size, dtype=float) + 3.0),
            "te": (np.arange(size, dtype=float) + 4.0),
            "bb": (np.arange(size, dtype=float) + 5.0),
        }


def test_psl_get_theory_basic():
    lmax = 4
    pl = PSLikelihood.__new__(PSLikelihood)
    pl.provider = DummyProviderCl(lmax)
    pl.kind = "tt"
    pl.lmax = lmax
    out = PSLikelihood._get_theory(pl)
    assert np.allclose(out, np.arange(lmax, dtype=float) + 2.0)

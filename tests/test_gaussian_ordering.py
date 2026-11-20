"""Ordering consistency between the data vector and the theory vector.

``GaussianLikelihood`` builds:

* ``self.x`` (ell bins) and the theory vector (in every ``_get_theory``) by
  iterating ``sacc.get_tracer_combinations()`` -> **combo-major** order, and
* ``self.y`` / ``self.cov`` straight from ``sacc.mean`` / ``covmat`` -> **sacc
  storage order**.

These two orderings only agree when the sacc file stores its data points grouped
by tracer-combination. When they do not (e.g. combos interleaved in storage),
``loglike`` silently subtracts a storage-order data vector from a combo-major
theory vector, producing a wrong chi-squared. These tests pin the invariant that
``y``/``cov`` must be in the same order the theory is built in.
"""

import os
from tempfile import gettempdir

import numpy as np
import sacc

from soliket.gaussian.gaussian import GaussianLikelihood
from soliket.utils import get_likelihood

# Deterministic, unique data value per (tracer-combination, ell).
COMBO_IDX = {("A", "A"): 0, ("B", "B"): 1}


def true_value(comb, ell):
    return COMBO_IDX[tuple(comb)] * 1000.0 + float(ell)


class OrderSensitiveToy(GaussianLikelihood):
    name = "order_toy"
    _allowable_tracers = ("cmb_temperature",)

    def _get_theory(self, **params_values):
        # Built combo-major, exactly like the real likelihoods (ccl_tracers,
        # cross_correlation, ...): loop get_tracer_combinations(), concatenate.
        vals = []
        for comb in self.sacc_data.get_tracer_combinations():
            ind = self.sacc_data.indices(tracers=comb)
            ells = self.sacc_data._get_tags_by_index(["ell"], ind)[0]
            vals.extend(true_value(comb, ell) for ell in ells)
        return np.array(vals)


def _make_interleaved_sacc(path):
    """Two auto-spectra whose points are INTERLEAVED in storage order."""
    s = sacc.Sacc()
    s.add_tracer("Misc", "A", quantity="cmb_temperature", spin=0)
    s.add_tracer("Misc", "B", quantity="cmb_temperature", spin=0)
    # storage order: A,B,A,B,A  -> NOT grouped by combo
    layout = [
        ("A", "A", 10),
        ("B", "B", 200),
        ("A", "A", 11),
        ("B", "B", 201),
        ("A", "A", 12),
    ]
    for t1, t2, ell in layout:
        s.add_data_point("cl_00", (t1, t2), true_value((t1, t2), ell), ell=float(ell))
    # distinctive covariance so a mis-ordering is detectable
    n = len(layout)
    rng = np.random.default_rng(0)
    cov = rng.standard_normal((n, n))
    cov = cov @ cov.T + n * np.eye(n)
    s.add_covariance(cov)
    s.save_fits(path, overwrite=True)
    return s, cov, layout


def test_data_vector_order_matches_theory_order():
    """delta = y - theory must be zero when the theory equals the data.

    With interleaved storage, ``y`` is storage-order while the theory is
    combo-major, so the per-element subtraction is misaligned and chi^2 != 0.
    """
    tmp = gettempdir()
    path = os.path.join(tmp, "interleaved.sacc.fits")
    _make_interleaved_sacc(path)

    like = get_likelihood(
        "tests.test_gaussian_ordering.OrderSensitiveToy",
        {"name": "order_toy", "datapath": path, "use_spectra": "all"},
    )

    theory = like._get_theory()
    # The data value of each point equals true_value(combo, ell), so a correctly
    # ordered data vector is identical to the (combo-major) theory vector.
    assert np.allclose(like.data.y, theory), (
        "data vector is not in the same (combo-major) order as the theory vector"
    )

    chi2 = -2.0 * (like.logp() - like.data.norm_const)
    assert np.isclose(chi2, 0.0), f"chi^2={chi2} != 0: y/cov order inconsistent"


def test_covariance_reordered_consistently():
    """The covariance must be permuted into the same combo-major order as y."""
    tmp = gettempdir()
    path = os.path.join(tmp, "interleaved.sacc.fits")
    s, cov_storage, layout = _make_interleaved_sacc(path)

    like = get_likelihood(
        "tests.test_gaussian_ordering.OrderSensitiveToy",
        {"name": "order_toy", "datapath": path, "use_spectra": "all"},
    )

    # Combo-major permutation of the storage-order covariance.
    perm = np.concatenate([s.indices(tracers=c) for c in s.get_tracer_combinations()])
    expected_cov = cov_storage[np.ix_(perm, perm)]

    assert np.allclose(like.data.cov, expected_cov), (
        "covariance is not in the combo-major order used by x/theory"
    )

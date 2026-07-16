"""Option type-validation is enabled (``_enforce_types``) for xcorr and ccl_tracers.

These components declare their options as typed class attributes; with
``_enforce_types`` set, cobaya type-checks the attribute values at init, so a
mistyped option is caught instead of silently accepted.
"""

import pytest

from soliket.ccl_tracers.ccl_tracers import CCLTracersLikelihood
from soliket.cosmopower.cosmopower import CosmoPower, CosmoPowerDerived
from soliket.xcorr.xcorr import XcorrLikelihood


def _validate_with(cls, **attrs):
    """Run cobaya's attribute validation on a bare instance with the given attrs."""
    inst = cls.__new__(cls)
    for name, value in attrs.items():
        setattr(inst, name, value)
    inst.validate_attributes(cls.get_annotations())


def test_xcorr_enforces_option_types():
    assert XcorrLikelihood._enforce_types is True
    # high_ell is annotated int | None; a string must be rejected.
    with pytest.raises(TypeError):
        _validate_with(XcorrLikelihood, high_ell="not-an-int")


def test_ccl_tracers_enforces_option_types():
    assert CCLTracersLikelihood._enforce_types is True
    # ncovsims is annotated int | None; a string must be rejected.
    with pytest.raises(TypeError):
        _validate_with(CCLTracersLikelihood, ncovsims="not-an-int")


def test_cosmopower_enforces_option_types():
    assert CosmoPower._enforce_types is True
    # network_path is annotated str; an int must be rejected.
    with pytest.raises(TypeError):
        _validate_with(CosmoPower, network_path=123)


def test_cosmopower_derived_enforces_option_types():
    assert CosmoPowerDerived._enforce_types is True
    # derived_parameters is annotated list[str]; an int must be rejected.
    with pytest.raises(TypeError):
        _validate_with(CosmoPowerDerived, derived_parameters=123)

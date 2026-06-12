"""Resolve role aliases on a Cobaya model.

Replaces fragile positional ``model.components[i]`` access with named roles
(``.mflike``, ``.lensing``, ``.foreground``, ``.cosmo``), matched by class so they
survive reordering and recurse into ``MultiGaussianLikelihood.likelihoods``.
"""

from cobaya.theories.camb.camb import CAMB
from cobaya.theories.classy import classy
from mflike import Foreground
from mflike.mflike import _MFLike

from ..gaussian import MultiGaussianLikelihood
from ..lensing import LensingLikelihood, LensingLiteLikelihood

# role -> the classes whose instances fill that role
_ROLE_CLASSES = {
    "mflike": (_MFLike,),
    "lensing": (LensingLikelihood, LensingLiteLikelihood),
    "foreground": (Foreground,),
    "cosmo": (CAMB, classy),
}


class AliasView:
    """A model's Likelihood/Theory members, reachable by role name."""

    def __init__(self, roles):
        for role in _ROLE_CLASSES:
            setattr(self, role, roles.get(role))


def _walk(component):
    """Yield a component and, for a MultiGaussianLikelihood, its sub-likelihoods."""
    if isinstance(component, MultiGaussianLikelihood):
        yield from component.likelihoods
    else:
        yield component


def resolve_aliases(model):
    """Return an :class:`AliasView` of ``model``'s components keyed by role."""
    roles = {}
    for component in model.components:
        for member in _walk(component):
            for role, classes in _ROLE_CLASSES.items():
                if role not in roles and isinstance(member, classes):
                    roles[role] = member
    return AliasView(roles)

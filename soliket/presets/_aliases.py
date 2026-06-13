"""Resolve role aliases on a Cobaya model.

Replaces fragile positional ``model.components[i]`` access with named roles
(``.mflike``, ``.lensing``, ``.foreground``, ``.cosmo``), matched by class so they
survive reordering and recurse into ``MultiGaussianLikelihood.likelihoods``.
"""

from cobaya.theories.camb.camb import CAMB
from cobaya.theories.classy import classy

from ..gaussian import MultiGaussianLikelihood
from ..lensing import LensingLikelihood, LensingLiteLikelihood

# Role names, in priority order. The class tuple for each role is resolved lazily
# (see ``_role_classes``) so importing ``soliket.presets`` does not require the
# optional ``mflike`` package -- only resolving aliases on a real model does.
_ROLES = ("mflike", "lensing", "foreground", "cosmo")


def _role_classes():
    """Map each role to the classes whose instances fill it.

    Imports ``mflike`` lazily: it is an optional dependency, so only callers that
    actually resolve aliases on a built model need it installed.
    """
    from mflike import Foreground
    from mflike.mflike import _MFLike

    return {
        "mflike": (_MFLike,),
        "lensing": (LensingLikelihood, LensingLiteLikelihood),
        "foreground": (Foreground,),
        "cosmo": (CAMB, classy),
    }


class AliasView:
    """A model's Likelihood/Theory members, reachable by role name."""

    def __init__(self, roles):
        for role in _ROLES:
            setattr(self, role, roles.get(role))


def _walk(component):
    """Yield a component and, for a MultiGaussianLikelihood, its sub-likelihoods."""
    if isinstance(component, MultiGaussianLikelihood):
        yield from component.likelihoods
    else:
        yield component


def resolve_aliases(model):
    """Return an :class:`AliasView` of ``model``'s components keyed by role."""
    role_classes = _role_classes()
    roles = {}
    for component in model.components:
        for member in _walk(component):
            for role, classes in role_classes.items():
                if role not in roles and isinstance(member, classes):
                    roles[role] = member
    return AliasView(roles)

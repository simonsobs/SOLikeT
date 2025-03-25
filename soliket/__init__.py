from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("soliket")
except PackageNotFoundError:
    # package is not installed
    pass

from .bias import Bias, Linear_bias
from .ccl import CCL
from .clusters import ClusterLikelihood
from .cosmopower import CosmoPower, CosmoPowerDerived
from .cross_correlation import (CrossCorrelationLikelihood,
                                GalaxyKappaLikelihood, ShearKappaLikelihood)
from .gaussian import GaussianLikelihood, MultiGaussianLikelihood
from .lensing import LensingLikelihood, LensingLiteLikelihood
from .ps import BinnedPSLikelihood, PSLikelihood
from .xcorr import XcorrLikelihood

__all__ = [
    # bias
    "Bias",
    "Linear_bias",
    # ccl
    "CCL",
    # clusters
    "ClusterLikelihood",
    # cosmopower
    "CosmoPower",
    "CosmoPowerDerived",
    # cross_correlation
    "CrossCorrelationLikelihood",
    "GalaxyKappaLikelihood",
    "ShearKappaLikelihood",
    # gaussian
    "GaussianLikelihood",
    "MultiGaussianLikelihood",
    # lensing
    "LensingLikelihood",
    "LensingLiteLikelihood",
    # ps
    "BinnedPSLikelihood",
    "PSLikelihood",
    # xcorr
    "XcorrLikelihood",
]

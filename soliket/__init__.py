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
from .cross_correlation import (
    CrossCorrelationLikelihood,
    GalaxyKappaLikelihood,
    ShearKappaLikelihood,
)
from .gaussian import GaussianLikelihood, MultiGaussianLikelihood
from .lensing import LensingLikelihood, LensingLiteLikelihood
from .ps import BinnedPSLikelihood, PSLikelihood
from .xcorr import XcorrLikelihood

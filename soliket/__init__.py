from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("soliket")
except PackageNotFoundError:
    # package is not installed
    pass

from .bias import Bias, Linear_bias
from .ccl import CCL
from .ccl_tracers import (
    CCLTracersLikelihood,
    GalaxyKappaLikelihood,
    ShearKappaLikelihood,
)
from .clusters import ClusterLikelihood
from .cosmopower import CosmoPower, CosmoPowerDerived
from .gaussian import GaussianLikelihood, MultiGaussianLikelihood
from .lensing import LensingLikelihood, LensingLiteLikelihood
from .xcorr import XcorrLikelihood

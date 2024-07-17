from soliket._version import __version__

from .bandpass import BandPass
from .bias import Bias, Linear_bias
from .ccl import CCL
from .clusters import ClusterLikelihood
from .cosmopower import CosmoPower, CosmoPowerDerived
from .cross_correlation import (CrossCorrelationLikelihood,
                                GalaxyKappaLikelihood, ShearKappaLikelihood)
from .foreground import Foreground
from .gaussian import GaussianLikelihood, MultiGaussianLikelihood
from .lensing import LensingLikelihood, LensingLiteLikelihood
from .mflike import MFLike, TheoryForge_MFLike
# from .studentst import StudentstLikelihood
from .ps import BinnedPSLikelihood, PSLikelihood
from .xcorr import XcorrLikelihood

__all__ = [
    # bandpass
    "BandPass",
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
    # foreground
    "Foreground",
    # gaussian
    "GaussianLikelihood",
    "MultiGaussianLikelihood",
    # lensing
    "LensingLikelihood",
    "LensingLiteLikelihood",
    # mflike
    "MFLike",
    "TheoryForge_MFLike",
    # studentst
    # "StudentstLikelihood",
    # ps
    "BinnedPSLikelihood",
    "PSLikelihood",
    # xcorr
    "XcorrLikelihood",
]

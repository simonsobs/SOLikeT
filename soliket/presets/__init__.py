"""Ready-to-run SOLikeT configurations for notebooks and quickstart use."""

from ._aliases import AliasView, resolve_aliases
from ._loader import build_params, load_fiducial_map, load_params
from ._session import PRESETS, Session, build_info, quickstart

__all__ = [
    "PRESETS",
    "AliasView",
    "Session",
    "build_info",
    "build_params",
    "load_fiducial_map",
    "load_params",
    "quickstart",
    "resolve_aliases",
]

"""Tests for the repo-only notebook helpers in notebooks/_nb.py.

The module is not part of the shipped package, so it is loaded from its file
path rather than imported.
"""

import importlib.util
import pathlib
from types import SimpleNamespace

import matplotlib
import numpy as np

matplotlib.use("Agg")

_NB_PATH = pathlib.Path(__file__).parent.parent / "notebooks" / "_nb.py"


def _load_nb():
    spec = importlib.util.spec_from_file_location("_nb", _NB_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_theory_dls_reads_from_the_model_provider():
    sentinel = {"ell": [2, 3], "tt": [1.0, 2.0]}
    session = SimpleNamespace(
        model=SimpleNamespace(
            provider=SimpleNamespace(get_Cl=lambda ell_factor=True: sentinel)
        )
    )

    assert _load_nb().theory_dls(session) is sentinel


def test_foreground_totals_reads_from_the_foreground_role():
    sentinel = object()
    session = SimpleNamespace(
        foreground=SimpleNamespace(get_fg_totals=lambda: sentinel)
    )

    assert _load_nb().foreground_totals(session) is sentinel


def test_plot_dls_draws_one_line_per_requested_spectrum():
    nb = _load_nb()
    ell = np.arange(2, 100)
    dls = {"ell": ell, "tt": ell * 1.0, "te": ell * 0.5, "ee": ell * 0.1}

    ax = nb.plot_dls(dls, spectra=("tt", "ee"))

    assert len(ax.get_lines()) == 2
    assert ax.get_xscale() == "log"

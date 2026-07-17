"""Notebook-only conveniences for SOLikeT tutorials.

Repo-only sugar (not part of the shipped ``soliket`` package): resolving the
packages path, triggering data installation, pulling spectra off a
:class:`soliket.presets.Session`, and quick plotting. The reusable API lives in
``soliket.presets``; this module is just glue to keep the notebooks short.
"""

import numpy as np

# Spectra Cobaya can return from get_Cl, in a sensible plotting order.
_DEFAULT_SPECTRA = ("tt", "te", "ee")


def packages_path():
    """Cobaya's resolved packages path (where installed likelihood data lives)."""
    from cobaya.tools import resolve_packages_path

    return resolve_packages_path()


def install_data(preset, path=None, **kwargs):
    """Download the data a preset's likelihoods need, via ``cobaya.install``."""
    from cobaya.install import install

    from soliket.presets import build_info

    return install(build_info(preset), path=path or packages_path(), **kwargs)


def theory_dls(session, ell_factor=True):
    """CMB :math:`D_\\ell` spectra for an evaluated session (a ``get_Cl`` dict).

    Call ``session.loglike()`` first so the provider has computed the point.
    """
    return session.model.provider.get_Cl(ell_factor=ell_factor)


def foreground_totals(session):
    """Total foreground bandpowers for the session's BandpowerForeground theory."""
    return session.foreground.get_fg_totals()


def plot_dls(dls, spectra=_DEFAULT_SPECTRA, ax=None):
    """Log-log plot of ``|D_\\ell|`` for the requested spectra; returns the Axes."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots()
    ell = dls["ell"]
    for spec in spectra:
        ax.loglog(ell, np.abs(dls[spec]), label=spec.upper())
    ax.set_xlabel(r"$\ell$")
    ax.set_ylabel(r"$|D_\ell|\ [\mu K^2]$")
    ax.legend()
    return ax

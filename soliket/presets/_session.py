"""Presets and the running layer: assemble a Cobaya ``info`` dict and drive it.

A *preset* names a ready-to-run configuration (which likelihood, which theory
blocks, which Fiducial-map groups). ``build_info`` turns a preset into a Cobaya
``info`` dict; sampling runs Python-native via ``cobaya.run`` (no run YAML).
"""

import copy
from importlib import resources

import yaml

from ._aliases import resolve_aliases
from ._loader import load_params

# Single-massive neutrino baseline (m=0.06, Neff=3.044), keyed by Boltzmann code.
# Transcribed from cobaya's cosmo_input ``one_heavy_planck`` as a starting point;
# edit it here to change the preset's neutrino setup. The SO normal-hierarchy setup
# is an ISO-sims override, not a preset default.
_ONE_HEAVY = {
    "camb": {
        "extra_args": {"num_massive_neutrinos": 1, "nnu": 3.044},
        "params": {"mnu": 0.06},
    },
    "classy": {
        "extra_args": {"N_ncdm": 1, "N_ur": 2.0328},
        "params": {"m_ncdm": {"value": 0.06, "renames": "mnu"}},
    },
}

# preset name -> info template file + the Fiducial-map groups it needs
PRESETS = {
    "mflike": {
        "template": "mflike.yaml",
        "groups": ["cosmo", "foreground", "systematics"],
    },
    "lensing": {
        "template": "lensing.yaml",
        "groups": ["cosmo"],
    },
    "multigaussian": {
        "template": "multigaussian.yaml",
        "groups": ["cosmo", "foreground", "systematics"],
    },
}


def _load_template(filename):
    text = resources.files(__package__).joinpath("templates", filename).read_text()
    return yaml.safe_load(text)


def build_info(preset, sample=None, theory="camb", params_dir=None):
    """Assemble the Cobaya ``info`` dict for ``preset``.

    ``sample`` is the explicit list of dual parameters to vary; the rest are fixed
    to their fiducial values. ``theory`` selects the Boltzmann solver: ``"camb"``
    (default) or ``"classy"``; both get the single-massive neutrino baseline
    (see :func:`_apply_neutrinos` / :data:`_ONE_HEAVY`). ``params_dir`` optionally points
    at a directory of override ``<group>.yaml`` files (per-file fallback to the
    bundled defaults). A relative ``params_dir`` is resolved against the process
    working directory. Returns a fresh dict each call.
    """
    if preset not in PRESETS:
        raise ValueError(f"unknown preset {preset!r}; choose from {sorted(PRESETS)}")
    if theory not in _ONE_HEAVY:
        raise ValueError(f"unknown theory {theory!r}; choose from {sorted(_ONE_HEAVY)}")
    spec = PRESETS[preset]
    info = _load_template(spec["template"])
    info["params"] = load_params(
        sample=sample, groups=spec["groups"], params_dir=params_dir
    )
    _apply_neutrinos(info, theory)
    return info


def _apply_neutrinos(info, theory):
    """Inject the :data:`_ONE_HEAVY` neutrino sub-block for ``theory`` in place.

    Single massive neutrino (m=0.06, Neff=3.044). Injected with ``setdefault`` so
    anything the preset/override already pins wins — e.g. the ISO-sims
    normal-hierarchy override that sets ``mnu`` and its own camb ``extra_args``.
    For ``classy``, drops the camb Boltzmann block (its precision ``extra_args`` do
    not translate) while preserving non-Boltzmann theory entries (e.g.
    ``mflike.BandpowerForeground``).
    """
    nu = copy.deepcopy(_ONE_HEAVY[theory])
    if theory == "classy":
        info["theory"].pop("camb", None)
        info["theory"].setdefault("classy", {"stop_at_error": True})
    block = info["theory"][theory]
    extra = block.setdefault("extra_args", {})
    for key, value in nu.get("extra_args", {}).items():
        extra.setdefault(key, value)
    for name, pspec in nu.get("params", {}).items():
        info["params"].setdefault(name, pspec)


class Session:
    """A wired Cobaya model plus its role aliases and Fiducial map.

    Exposes the model's Likelihood/Theory members by role (``.mflike``,
    ``.lensing``, ``.foreground``, ``.cosmo``), the ``info`` dict it was built
    from, and the fiducial ``params`` dict. ``run()`` samples Python-native via
    ``cobaya.run``.
    """

    def __init__(self, info, model):
        self.info = info
        self.model = model
        self.fiducial = info["params"]
        roles = resolve_aliases(model)
        self.mflike = roles.mflike
        self.lensing = roles.lensing
        self.foreground = roles.foreground
        self.cosmo = roles.cosmo

    def loglike(self, point=None):
        """Total log-likelihood at ``point`` (default: the fiducial point)."""
        loglikes, _ = self.model.loglikes(point or {})
        return float(sum(loglikes))

    def run(self, **kwargs):
        """Run the preset with Cobaya's Python API; forwards to ``cobaya.run``."""
        from cobaya import run

        return run(self.info, **kwargs)


def quickstart(
    preset, *, sample=None, theory="camb", packages_path=None, params_dir=None
):
    """Build a ready-to-use :class:`Session` for ``preset``.

    ``sample`` lists the dual parameters to vary (default: all fixed, ready to
    evaluate at the fiducial point). ``theory`` selects the Boltzmann solver
    (``"camb"`` or ``"classy"``). ``packages_path`` overrides
    Cobaya's default location for installed likelihood data. ``params_dir``
    optionally points at a directory of override ``<group>.yaml`` files (per-file
    fallback to the bundled defaults). A relative ``params_dir`` is resolved
    against the process working directory.
    """
    from cobaya.model import get_model
    from cobaya.tools import resolve_packages_path

    info = build_info(preset, sample=sample, theory=theory, params_dir=params_dir)
    info["packages_path"] = packages_path or resolve_packages_path()
    return Session(info, get_model(info))

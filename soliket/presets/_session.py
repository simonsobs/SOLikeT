"""Presets and the running layer: assemble a Cobaya ``info`` dict and drive it.

A *preset* names a ready-to-run configuration (which likelihood, which theory
blocks, which Fiducial-map groups). ``build_info`` turns a preset into a Cobaya
``info`` dict; sampling runs Python-native via ``cobaya.run`` (no run YAML).
"""

from importlib import resources

import yaml
from cobaya.tools import recursive_update

from ._aliases import resolve_aliases
from ._loader import load_params, load_theory

# Boltzmann codes the presets can target; the neutrino baseline for each lives in
# the Fiducial map's ``theory.yaml`` (folder-overridable), not here.
_THEORY_CODES = ("camb", "classy")

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


def build_info(preset, sample=None, theory="camb", defaults_dir=None):
    """Assemble the Cobaya ``info`` dict for ``preset``.

    ``sample`` is the explicit list of dual parameters to vary; the rest are fixed
    to their fiducial values. ``theory`` selects the Boltzmann solver: ``"camb"``
    (default) or ``"classy"``. ``defaults_dir`` optionally points at a directory of
    override files (``<group>.yaml`` for the param groups and ``theory.yaml`` for
    the neutrino sector), each with per-file fallback to the bundled defaults. A
    relative ``defaults_dir`` is resolved against the process working directory.
    Returns a fresh dict each call.
    """
    if preset not in PRESETS:
        raise ValueError(f"unknown preset {preset!r}; choose from {sorted(PRESETS)}")
    if theory not in _THEORY_CODES:
        raise ValueError(
            f"unknown theory {theory!r}; choose from {sorted(_THEORY_CODES)}"
        )
    spec = PRESETS[preset]
    info = _load_template(spec["template"])
    info["params"] = load_params(
        sample=sample, groups=spec["groups"], defaults_dir=defaults_dir
    )
    _apply_theory(info, theory, defaults_dir=defaults_dir)
    return info


def _apply_theory(info, theory, defaults_dir=None):
    """Overlay the Fiducial map's theory fragment (``theory.yaml``) for ``theory``.

    The skeleton template owns the preset-specific precision ``extra_args``; this
    layers the neutrino-sector ``extra_args`` on top via cobaya's ``recursive_update``
    (last wins). The neutrino baseline is wholesale-replaceable per-file: a folder
    ``theory.yaml`` supersedes the bundled one, so a different neutrino setup (e.g.
    the ISO normal hierarchy) never collides with the packaged single-massive keys.

    For ``classy``, drops the camb Boltzmann block (its precision ``extra_args`` do
    not translate) while preserving non-Boltzmann theory entries (e.g.
    ``mflike.BandpowerForeground``), and swaps the camb-native ``mnu`` param for the
    classy-native ``m_ncdm`` carried in ``theory.yaml``.
    """
    block = load_theory(defaults_dir).get(theory, {})
    if theory == "classy":
        info["theory"].pop("camb", None)
        info["theory"].setdefault("classy", {"stop_at_error": True})
        info["params"].pop("mnu", None)  # classy uses m_ncdm (from theory.yaml)
        for name, pspec in (block.get("params") or {}).items():
            info["params"][name] = pspec
    info["theory"][theory] = recursive_update(
        info["theory"].get(theory, {}), {"extra_args": block.get("extra_args", {})}
    )


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
    preset, *, sample=None, theory="camb", packages_path=None, defaults_dir=None
):
    """Build a ready-to-use :class:`Session` for ``preset``.

    ``sample`` lists the dual parameters to vary (default: all fixed, ready to
    evaluate at the fiducial point). ``theory`` selects the Boltzmann solver
    (``"camb"`` or ``"classy"``). ``packages_path`` overrides
    Cobaya's default location for installed likelihood data. ``defaults_dir``
    optionally points at a directory of override ``<group>.yaml`` files (per-file
    fallback to the bundled defaults). A relative ``defaults_dir`` is resolved
    against the process working directory.
    """
    from cobaya.model import get_model
    from cobaya.tools import resolve_packages_path

    info = build_info(preset, sample=sample, theory=theory, defaults_dir=defaults_dir)
    info["packages_path"] = packages_path or resolve_packages_path()
    return Session(info, get_model(info))

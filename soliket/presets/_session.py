"""Presets and the running layer: assemble a Cobaya ``info`` dict and drive it.

A *preset* names a ready-to-run configuration (which likelihood, which theory
blocks, which Fiducial-map groups). ``build_info`` turns a preset into a Cobaya
``info`` dict; sampling runs Python-native via ``cobaya.run`` (no run YAML).
"""

from importlib import resources

import yaml

from ._aliases import resolve_aliases
from ._loader import load_params

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


# Provisional CLASS Boltzmann block used when theory="classy". The cosmo params
# route to CLASS via Cobaya's renames, but CAMB-specific neutrino params
# (mnu1/2/3, nnu) do not translate and a real CLASS run is not yet validated.
_CLASSY_THEORY = {"classy": {"stop_at_error": True}}


def build_info(preset, sample=None, theory="camb", params_dir=None):
    """Assemble the Cobaya ``info`` dict for ``preset``.

    ``sample`` is the explicit list of dual parameters to vary; the rest are fixed
    to their fiducial values. ``theory`` selects the Boltzmann solver: ``"camb"``
    (default) or ``"classy"`` (provisional — see :data:`_CLASSY_THEORY`).
    ``params_dir`` optionally points at a directory of override ``<group>.yaml``
    files (per-file fallback to the bundled defaults). Returns a fresh dict each
    call.
    """
    if preset not in PRESETS:
        raise ValueError(f"unknown preset {preset!r}; choose from {sorted(PRESETS)}")
    spec = PRESETS[preset]
    info = _load_template(spec["template"])
    if theory == "classy":
        info["theory"].pop("camb", None)
        info["theory"] = {**_CLASSY_THEORY, **info["theory"]}
    info["params"] = load_params(
        sample=sample, groups=spec["groups"], params_dir=params_dir
    )
    return info


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
    (``"camb"`` or provisional ``"classy"``). ``packages_path`` overrides
    Cobaya's default location for installed likelihood data. ``params_dir``
    optionally points at a directory of override ``<group>.yaml`` files (per-file
    fallback to the bundled defaults).
    """
    from cobaya.model import get_model
    from cobaya.tools import resolve_packages_path

    info = build_info(preset, sample=sample, theory=theory, params_dir=params_dir)
    info["packages_path"] = packages_path or resolve_packages_path()
    return Session(info, get_model(info))

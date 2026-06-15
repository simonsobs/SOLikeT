"""Presets and the running layer: assemble a Cobaya ``info`` dict and drive it.

A *preset* names a ready-to-run configuration (which likelihood, which theory
blocks, which Fiducial-map groups). ``build_info`` turns a preset into a Cobaya
``info`` dict; sampling runs Python-native via ``cobaya.run`` (no run YAML).
"""

from importlib import resources
from pathlib import Path

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
        # Per-component options and precision are composed from these member
        # presets (single source of truth), not duplicated in the skeleton.
        "members": ["mflike", "lensing"],
    },
}


def _load_template(filename, defaults_dir=None):
    """Load the packaged info skeleton, optionally overlaid by a folder template.

    Unlike the param groups (wholesale per-file replacement), a folder
    ``defaults_dir/templates/<filename>`` is layered onto the packaged skeleton via
    cobaya's ``recursive_update`` (last wins): it patches only the keys it sets --
    e.g. a single ``theory_lmax`` -- and inherits the rest, so it tracks future
    package-template changes. Absent the file, the packaged skeleton is used as-is.
    """
    text = resources.files(__package__).joinpath("templates", filename).read_text()
    info = yaml.safe_load(text)
    if defaults_dir is not None:
        override = Path(defaults_dir) / "templates" / filename
        if override.is_file():
            info = recursive_update(info, yaml.safe_load(override.read_text()))
    return info


def build_info(preset, sample=None, theory="camb", defaults_dir=None):
    """Assemble the Cobaya ``info`` dict for ``preset``.

    ``sample`` is the explicit list of dual parameters to vary; the rest are fixed
    to their fiducial values. ``theory`` selects the Boltzmann solver: ``"camb"``
    (default) or ``"classy"``. ``defaults_dir`` optionally points at a directory of
    override files: ``<group>.yaml`` for the param groups and ``theory.yaml`` for
    the neutrino sector (per-file wholesale replacement, fallback to the bundled
    defaults), plus ``templates/<preset>.yaml`` to overlay likelihood/theory options
    onto the packaged skeleton (``recursive_update``, last wins). A relative
    ``defaults_dir`` is resolved against the process working directory. Returns a
    fresh dict each call.
    """
    if preset not in PRESETS:
        raise ValueError(f"unknown preset {preset!r}; choose from {sorted(PRESETS)}")
    if theory not in _THEORY_CODES:
        raise ValueError(
            f"unknown theory {theory!r}; choose from {sorted(_THEORY_CODES)}"
        )
    spec = PRESETS[preset]
    info = _load_template(spec["template"], defaults_dir=defaults_dir)
    if "members" in spec:
        _compose_members(info, spec["members"], defaults_dir=defaults_dir)
    info["params"] = load_params(
        sample=sample, groups=spec["groups"], defaults_dir=defaults_dir
    )
    _apply_theory(info, theory, defaults_dir=defaults_dir)
    return info


def _compose_members(info, members, defaults_dir=None):
    """Fill a multi-component preset's per-component options and theory from its
    member presets, so each member's config lives in ONE place (its own template)
    instead of being duplicated in the joint skeleton.

    Options are matched to the skeleton's ``components`` by likelihood class name and
    emitted in that order -- the positional list ``MultiGaussianLikelihood`` zips
    against ``components``. Theory blocks are unioned across members via
    ``recursive_update``, then the skeleton's own theory is applied last as the
    joint-level override. The merge is **last-wins, so member order is significant**:
    a coherent joint analysis is expected to carry coherent member precision (we do
    not reconcile conflicting accuracy keys). Because the members are loaded through
    ``_load_template``, a folder override on a member (``templates/<member>.yaml``)
    flows in here too -- one override reaches both the standalone member preset and
    this joint preset, keeping e.g. an imprint and its fit consistent by construction.
    """
    member_infos = [
        _load_template(PRESETS[m]["template"], defaults_dir=defaults_dir)
        for m in members
    ]
    options_by_class = {}
    for minfo in member_infos:
        for cls, opts in minfo.get("likelihood", {}).items():
            options_by_class[cls] = opts or {}

    _, mgl = next(iter(info["likelihood"].items()))
    components = mgl["components"]
    missing = [c for c in components if c not in options_by_class]
    if missing:
        raise ValueError(
            f"preset members {members} provide no options for component(s) {missing}; "
            f"members expose: {sorted(options_by_class)}"
        )
    mgl["options"] = [options_by_class[c] for c in components]

    composed = {}
    for minfo in member_infos:
        composed = recursive_update(composed, minfo.get("theory", {}))
    info["theory"] = recursive_update(composed, info.get("theory", {}))


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

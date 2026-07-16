"""Load the Fiducial map and turn it into a Cobaya ``params`` dict.

A *dual parameter* carries a sampling spec (``prior``/``ref``/``proposal``/
``latex``). It is emitted in its full sampling form only when its name appears in
the explicit ``sample`` list; otherwise it collapses to ``{value: <central>}``,
where the central value is the spec's ``ref`` (or prior centre). Everything else
on the parameter (``latex``, ``drop``, ``renames``) is carried over.
"""

from importlib import resources
from pathlib import Path

import yaml

# Sampling-only keys, dropped when a dual parameter is fixed to its central value.
_SAMPLING_KEYS = ("prior", "ref", "proposal")


def _central_value(pspec):
    """The fixed value a dual parameter takes when it is not being sampled."""
    if "value" in pspec:
        return pspec["value"]
    ref = pspec.get("ref")
    if isinstance(ref, dict):
        return ref["loc"]
    if ref is not None:
        return ref
    prior = pspec["prior"]
    if "loc" in prior:
        return prior["loc"]
    if "min" in prior and "max" in prior:
        return 0.5 * (prior["min"] + prior["max"])
    raise ValueError(
        f"cannot derive a fixed value from prior {prior!r}; add an explicit 'ref'"
    )


def build_params(spec, sample=None, groups=None):
    """Flatten the grouped Fiducial map ``spec`` into a Cobaya ``params`` dict.

    ``groups`` optionally restricts the output to the named top-level groups
    (e.g. ``["cosmo"]``); by default every group is included.
    """
    sample = set(sample or [])
    if groups is None:
        selected = spec
    else:
        missing = [g for g in groups if g not in spec]
        if missing:
            raise ValueError(
                f"missing parameter group(s) {missing}: no defaults/<group>.yaml "
                f"provides them; available groups: {sorted(spec)}"
            )
        selected = {g: spec[g] for g in groups}
    # Only dual parameters (those carrying a prior) can be sampled, so an unknown
    # name -- a typo, a param from a group this preset does not load, or one fixed
    # by value -- would otherwise be dropped in silence and yield a fully-fixed run.
    sampleable = {n for g in selected.values() for n, p in g.items() if "prior" in p}
    unknown = sample - sampleable
    if unknown:
        raise ValueError(
            f"cannot sample {sorted(unknown)}: not a dual parameter in group(s) "
            f"{sorted(selected)}; sampleable here: {sorted(sampleable)}"
        )
    params = {}
    for group in selected.values():
        for name, pspec in group.items():
            if "prior" in pspec and name not in sample:
                fixed = {k: v for k, v in pspec.items() if k not in _SAMPLING_KEYS}
                fixed["value"] = _central_value(pspec)
                params[name] = fixed
            else:
                params[name] = pspec
    return params


def _theory_renames():
    """Map each native parameter name to its common aliases, per Cobaya's tables.

    Cobaya's camb/classy theories share a common alias namespace (e.g. the alias
    ``omegabh2`` maps to ``ombh2`` in CAMB and ``omega_b`` in CLASS). Sourcing
    aliases from here keeps the Fiducial map theory-portable without us tracking
    the mapping ourselves.
    """
    from cobaya.theories.camb.camb import CAMB
    from cobaya.theories.classy.classy import classy

    by_native = {}
    for cls in (CAMB, classy):
        for alias, native in cls.get_defaults().get("renames", {}).items():
            by_native.setdefault(native, set()).add(alias)
    return by_native


def _attach_renames(params):
    """Add Cobaya's common aliases to each parameter's ``renames`` (in place)."""
    by_native = _theory_renames()
    for name, pspec in params.items():
        aliases = by_native.get(name)
        if aliases:
            merged = set(pspec.get("renames", [])) | aliases
            params[name] = {**pspec, "renames": sorted(merged)}
    return params


def _bundled_defaults_dir():
    """The packaged ``presets/defaults/`` directory (a Traversable)."""
    return resources.files(__package__).joinpath("defaults")


# Reserved stem: ``theory.yaml`` is the neutrino/theory overlay, not a param
# group, so it is excluded from group discovery and loaded via ``load_theory``.
_THEORY_GROUP = "theory"


def _group_names():
    """Canonical group set: the stems of the bundled ``defaults/*.yaml`` files,
    excluding the reserved ``theory`` overlay."""
    return sorted(
        entry.name[: -len(".yaml")]
        for entry in _bundled_defaults_dir().iterdir()
        if entry.name.endswith(".yaml") and entry.name[: -len(".yaml")] != _THEORY_GROUP
    )


def _as_mapping(text, source):
    """Parse a group YAML file, requiring a mapping of param specs."""
    doc = yaml.safe_load(text)
    if not isinstance(doc, dict):
        raise ValueError(
            f"{source}: expected a mapping of param specs, got {type(doc).__name__}"
        )
    return doc


def _read_group(group, defaults_dir):
    """Read one group's param dict, preferring an override file in ``defaults_dir``.

    Per-file fallback: ``defaults_dir/<group>.yaml`` replaces the bundled file for
    that group when present; otherwise the bundled file is used.
    """
    if defaults_dir is not None:
        override = Path(defaults_dir) / f"{group}.yaml"
        if override.is_file():
            return _as_mapping(override.read_text(), override)
    bundled = _bundled_defaults_dir().joinpath(f"{group}.yaml")
    return _as_mapping(bundled.read_text(), bundled)


def load_fiducial_map(defaults_dir=None):
    """Parse the Fiducial map (packaged defaults, optionally with per-group
    overrides) into its grouped form.

    Each top-level group is one ``defaults/<group>.yaml`` file. ``defaults_dir``
    optionally supplies override files; any ``<group>.yaml`` it contains replaces
    the bundled file for that group (per-file fallback, mflike-style).
    """
    return {group: _read_group(group, defaults_dir) for group in _group_names()}


def load_theory(defaults_dir=None):
    """Read the theory overlay (``theory.yaml``): the neutrino-sector ``extra_args``
    keyed by Boltzmann code, plus any code-specific neutrino params.

    Per-file fallback like the param groups: ``defaults_dir/theory.yaml`` replaces
    the bundled file wholesale when present; otherwise the bundled file is used.
    """
    return _read_group(_THEORY_GROUP, defaults_dir)


def load_params(sample=None, groups=None, defaults_dir=None):
    """Return the Cobaya ``params`` dict for the Fiducial map (packaged defaults,
    optionally with per-group overrides).

    ``sample`` is the explicit list of dual parameters to vary; every other dual
    parameter is fixed to its fiducial central value. ``groups`` optionally
    restricts the output to the named groups. ``defaults_dir`` optionally points at
    a directory of override ``<group>.yaml`` files (per-file fallback).
    """
    spec = load_fiducial_map(defaults_dir=defaults_dir)
    params = build_params(spec, sample=sample, groups=groups)
    return _attach_renames(params)

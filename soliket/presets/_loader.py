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
    selected = spec if groups is None else {g: spec[g] for g in groups}
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


def _bundled_params_dir():
    """The packaged ``presets/params/`` directory (a Traversable)."""
    return resources.files(__package__).joinpath("params")


def _group_names():
    """Canonical group set: the stems of the bundled ``params/*.yaml`` files."""
    return sorted(
        entry.name[: -len(".yaml")]
        for entry in _bundled_params_dir().iterdir()
        if entry.name.endswith(".yaml")
    )


def _as_mapping(text, source):
    """Parse a group YAML file, requiring a mapping of param specs."""
    doc = yaml.safe_load(text)
    if not isinstance(doc, dict):
        raise ValueError(
            f"{source}: expected a mapping of param specs, got {type(doc).__name__}"
        )
    return doc


def _read_group(group, params_dir):
    """Read one group's param dict, preferring an override file in ``params_dir``.

    Per-file fallback: ``params_dir/<group>.yaml`` replaces the bundled file for
    that group when present; otherwise the bundled file is used.
    """
    if params_dir is not None:
        override = Path(params_dir) / f"{group}.yaml"
        if override.is_file():
            return _as_mapping(override.read_text(), override)
    bundled = _bundled_params_dir().joinpath(f"{group}.yaml")
    return _as_mapping(bundled.read_text(), bundled)


def load_fiducial_map(params_dir=None):
    """Parse the packaged Fiducial map into its grouped form.

    Each top-level group is one ``params/<group>.yaml`` file. ``params_dir``
    optionally supplies override files; any ``<group>.yaml`` it contains replaces
    the bundled file for that group (per-file fallback, mflike-style).
    """
    return {group: _read_group(group, params_dir) for group in _group_names()}


def load_params(sample=None, groups=None, params_dir=None):
    """Return the Cobaya ``params`` dict for the Fiducial map.

    ``sample`` is the explicit list of dual parameters to vary; every other dual
    parameter is fixed to its fiducial central value. ``groups`` optionally
    restricts the output to the named groups. ``params_dir`` optionally points at
    a directory of override ``<group>.yaml`` files (per-file fallback).
    """
    spec = load_fiducial_map(params_dir=params_dir)
    params = build_params(spec, sample=sample, groups=groups)
    return _attach_renames(params)

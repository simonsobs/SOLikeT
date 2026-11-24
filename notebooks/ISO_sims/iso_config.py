"""Read the ISO_sims configuration from the ``defaults/`` folder.

This is the only code that reads ``defaults/``; every ISO_sims notebook goes
through it, so they all share the same fiducial, priors and modelling choices.
See ``defaults/README.md`` for the layout.

A run's Cobaya ``info`` is built exactly like ``cobaya-grid`` builds a job::

    info = merge_info(base_info(), model_info(model), dataset_info(dataset))

so ``build_info(dataset, model)`` in a notebook equals the input of that grid job.
"""

from pathlib import Path

import numpy as np
import yaml
from cobaya.input import merge_info
from cobaya.tools import resolve_packages_path

HERE = Path(__file__).resolve().parent
DEFAULTS = HERE / "defaults"

MULTIGAUSSIAN = "soliket.gaussian.MultiGaussianLikelihood"
THEORY_CODES = ["camb", "classy"]

# Neutrino params of cosmo.yaml that only camb understands; classy gets its fixed
# normal-hierarchy masses from theory/classy.yaml instead.
CAMB_ONLY_PARAMS = ["mnu", "nnu"]
CAMB_ONLY_MODELS = ["mnu", "neff"]

# Keys of a dual parameter that only make sense when it is sampled.
SAMPLING_KEYS = ["prior", "ref", "proposal"]


# ---------------------------------------------------------------- reading files


def load_yaml(relpath):
    """Load ``defaults/<relpath>`` as is (no placeholder replacement)."""
    with open(DEFAULTS / relpath) as f:
        return yaml.safe_load(f) or {}


def load_paths():
    """The ``paths.yaml`` mapping, with values still relative to ISO_sims/."""
    return load_yaml("paths.yaml")


def path(key):
    """Absolute path of the ``paths.yaml`` entry ``key``."""
    paths = load_paths()
    if key not in paths:
        raise KeyError(f"no '{key}' in defaults/paths.yaml; known: {sorted(paths)}")
    return (HERE / paths[key]).resolve()


def grid_spec():
    """The ``grid_spec.yaml`` mapping (theory code, models, datasets, groups)."""
    return load_yaml("grid_spec.yaml")


def _placeholder_values():
    """Values for the ``{name}`` placeholders allowed in the YAML files."""
    values = {key: str(path(key)) for key in load_paths()}
    packages = resolve_packages_path()
    if packages:
        values["packages"] = str(packages)
    return values


def _fill_placeholders(item, values):
    """Replace ``{name}`` by ``values[name]`` in every string of ``item``."""
    if isinstance(item, dict):
        return {key: _fill_placeholders(value, values) for key, value in item.items()}
    if isinstance(item, list):
        return [_fill_placeholders(value, values) for value in item]
    if isinstance(item, str):
        if "{packages}" in item and "packages" not in values:
            raise ValueError(
                "a defaults file uses {packages} but no cobaya packages path is set "
                "(set COBAYA_PACKAGES_PATH or run cobaya-install)"
            )
        for name, value in values.items():
            item = item.replace("{" + name + "}", value)
    return item


def load_fragment(relpath):
    """Load an ``info`` fragment from ``defaults/``, with placeholders replaced."""
    return _fill_placeholders(load_yaml(relpath), _placeholder_values())


# --------------------------------------------------------- building info dicts


def _check_theory(theory):
    if theory is None:
        theory = grid_spec()["theory"]
    if theory not in THEORY_CODES:
        raise ValueError(f"unknown theory '{theory}'; choose from {THEORY_CODES}")
    return theory


def base_info(theory=None):
    """What every run shares: base cosmology, theory code setup and sampler."""
    theory = _check_theory(theory)
    info = merge_info(
        load_fragment("cosmo.yaml"),
        load_fragment(f"theory/{theory}.yaml"),
        load_fragment("sampler.yaml"),
    )
    if theory == "classy":
        for name in CAMB_ONLY_PARAMS:
            del info["params"][name]
    return info


def model_info(model, theory=None):
    """The params a model adds on top of the base (nothing for ``lcdm``)."""
    theory = _check_theory(theory)
    models = grid_spec()["models"]
    if model not in models:
        raise ValueError(f"unknown model '{model}'; choose from {sorted(models)}")
    pieces = [{"params": {}}]
    for name in models[model]:
        if theory == "classy" and name in CAMB_ONLY_MODELS:
            raise ValueError(f"model '{model}' needs camb (see theory/classy.yaml)")
        pieces.append(load_fragment(f"models/{name}.yaml"))
    return merge_info(*pieces)


def probe_info(probe, theory=None, data="sims"):
    """Everything one probe needs: likelihood, its nuisance params and theory.

    ``data="sims"`` points the likelihood at the smooth twin in sims/ (merging
    ``sims.yaml``); ``data="shipped"`` keeps the data shipped with the packages.
    """
    theory = _check_theory(theory)
    if data not in ["sims", "shipped"]:
        raise ValueError(f"unknown data '{data}'; choose 'sims' or 'shipped'")
    folder = DEFAULTS / "probes" / probe
    if not folder.is_dir():
        raise ValueError(f"no folder defaults/probes/{probe}/")

    # Files merged in every case, then the theory-code file, then the data file.
    special_files = [f"{code}.yaml" for code in THEORY_CODES] + ["sims.yaml"]
    files = [f.name for f in sorted(folder.glob("*.yaml")) if f.name not in special_files]
    files.append(f"{theory}.yaml")
    if data == "sims":
        files.append("sims.yaml")

    pieces = [{"params": {}}]
    for name in files:
        if (folder / name).is_file():
            pieces.append(load_fragment(f"probes/{probe}/{name}"))
    return merge_info(*pieces)


def dataset_info(dataset, theory=None, data="sims"):
    """The ``info`` fragment of a dataset of ``grid_spec.yaml``.

    One probe gives its likelihood directly; several probes are combined in one
    MultiGaussianLikelihood, with their cross-covariance if ``cross_cov`` is set.
    """
    datasets = grid_spec()["datasets"]
    if dataset not in datasets:
        raise ValueError(f"unknown dataset '{dataset}'; choose from {sorted(datasets)}")
    spec = datasets[dataset]
    probes = [probe_info(probe, theory, data) for probe in spec["probes"]]
    if len(probes) == 1:
        return probes[0]

    components = []
    options = []
    rest = []  # each probe's theory and params, without its likelihood
    for info in probes:
        likelihood = info.pop("likelihood")
        if len(likelihood) != 1:
            raise ValueError("a probe must define exactly one likelihood")
        name = list(likelihood)[0]
        components.append(name)
        options.append(likelihood[name] or {})
        rest.append(info)

    combined = merge_info(*rest)
    multigaussian = {"components": components, "options": options, "stop_at_error": True}
    if "cross_cov" in spec:
        multigaussian["cross_cov_path"] = str(path(spec["cross_cov"]))
    combined["likelihood"] = {MULTIGAUSSIAN: multigaussian}
    return combined


def build_info(dataset, model="lcdm", theory=None, data="sims"):
    """Full Cobaya ``info`` for ``dataset`` x ``model``, as a cobaya-grid job builds it.

    Every parameter with a prior is sampled; use ``fix_to_fiducial`` on
    ``info["params"]`` to fix them.
    """
    theory = _check_theory(theory)
    return merge_info(
        base_info(theory),
        model_info(model, theory),
        dataset_info(dataset, theory, data),
    )


# ------------------------------------------------------------------ parameters


def fiducial_value(spec):
    """The value a dual parameter takes when fixed: its ``ref`` centre, else prior centre."""
    if "value" in spec:
        return spec["value"]
    ref = spec.get("ref")
    if isinstance(ref, dict):
        if "loc" in ref:
            return ref["loc"]
        if "min" in ref and "max" in ref:
            return 0.5 * (ref["min"] + ref["max"])
    elif ref is not None:
        return ref
    prior = spec["prior"]
    if "loc" in prior:
        return prior["loc"]
    return 0.5 * (prior["min"] + prior["max"])


def fix_to_fiducial(params, sample=()):
    """Copy of ``params`` where every prior-carrying param not in ``sample`` is fixed.

    Fixed params keep their other keys (``latex``, ``drop``, ...).
    """
    samplable = [name for name, spec in params.items() if "prior" in (spec or {})]
    unknown = [name for name in sample if name not in samplable]
    if unknown:
        raise ValueError(f"cannot sample {unknown}: no prior in the defaults")

    fixed = {}
    for name, spec in params.items():
        if name in samplable and name not in sample:
            new_spec = {key: value for key, value in spec.items() if key not in SAMPLING_KEYS}
            new_spec["value"] = fiducial_value(spec)
            fixed[name] = new_spec
        else:
            fixed[name] = spec
    return fixed


def fiducial_values(params):
    """Numeric fiducial value of every fixed param (lambdas and derived skipped)."""
    values = {}
    for name, spec in fix_to_fiducial(params).items():
        if isinstance(spec, dict) and "value" in spec:
            value = spec["value"]
        elif isinstance(spec, (int, float)):
            value = spec
        else:
            continue
        if isinstance(value, (int, float)):
            values[name] = value
    return values


# ------------------------------------------------------------------------- camb


def cosmology_param_names():
    """Names of the cosmological parameters: those of cosmo.yaml and of models/."""
    names = set(load_yaml("cosmo.yaml")["params"])
    for model_file in sorted((DEFAULTS / "models").glob("*.yaml")):
        names.update(load_yaml(f"models/{model_file.name}").get("params", {}))
    return names


def camb_arguments(model, info):
    """``camb.set_params`` arguments reproducing the camb run of an evaluated model.

    ``model`` is the cobaya model built from ``info`` (every parameter fixed).
    Returns ``(cosmo, accuracy)``:

    - ``cosmo``: the cosmological parameters at the values the model uses, with
      derived inputs computed (e.g. As from logA) and dropped ones (logA) left out;
    - ``accuracy``: the camb ``extra_args`` of ``info`` (neutrino setup, precision).

    Use them where camb is called outside cobaya (e.g. the cross-covariance
    derivative), so that it runs with exactly the same settings as the analysis.
    """
    inputs = model.parameterization.to_input({})
    cosmo = {}
    for name in cosmology_param_names():
        spec = info["params"].get(name)
        if name in inputs and not (isinstance(spec, dict) and spec.get("drop")):
            cosmo[name] = float(inputs[name])
    accuracy = dict(info["theory"]["camb"].get("extra_args", {}))
    return cosmo, accuracy


# --------------------------------------------------------------------- CCL sims


def ccl_cosmology(**ccl_overrides):
    """``pyccl.Cosmology`` at the base fiducial of cosmo.yaml.

    ``ccl_overrides`` replace CCL arguments, e.g. ``A_s=1.8e-9``.
    """
    import pyccl as ccl

    p = fiducial_values(load_fragment("cosmo.yaml")["params"])
    h = p["H0"] / 100
    settings = load_yaml("sims/galaxy_kappa.yaml")
    arguments = {
        "Omega_c": p["omch2"] / h**2,
        "Omega_b": p["ombh2"] / h**2,
        "h": h,
        "n_s": p["ns"],
        "A_s": 1e-10 * np.exp(p["logA"]),
        "Neff": p["nnu"],
        "m_nu": p["mnu"],
        "mass_split": settings["neutrino_mass_split"],
        "matter_power_spectrum": settings["matter_power_spectrum"],
    }
    arguments.update(ccl_overrides)
    return ccl.Cosmology(**arguments)


# ------------------------------------------------------------------------- grid


def grid_info():
    """The cobaya-grid input built from ``grid_spec.yaml``."""
    spec = grid_spec()
    theory = _check_theory(spec["theory"])
    groups = {}
    for name, group in spec["groups"].items():
        groups[name] = {
            "models": group["models"],
            "datasets": group["datasets"],
            "base": theory,
        }
    return {
        "defaults": base_info(theory),
        "groups": groups,
        "models": {model: model_info(model, theory) for model in spec["models"]},
        "datasets": {dataset: dataset_info(dataset, theory) for dataset in spec["datasets"]},
        "cov_dir": None,  # set once first runs are available
    }

"""Read the ISO configuration and locate the ISO data files.

The configuration (YAML files only) lives in a separate private repository, whose
checkout is given by the ``ISO_CONFIG`` environment variable; the data files live
under ``ISO_DATA``. Nothing is read from, or written to, this SOLikeT checkout.
See the README of the configuration repository for its layout, and README.md in
this folder for the setup.

Every ISO notebook goes through this module, so they all share the same fiducial,
priors and modelling choices. A run's Cobaya ``info`` is built exactly like
``cobaya-grid`` builds a job::

    info = merge_info(base_info(), model_info(model), dataset_info(dataset, level))

so ``build_info(dataset, model, level=...)`` in a notebook equals the input of
that grid job.

The data are used at three levels, which are sections of ``paths.yaml`` and only
differ by the files loaded: ``mock`` (realistic simulations of the ISO data),
``twin`` (smooth twins of the mocks) and ``real`` (the ISO data).
"""

import os
from pathlib import Path

import yaml
from cobaya.input import merge_info
from cobaya.tools import resolve_packages_path

MULTIGAUSSIAN = "soliket.gaussian.MultiGaussianLikelihood"
THEORY_CODES = ["camb", "classy"]
LEVELS = ["mock", "twin", "real"]

# Neutrino params of cosmo.yaml that only camb understands; classy gets its fixed
# normal-hierarchy masses from theory/classy.yaml instead.
CAMB_ONLY_PARAMS = ["mnu", "nnu"]
CAMB_ONLY_MODELS = ["mnu", "neff"]

# Keys of a dual parameter that only make sense when it is sampled.
SAMPLING_KEYS = ["prior", "ref", "proposal"]


# ------------------------------------------------------------------ locations


def _folder_from_env(variable, what):
    """The folder given by the environment variable ``variable``."""
    if variable not in os.environ:
        raise RuntimeError(
            f"${variable} is not set: export {variable}=/path/to/{what} "
            "(see notebooks/ISO_sims/README.md)"
        )
    folder = Path(os.environ[variable]).expanduser().resolve()
    if not folder.is_dir():
        raise FileNotFoundError(f"${variable} points to a missing folder: {folder}")
    return folder


def config_dir():
    """The checkout of the private ISO configuration repository (``$ISO_CONFIG``)."""
    return _folder_from_env("ISO_CONFIG", "iso-config")


def data_dir():
    """The root of the ISO data files (``$ISO_DATA``)."""
    return _folder_from_env("ISO_DATA", "iso_data")


def runs_dir():
    """Where runs write their products: chains, grid yaml, grid folder."""
    return data_dir() / "runs"


def _packages_path():
    packages = resolve_packages_path()
    if not packages:
        raise ValueError(
            "a configuration file uses {packages} but no cobaya packages path is set "
            "(set COBAYA_PACKAGES_PATH or run cobaya-install)"
        )
    return str(packages)


# ---------------------------------------------------------------- reading files


def load_yaml(relpath):
    """Load ``<config>/<relpath>`` as is (no placeholder replacement)."""
    with open(config_dir() / relpath) as f:
        return yaml.safe_load(f) or {}


def yaml_names(relfolder):
    """Sorted names of the ``.yaml`` files of ``<config>/<relfolder>``."""
    folder = config_dir() / relfolder
    if not folder.is_dir():
        raise ValueError(f"no folder {relfolder}/ in {config_dir()}")
    return sorted(f.name for f in folder.glob("*.yaml"))


def grid_spec():
    """The ``grid_spec.yaml`` mapping (theory code, level, models, datasets, groups)."""
    return load_yaml("grid_spec.yaml")


def paths(level=None):
    """Absolute path of every file of ``level``, from its section of ``paths.yaml``.

    Paths in ``paths.yaml`` are relative to ``$ISO_DATA``; absolute paths and the
    ``{packages}`` placeholder are also allowed.
    """
    level = _check_level(level)
    sections = load_yaml("paths.yaml")
    if level not in sections:
        raise KeyError(f"no '{level}' section in paths.yaml; found: {sorted(sections)}")
    files = {}
    for key, value in (sections[level] or {}).items():
        if "{packages}" in value:
            value = value.replace("{packages}", _packages_path())
        files[key] = (data_dir() / value).resolve()
    return files


def path(key, level=None):
    """Absolute path of the file ``key`` of ``level`` (a key of ``paths.yaml``)."""
    level = _check_level(level)
    files = paths(level)
    if key not in files:
        raise KeyError(f"no '{key}' in the '{level}' section of paths.yaml")
    return files[key]


def all_path_keys():
    """The keys used in any section of ``paths.yaml``."""
    keys = set()
    for section in load_yaml("paths.yaml").values():
        keys.update(section or {})
    return keys


def make_data_dirs():
    """Create the folders of every file listed in ``paths.yaml``, and ``runs/``.

    Returns the folders, so you can see where to put the mocks.
    """
    folders = {runs_dir()}
    for level in load_yaml("paths.yaml"):
        for file in paths(level).values():
            if data_dir() in file.parents:  # never create folders outside $ISO_DATA
                folders.add(file.parent)
    for folder in sorted(folders):
        folder.mkdir(parents=True, exist_ok=True)
    return sorted(folders)


def _placeholder_values(level):
    """Values of the ``{name}`` placeholders allowed in the YAML files."""
    values = {"data": str(data_dir())}
    if level is not None:
        values.update({key: str(file) for key, file in paths(level).items()})
    return values


def _fill_placeholders(item, values):
    """Replace ``{name}`` by ``values[name]`` in every string of ``item``."""
    if isinstance(item, dict):
        return {key: _fill_placeholders(value, values) for key, value in item.items()}
    if isinstance(item, list):
        return [_fill_placeholders(value, values) for value in item]
    if isinstance(item, str):
        if "{packages}" in item:
            item = item.replace("{packages}", _packages_path())
        for name, value in values.items():
            item = item.replace("{" + name + "}", value)
    return item


def _unfilled_keys(item, keys):
    """The keys among ``keys`` still written ``{key}`` somewhere in ``item``."""
    if isinstance(item, dict):
        return set().union(*[_unfilled_keys(value, keys) for value in item.values()])
    if isinstance(item, list):
        return set().union(*[_unfilled_keys(value, keys) for value in item])
    if isinstance(item, str):
        return {key for key in keys if "{" + key + "}" in item}
    return set()


def load_fragment(relpath, level=None):
    """Load an ``info`` fragment of the configuration, with placeholders replaced.

    A fragment that refers to a file must be loaded with a ``level``; a file key
    the level does not define is an error.
    """
    fragment = _fill_placeholders(load_yaml(relpath), _placeholder_values(level))
    missing = _unfilled_keys(fragment, all_path_keys())
    if missing:
        where = f"the '{level}' section of paths.yaml" if level else "no level was given"
        raise KeyError(f"{relpath} uses {sorted(missing)}, but {where} has none")
    return fragment


# --------------------------------------------------------- building info dicts


def _check_theory(theory):
    if theory is None:
        theory = grid_spec()["theory"]
    if theory not in THEORY_CODES:
        raise ValueError(f"unknown theory '{theory}'; choose from {THEORY_CODES}")
    return theory


def _check_level(level):
    if level is None:
        level = grid_spec()["level"]
    if level not in LEVELS:
        raise ValueError(f"unknown level '{level}'; choose from {LEVELS}")
    return level


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


def probe_info(probe, theory=None, level=None):
    """Everything one probe needs: likelihood, its nuisance params and theory,
    reading the data files of ``level``."""
    theory = _check_theory(theory)
    level = _check_level(level)
    names = yaml_names(f"probes/{probe}")

    # Files merged in every case, then the file of the theory code, if any.
    theory_files = [f"{code}.yaml" for code in THEORY_CODES]
    files = [name for name in names if name not in theory_files]
    if f"{theory}.yaml" in names:
        files.append(f"{theory}.yaml")

    pieces = [{"params": {}}]
    for name in files:
        pieces.append(load_fragment(f"probes/{probe}/{name}", level))
    return merge_info(*pieces)


def dataset_info(dataset, theory=None, level=None):
    """The ``info`` fragment of a dataset of ``grid_spec.yaml``, at ``level``.

    One probe gives its likelihood directly; several probes are combined in one
    MultiGaussianLikelihood, with their cross-covariance if ``cross_cov`` is set.
    """
    level = _check_level(level)
    datasets = grid_spec()["datasets"]
    if dataset not in datasets:
        raise ValueError(f"unknown dataset '{dataset}'; choose from {sorted(datasets)}")
    spec = datasets[dataset]
    probes = [probe_info(probe, theory, level) for probe in spec["probes"]]
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
        multigaussian["cross_cov_path"] = str(path(spec["cross_cov"], level))
    combined["likelihood"] = {MULTIGAUSSIAN: multigaussian}
    return combined


def build_info(dataset, model="lcdm", theory=None, level=None):
    """Full Cobaya ``info`` for ``dataset`` x ``model`` at ``level``, as a cobaya-grid
    job builds it.

    Every parameter with a prior is sampled; use ``fix_to_fiducial`` on
    ``info["params"]`` to fix them.
    """
    theory = _check_theory(theory)
    return merge_info(
        base_info(theory),
        model_info(model, theory),
        dataset_info(dataset, theory, level),
    )


# ------------------------------------------------------------------ parameters


def fiducial_value(spec):
    """Value of a dual parameter when fixed: its ``ref`` centre, else its prior centre."""
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
        raise ValueError(f"cannot sample {unknown}: no prior in the configuration")

    fixed = {}
    for name, spec in params.items():
        if name in samplable and name not in sample:
            new_spec = {
                key: value for key, value in spec.items() if key not in SAMPLING_KEYS
            }
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
    for name in yaml_names("models"):
        names.update(load_yaml(f"models/{name}").get("params", {}))
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


# ------------------------------------------------------------------------- grid


def grid_info(level=None):
    """The cobaya-grid input built from ``grid_spec.yaml``, at ``level``."""
    spec = grid_spec()
    theory = _check_theory(spec["theory"])
    level = _check_level(level)
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
        "datasets": {
            dataset: dataset_info(dataset, theory, level) for dataset in spec["datasets"]
        },
        "cov_dir": None,  # set once first runs are available
    }

# ISO_sims

End-to-end validation of the ISO analysis on simulations: build the smooth twins of
the ISO mocks, the cross-covariance between the probes, and run the joint fits.

| Notebook | What it does |
| --- | --- |
| [`create_datasets.ipynb`](create_datasets.ipynb) | builds the smooth twins of the mocks at the fiducial |
| [`create_cross_covariance.ipynb`](create_cross_covariance.ipynb) | builds the MFLike x lensing cross-covariance from the mocks |
| [`analyse_datasets.ipynb`](analyse_datasets.ipynb) | evaluates and samples the joint likelihood, on the twins or the mocks |
| [`generate_grid.ipynb`](generate_grid.ipynb) | writes and creates the cobaya grid of every model x dataset |

These notebooks hold no configuration and no data. Everything comes from two
folders, given by environment variables:

- `ISO_CONFIG`: the configuration (YAML files: fiducial values, priors, modelling
  choices, data file names). Either
  - [`example_config/`](example_config), public, which runs on the data shipped
    with the likelihoods and lets anyone run the notebooks, or
  - a checkout of the private ISO configuration repository
    [`simonsobs/soliket-config-files`](https://github.com/simonsobs/soliket-config-files),
    for the ISO mocks and, later, the ISO data. It has the same layout as
    `example_config/`.
- `ISO_DATA`: the root of the data files, laid out (for the ISO configuration) as

  ```
  $ISO_DATA/
  ├── mocks/v1/             the ISO mocks, downloaded (treat as read-only)
  │   ├── mflike/           bandpowers, cov_Bbl, ...
  │   ├── lensing/          bandpowers, N0/N1 corrections, fiducial, ...
  │   ├── galaxy_kappa/
  │   └── README.md         where and when each file came from (+ checksums)
  ├── twins/v1/             made from mocks/v1 by the notebooks
  │   ├── mflike/           same layout as the mocks
  │   ├── lensing/
  │   ├── galaxy_kappa/
  │   └── cross/            products spanning several probes (cross-covariance)
  ├── real/                 the ISO data, later (same per-probe layout)
  └── runs/                 chains, grid yaml and grid folder, generated yaml
  ```

  One folder per probe keeps each probe's files together (a data vector next to
  its covariance and windows), lets one probe's files be replaced without touching
  the others, and puts each twin at the same relative path as its mock. Files can
  keep the names they were delivered with. The code does not depend on this
  layout: every file is reached through `paths.yaml`.

## Setup

With the public example configuration:

```bash
export ISO_CONFIG=/path/to/SOLikeT/notebooks/ISO_sims/example_config
export ISO_DATA=/path/to/scratch_folder   # where the twins and runs are written
```

Its "mocks" are the data shipped with the likelihoods, which `create_datasets.ipynb`
installs, and a galaxy x kappa test file of SOLikeT; the twins go to
`$ISO_DATA/twins/example/`.

With the ISO configuration (access to the private repository needed):

```bash
# clone the configuration outside the SOLikeT checkout
git clone git@github.com:simonsobs/soliket-config-files.git ~/soliket-config-files
export ISO_CONFIG=~/soliket-config-files
export ISO_DATA=/path/to/iso_data
```

Then put the mocks where the `mock` section of its `paths.yaml` expects them.

In both cases, the data folders listed in `paths.yaml` (and `runs/`) are created by
`create_datasets.ipynb`, or by hand with

```bash
cd notebooks/ISO_sims
python -c "import iso_config; iso_config.make_data_dirs()"
```

and the notebooks run in the order of the table above.

`paths.yaml` gives each file relative to `$ISO_DATA`; it may also use
`{packages}` (the cobaya packages path) and `{soliket}` (this SOLikeT checkout).

## Data levels

The same configuration is used at three levels, sections of `paths.yaml` that
only differ by the data files loaded: `mock`, `twin` and `real`. A run picks one
with `iso_config.build_info(dataset, model, level=...)`; the default is `level` in
the configuration's `grid_spec.yaml`.

## Keeping private things private

This repository is public. When working with the ISO configuration:

- clear the notebook outputs before committing (they print file names and values):
  `jupyter nbconvert --clear-output --inplace notebooks/ISO_sims/*.ipynb`, or
  install [`nbstripout`](https://github.com/kynan/nbstripout) once in your checkout
  with `nbstripout --install`;
- nothing is written into this checkout: twins, cross-covariance, chains, grid and
  generated yaml all go under `$ISO_DATA`;
- never copy files of the private configuration into `example_config/`: that
  folder is public.

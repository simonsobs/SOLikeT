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
places outside this repository, given by environment variables:

- `ISO_CONFIG`: a checkout of the private ISO configuration repository (YAML files:
  fiducial values, priors, modelling choices, data file names). Its README
  describes its layout.
- `ISO_DATA`: the root of the data files, laid out as

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

```bash
export ISO_CONFIG=/path/to/iso-config
export ISO_DATA=/path/to/iso_data

# create the data folders listed in the configuration's paths.yaml, and runs/
cd notebooks/ISO_sims
python -c "import iso_config; iso_config.make_data_dirs()"
```

Then put the mocks where the `mock` section of `paths.yaml` expects them, and run
the notebooks in the order of the table above.

## Data levels

The same configuration is used at three levels, sections of `paths.yaml` that
only differ by the data files loaded: `mock`, `twin` and `real`. A run picks one
with `iso_config.build_info(dataset, model, level=...)`; the default is `level` in
the configuration's `grid_spec.yaml`.

## Keeping private things private

This repository is public, so:

- clear the notebook outputs before committing (they print file names and values):
  `jupyter nbconvert --clear-output --inplace notebooks/ISO_sims/*.ipynb`, or
  install [`nbstripout`](https://github.com/kynan/nbstripout) once in your checkout
  with `nbstripout --install`;
- nothing is written into this checkout: twins, cross-covariance, chains, grid and
  generated yaml all go under `$ISO_DATA`.

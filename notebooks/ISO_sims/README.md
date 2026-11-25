# ISO_sims

End-to-end validation of an analysis on simulations: build the smooth twins of the
mocks, the cross-covariance between the probes, and run the joint fits.

| Notebook | What it does |
| --- | --- |
| [`create_datasets.ipynb`](create_datasets.ipynb) | builds the smooth twins of the mocks at the fiducial |
| [`create_cross_covariance.ipynb`](create_cross_covariance.ipynb) | builds the MFLike x lensing cross-covariance from the mocks |
| [`analyse_datasets.ipynb`](analyse_datasets.ipynb) | evaluates and samples the joint likelihood, on the twins or the mocks |
| [`generate_grid.ipynb`](generate_grid.ipynb) | writes and creates the cobaya grid of every model x dataset |

The notebooks hold no configuration and no data. They read two folders, given by
environment variables:

- `ISO_CONFIG`: the configuration (YAML files), e.g. [`example_config/`](example_config),
  which runs on the data shipped with the likelihoods. Any folder with the same
  layout can be used instead;
- `ISO_DATA`: the data files, and everything the notebooks write.

## Setup

```bash
export ISO_CONFIG=/path/to/SOLikeT/notebooks/ISO_sims/example_config
export ISO_DATA=/path/to/data_folder
```

`create_datasets.ipynb` creates the data folders listed in the configuration's
`paths.yaml` (or run `python -c "import iso_config; iso_config.make_data_dirs()"`
in this folder); then run the notebooks in the order of the table above.

## Data levels

The configuration is used at three levels, sections of `paths.yaml` that only
differ by the data files loaded: `mock`, `twin` (smooth twins of the mocks) and
`real`. A run picks one with `iso_config.build_info(dataset, model, level=...)`;
the default is `level` in `grid_spec.yaml`.

## Data folder

Files are given in `paths.yaml` relative to `$ISO_DATA` (`{packages}` and
`{soliket}` stand for the cobaya packages path and this SOLikeT checkout). A
suggested layout, one folder per probe:

```
$ISO_DATA/
├── mocks/<version>/      the mocks (treat as read-only)
│   ├── mflike/
│   ├── lensing/
│   └── galaxy_kappa/
├── twins/<version>/      written by the notebooks, same layout as the mocks
│   └── cross/            products spanning several probes (cross-covariance)
├── real/                 the data, later
└── runs/                 chains, grid yaml and grid folder, generated yaml
```

## Before committing

Notebook outputs print file names and values: clear them before committing
(`jupyter nbconvert --clear-output --inplace notebooks/ISO_sims/*.ipynb`, or
`nbstripout --install` once in your checkout). Nothing is written into this
checkout: twins, cross-covariance, chains, grid and generated yaml all go under
`$ISO_DATA`.

# ISO_sims

End-to-end validation of an analysis on simulations: build the smooth twins of the
mocks, the cross-covariance between the probes, and run the joint fits.

| Notebook | What it does |
| --- | --- |
| [`create_datasets.ipynb`](create_datasets.ipynb) | builds the smooth twins of the mocks at the fiducial |
| [`create_cross_covariance.ipynb`](create_cross_covariance.ipynb) | builds the MFLike x lensing cross-covariance from the mocks |
| [`analyse_datasets.ipynb`](analyse_datasets.ipynb) | evaluates and samples the joint likelihood, on the twins or the mocks |
| [`generate_grid.ipynb`](generate_grid.ipynb) | writes and creates the cobaya grid of every model x dataset |

The notebooks hold no configuration and no data. They read two folders:

- the configuration (YAML files): `$ISO_CONFIG`, by default
  [`example_config/`](example_config), which runs on the data shipped with the
  likelihoods. Any folder with the same layout can be used instead;
- the data files, and everything the notebooks write: `$ISO_DATA`, by default
  `iso_data/` in this folder (git-ignored, so nothing in it is committed).

## Setup

With the defaults, there is nothing to set: run the notebooks in the order of the
table above. `create_datasets.ipynb` creates the data folders listed in the
configuration's `paths.yaml` (by hand:
`python -c "import iso_config; iso_config.make_data_dirs()"` in this folder).

To use another configuration, or a data folder elsewhere (e.g. shared on a
cluster), set the variables before starting Jupyter, or in a first cell:

```python
%env ISO_CONFIG=/path/to/configuration
%env ISO_DATA=/path/to/data_folder
```

and restart the kernel after changing them. Several configurations can share one
data folder: their files are in different folders of `paths.yaml`, and each writes
its runs to `runs/<configuration folder name>/`.

## Data levels

The configuration is used at three levels, sections of `paths.yaml` that only
differ by the data files loaded: `mock`, `twin` (smooth twins of the mocks) and
`real`. A run picks one with `iso_config.build_info(dataset, model, level=...)`;
the default is `level` in `grid_spec.yaml`.

## Data folder

Files are given in `paths.yaml` relative to the data folder (`{packages}` and
`{soliket}` stand for the cobaya packages path and this SOLikeT checkout). A
suggested layout, one folder per probe:

```
iso_data/                 (or $ISO_DATA)
├── mocks/<version>/      the mocks (treat as read-only)
│   ├── mflike/
│   ├── lensing/
│   └── galaxy_kappa/
├── twins/<version>/      written by the notebooks, same layout as the mocks
│   └── cross/            products spanning several probes (cross-covariance)
├── real/                 the data, later
└── runs/<configuration>/ chains, grid yaml and grid folder, generated yaml
```

## Before committing

Notebook outputs print file names and values: clear them before committing
(`jupyter nbconvert --clear-output --inplace notebooks/ISO_sims/*.ipynb`, or
`nbstripout --install` once in your checkout). Everything the notebooks write goes
to the data folder, which git ignores by default; never `git add -f` it.

# Example configuration

A configuration for the `ISO_sims` notebooks that runs on public data. Its mocks
are the MFLike and CMB-lensing data shipped with the likelihoods (installed by
`create_datasets.ipynb`) and the galaxy x kappa test file of `tests/data/`; the
twins are written to `twins/example/` of the data folder.

Another configuration is a folder with the same layout. It usually only changes
`paths.yaml` (its data files) and the fiducial values.

| File | Holds |
| --- | --- |
| `cosmo.yaml` | base cosmology: fiducial values and priors |
| `sampler.yaml` | sampler settings |
| `theory/<code>.yaml` | setup of the Boltzmann code (camb or classy) |
| `models/*.yaml` | parameters each model extension adds |
| `probes/<probe>/` | likelihood, nuisance parameters, `<code>.yaml` precision |
| `grid_spec.yaml` | models, datasets, grid groups, theory code, default level |
| `paths.yaml` | the data files of each level |
| `cross_cov.yaml` (optional) | `fsky` and `lmax` of the cross-covariance, if the MFLike data file's metadata lacks them |

Every file except `grid_spec.yaml`, `paths.yaml` and `cross_cov.yaml` is a piece of a
Cobaya `info`; the files of `probes/<probe>/` are merged in alphabetical order, and
`camb.yaml` / `classy.yaml` only when that code is used.

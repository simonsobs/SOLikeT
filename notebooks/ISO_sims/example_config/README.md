# Example configuration

A public configuration for the `ISO_sims` notebooks, so they can be run without
access to the private ISO configuration (`simonsobs/soliket-config-files`). It has
the same layout and the same modelling choices; only `paths.yaml` differs. Its
"mocks" are public files:

- the MFLike and CMB-lensing data shipped with the likelihoods (in the cobaya
  packages path, installed by `create_datasets.ipynb`);
- the galaxy x kappa test file `tests/data/unwise_g-so_kappa.sim.sacc.fits`. It was
  not made at this configuration's fiducial, so fitting it directly gives a large
  chi-square; its smooth twin is what the example checks.

The smooth twins and the cross-covariance are written to `$ISO_DATA/twins/example/`.

```bash
export ISO_CONFIG=/path/to/SOLikeT/notebooks/ISO_sims/example_config
export ISO_DATA=/path/to/scratch_folder
```

See `notebooks/ISO_sims/README.md` for the rest of the setup. Each file of
`probes/<probe>/` is merged into the likelihood's Cobaya `info`; `camb.yaml` /
`classy.yaml` only when that Boltzmann code is used.

This folder is public: never copy files of the private configuration into it.

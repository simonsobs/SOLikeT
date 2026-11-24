# ISO_sims defaults

Every fiducial value, prior, modelling choice and file path used by the ISO_sims
notebooks lives in this folder. The notebooks read it only through
[`../iso_config.py`](../iso_config.py); nothing comes from `soliket/presets`.

## The rule: every YAML here is a piece of a Cobaya `info` dict

Each file only uses the top-level keys `likelihood`, `theory`, `params` and
`sampler`. A run's `info` is built by merging pieces with Cobaya's own
`cobaya.input.merge_info`, in the same order `cobaya-grid` uses:

```
info = base  ⊕  model  ⊕  dataset
```

So the `info` a notebook builds for a (dataset, model) pair is exactly the input
of the corresponding grid job.

The three exceptions are `grid_spec.yaml`, `paths.yaml` and `sims/`, which
describe *which* pieces to combine and *where* files are, not Cobaya input.

## Layout

```
defaults/
├── cosmo.yaml            base cosmology (fiducial + priors), incl. neutrino mass sum
├── sampler.yaml          sampler settings shared by every run
├── theory/
│   ├── camb.yaml         camb neutrino setup (camb's built-in normal hierarchy)
│   └── classy.yaml       classy equivalent (fixed NH masses)
├── models/               one file per model extension (lcdm = nothing extra)
│   ├── mnu.yaml
│   └── neff.yaml
├── probes/               one folder per likelihood
│   ├── mflike/
│   │   ├── likelihood.yaml   likelihood block, pointing at the shipped data
│   │   ├── foreground.yaml   foreground params + BandpowerForeground theory
│   │   ├── systematics.yaml  calibration / bandpass params
│   │   ├── camb.yaml         camb precision this probe needs   (optional)
│   │   └── sims.yaml         override pointing at the smooth twin (optional)
│   ├── lensing/
│   │   ├── likelihood.yaml
│   │   ├── camb.yaml
│   │   └── sims.yaml
│   └── galaxy_kappa/
│       ├── likelihood.yaml   galaxy x kappa cross-spectrum, on the CCL simulation
│       ├── bias.yaml         b1 (sampled) and s1
│       ├── theory.yaml       soliket.CCL
│       ├── camb.yaml         non-linear P(k) precision
│       └── sims.yaml         override pointing at the smooth twin
├── grid_spec.yaml        which models, datasets and groups exist; theory code
├── paths.yaml            file locations, relative to notebooks/ISO_sims/
└── sims/
    └── galaxy_kappa.yaml settings of the CCL galaxy x kappa simulation
```

## How a probe folder is read

All `.yaml` files in `probes/<probe>/` are merged in alphabetical order, except:

- `camb.yaml` / `classy.yaml`: only merged when that Boltzmann code is used;
- `sims.yaml`: only merged when analysing the smooth twins in `sims/`
  (`data="sims"`, the default). With `data="shipped"` the likelihood reads
  the data shipped with the cobaya packages, which is what the
  `create_*` notebooks need to build the twins.

Missing optional files simply contribute nothing. Adding a probe means adding a
folder here and listing it in `grid_spec.yaml`.

## Placeholders

Strings may contain `{name}`, replaced by `iso_config.py`:

- `{packages}`: the cobaya packages path;
- any key of `paths.yaml` (e.g. `{mflike_smooth}`): that file's absolute path.

## Dual parameters

A parameter with a `prior` is sampled by `cobaya-grid`. In the notebooks,
`iso_config.fix_to_fiducial(params, sample=[...])` fixes every such parameter
that is not listed to its fiducial value, read from `ref` (its `loc`, or the
middle of its range), falling back to the `prior`. So `ref.loc` is both the
fiducial value and the centre of the MCMC starting point: keep its `scale`
small.

# SOLikeT
[![Build Status](https://github.com/simonsobs/soliket/workflows/Testing/badge.svg)](https://github.com/simonsobs/SOLikeT/actions?query=workflow%3ATesting)

**SO Likelihoods and Theories**

A centralized package for likelihood and theory implementations for SO.

## Installation

For a set of detailed instructions for different machines (e.g. NERSC), please see [here](INSTALL.rst).

To install SOLikeT we expect that you have the following system-level tools:
  - python>=3.7,<3.11
  - pip
  - compilers (c, cxx, fortran)
  - cmake
  - swig
  - gsl
  - fftw
  - cython
  - mpi4py

A convenient way to obtain these things (along with the python dependencies listed in requirements.txt) is through using the conda environment in soliket-tests.yml. This conda environment is the one we use for running tests.

You can then install SOLikeT in the usual way with pip:

```
git clone https://github.com/simonsobs/soliket
cd soliket
pip install -e .
```

### Optional Extras

In order to use the CosmoPower Theories within SOLikeT you will need to additionally install CosmoPower (and with it tensorflow, which is rather heavy and hence left out of the default installation).

This should be easily achievable with:

```
pip install cosmopower
```

## Running Tests

Running tests

There are (at least) two reasons you might want to run tests:

1. To see if tests you have written when developing SOLikeT are valid and will pass the Continuous Integration (CI) tests which we require for merging on github.

If you are using conda, the easiest way to run tests (and the way we run them) is to use tox-conda
```
pip install tox-conda
tox -e test
```

This will create a fresh virtual environment replicating the one which is used for CI then run the tests (i.e. without touching your current environment). Note that any args after a '--' string will be passed to pytest, so

```
tox -e test -- -k my_new_module
```

will only run tests which have names containing the string 'my_new_model', and 

```
tox -e test -- -pdb
```

will start a pdb debug instance when (sorry, _if_) a test fails.

2. Check SOLikeT is working as intended in an environment of your own specification.

For this you need to make sure all of the above system-level and python dependencies are working correctly, then run:
```
pytest -v soliket
```

Good luck!

Please raise an issue if you have trouble installing or any of the tests fail.

## Contains

This repo currently implements the following specific likelihoods:

* `MFLike`: the SO LAT multi-frequency TT-TE-EE power spectrum likelihood. (Adapted from, and tested against, the original implementation [here](https://github.com/simonsobs/lat_mflike)).
* `ClusterLikelihood`: An unbinned SZ-cluster count likelihood based on the original ACT SZ clusters likelihood.
* `LensingLikelihood`: Lensing power-spectrum likelihood, adapted from [here](https://github.com/simonsobs/so-lenspipe/blob/6abdc185764894cefa76fd4666243669d7e8a4b0/bin/SOlikelihood/cobayalike.py#L80).
* `LensingLiteLikelihood`: A no-frills, simple $\chi^2$ lensing power spectrum.
* `CrossCorrelationLikelihood`: A likelihood for cross-power spectra between galaxy surveys and CMB lensing maps.
* `XcorrLikelihood`: An alternative likelihood for cross-power spectra between galaxy surveys and CMB lensing maps.

## Contributing

If you would like to contribute to SOLikeT, either addressing any of our [Issues](https://github.com/simonsobs/SOLikeT/issues), adding features to the current likelihoods please read our [Contributor Guidelines](CONTRIBUTING.rst). If you plan on extending SOLikeT by adding new Likelihoods, please also have a look at the [guidelines for doing this](guidelines.md).

## Extending

Please see [these guidelines](guidelines.md) for instructions on bringing a new likelihood into **soliket**.

## Usage

These likelihoods are designed for direct use with **cobaya**.  This means that 
they may be specified directly when creating a **cobaya** `Model`.  E.g., if
you wanted to compute the likelihood of the simulated lensing data, you could do the following:

```python

from cobaya.yaml import yaml_load
from cobaya.model import get_model

info_yaml = """
debug: True

likelihood:
  soliket.LensingLiteLikelihood:
    sim_number: 1
    stop_at_error: True

params:
  # Sampled
  logA:
    prior:
      min: 2.6
      max: 3.5
    proposal: 0.0036
    drop: True
    latex: \log(10^{10} A_\mathrm{s})
  As:
    value: "lambda logA: 1e-10*np.exp(logA)"
    latex: A_\mathrm{s}
  ns:
    prior:
      min: 0.9
      max: 1.1
    proposal: 0.0033
    latex: n_\mathrm{s}


theory:
  camb:
    stop_at_error: False
    extra_args:
      lens_potential_accuracy: 1

"""

info = yaml_load(info_yaml)
model = get_model(info)
```
The likelihood could then be either directly computed as 
```python
model.loglike(dict(logA=3.0, ns=0.98))
```
and used outside of **cobaya** (e.g., directly passed to **emcee** or some other
sampler or optimizer), or this same YAML setup (with an additional 'sampler' block specified) 
could be used as input to `cobaya-run` to have **cobaya** manage the sampling.

For more information on how to use **cobaya**, check out its [documentation](http://cobaya.readthedocs.io).


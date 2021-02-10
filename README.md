# SOLikeT
![Build Status](https://github.com/simonsobs/soliket/workflows/Testing/badge.svg)

**SO Likelihoods and Theories**

A centralized package for likelihood and theory implementations for SO.

## Installation

```
git clone https://github.com/simonsobs/soliket
cd soliket
pip install -e .
```
You will also need to either run
```
pip install camb
```
or, for a fuller cobaya install:
```
cobaya-install cosmo -p /your/path/to/cobaya/packages
```
To run tests, you will also need the original LAT_MFlike package:
```
pip install git+https://github.com/simonsobs/lat_mflike
```
Then, you can run tests with 
```
pip install pytest
pytest -v .
```

Please raise an issue if you have trouble installing or any of the tests fail.

## Contains

This repo currently implements the following specific likelihoods:

* `MFLike`: the SO LAT multi-frequency TT-TE-EE power spectrum likelihood. (Adapted from, and tested against, the original implementation [here](https://github.com/simonsobs/lat_mflike)).
* `ClusterLikelihood`: An SZ-cluster count likelihood based on the original ACT SZ clusters likelihood.
* `LensingLikelihood`: Lensing power-spectrum likelihood, adapted from [here](https://github.com/simonsobs/so-lenspipe/blob/6abdc185764894cefa76fd4666243669d7e8a4b0/bin/SOlikelihood/cobayalike.py#L80).
* `LensingLiteLikelihood`: A no-frills, simple $\chi^2$ lensing power spectrum.

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


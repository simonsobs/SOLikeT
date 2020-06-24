# solt
[![Build Status](https://travis-ci.com/simonsobs/Likelihoods.svg?token=XsF5PBVv1xW2zmK74yrc&branch=master)](https://travis-ci.com/simonsobs/Likelihoods)

A centralized package for **cobaya**-compatible likelihood functions for SO.

## Installation

Clone this repository, then
```
python setup.py install
```
will install the **solt** package.

## Contains

This repo implements the following likelihoods, as initial demonstrations:

* `LensingLiteLikelihood`: a $\chi^2$ lensing power spectrum likelihood function
that takes as options the locations of a $C_l$ data file (`'datapath'`) and a
covariance matrix (`'covpath'`).  The simulated data used in this likelihood
is included in and installed with this package.
* `SimulatedLensingLiteLikelihood`: This subclasses the above, with its options
being the path to a directory that contains the simulations (`'dataroot'`),
the file path/patterns to the $C_l$ and covariance files, and a simulation 
number (`'sim_number'`).  The main purpose for this is an exercise to
demonstrate how to subclass a likelihood while changing its options.

More likelihoods coming soon!

## Usage

These likelihoods are designed for direct use with **cobaya**.  This means that 
they may be specified directly when creating a **cobaya** `Model`.  E.g., if
you wanted to compute the likelihood of the simulated lensing data with only 
$log A$ and $n_s$ as free parameters, you could do the following:

```python

from cobaya.yaml import yaml_load
from cobaya.model import get_model

info_yaml = """
likelihood: 
    solt.SimulatedLensingLiteLikelihood:
        sim_number: 1
theory: 
    classy:
        extra_args:
            output: lCl, tCl        
params:
    logA:
        prior:
          min: 1.61
          max: 3.91
        ref:
          dist: norm
          loc: 3.05
          scale: 0.001
        proposal: 0.001
        latex: \log(10^{10} A_\mathrm{s})
        drop: True

    A_s:
        value: 'lambda logA: 1e-10*np.exp(logA)'
        latex: A_\mathrm{s}

    n_s:
        prior:
            min: 0.8
            max: 1.2
        ref:
            dist: norm
            loc: 0.965
            scale: 0.004
        proposal: 0.002
        latex: n_\mathrm{s}
"""

info = yaml_load(info_yaml)
model = get_model(info)
```
The likelihood could then be either directly computed as 
```python
model.loglike(dict(logA=3.0, n_s=0.98))
```
and used outside of **cobaya** (e.g., directly passed to **emcee** or some other
sampler or optimizer), or this same YAML setup (with an additional 'sampler' block specified) 
could be used as input to `cobaya-run` to have **cobaya** manage the sampling.

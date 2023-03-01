=======
SOLikeT
=======

.. image:: https://github.com/simonsobs/soliket/workflows/Testing/badge.svg
   :target: https://github.com/simonsobs/SOLikeT/actions?query=workflow%3ATesting
   :alt: Testing Status

.. image:: https://readthedocs.org/projects/soliket/badge/?version=latest
   :target: https://soliket.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation Status

**SO Likelihoods and Theories**

A centralized package for likelihood and theory implementations for SO.


Installation
============

For a set of detailed instructions for different machines (e.g. NERSC), please see :ref:`the installation page <INSTALL>`.

To install SOLikeT we expect that you have the following system-level tools:

* python>=3.7,<3.11
* pip
* compilers (c, cxx, fortran)
* cmake
* swig
* gsl
* fftw
* cython
* mpi4py

A convenient way to obtain these things (along with the python dependencies listed in requirements.txt) is through using the conda environment in soliket-tests.yml. This conda environment is the one we use for running tests.

You can then install SOLikeT in the usual way with pip::

  git clone https://github.com/simonsobs/soliket
  cd soliket
  pip install -e .


Optional Extras
---------------

In order to use the CosmoPower Theories within SOLikeT you will need to additionally install CosmoPower (and with it tensorflow, which is rather heavy and hence left out of the default installation).

This should be easily achievable with::

  pip install cosmopower


Running Tests
=============

There are (at least) two reasons you might want to run tests:

1. To see if tests you have written when developing SOLikeT are valid and will pass the Continuous Integration (CI) tests which we require for merging on github.

If you are using conda, the easiest way to run tests (and the way we run them) is to use tox-conda::

  pip install tox-conda
  tox -e test

This will create a fresh virtual environment replicating the one which is used for CI then run the tests (i.e. without touching your current environment). Note that any args after a '--' string will be passed to pytest, so::

  tox -e test -- -k my_new_module

will only run tests which have names containing the string 'my_new_model', and ::

  tox -e test -- -pdb

will start a pdb debug instance when (sorry, *if*) a test fails.

2. Check SOLikeT is working as intended in an environment of your own specification.

For this you need to make sure all of the above system-level and python dependencies are working correctly, then run::

  pytest -v soliket

Good luck!

Please raise an issue if you have trouble installing or any of the tests fail.

====================================
SOLikeT: SO Likelihoods and Theories
====================================

|workflow-badge| |coverage-badge| |docs-badge|

.. |workflow-badge| image:: https://github.com/simonsobs/soliket/workflows/Testing/badge.svg
   :target: https://github.com/simonsobs/SOLikeT/actions?query=workflow%3ATesting
   :alt: Testing Status   
.. |coverage-badge| image:: https://codecov.io/gh/simonsobs/SOLikeT/branch/master/graph/badge.svg?token=ND945EQDWR 
   :target: https://codecov.io/gh/simonsobs/SOLikeT
   :alt: Test Coverage
.. |docs-badge| image:: https://readthedocs.org/projects/soliket/badge/?version=latest
   :target: https://soliket.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation Status

SOLikeT is a centralized package for likelihood and theory implementations for the `Simons Observatory <https://simonsobservatory.org/>`_.
For more extensive details please see our main documentation pages at: `http://soliket.readthedocs.io/ <http://soliket.readthedocs.io/>`_.

.. image:: docs/images/Sky_UCSD2b.jpg
  :target: https://simonsobservatory.org/
  :alt: Simons Observatory Logo
  :width: 200

Installation
============

For a set of detailed requirements and installation instructions for different machines (e.g. NERSC, M1 Mac), please see `the installation page <INSTALL.rst>`_.

A preferred and convenient way to install SOLikeT and its dependents is through using the conda environment defined in `soliket-tests.yml <soliket-tests.yml>`_. After installing an anaconda distribution (e.g. as described `here <https://docs.anaconda.com/free/anaconda/install/index.html>`_), you can create the environment and install a locally cloned version of SOLikeT using pip::

  git clone https://github.com/simonsobs/soliket
  cd soliket
  conda env create -f soliket-tests.yml
  conda activate soliket-tests
  pip install -e .


Running an Example
==================

SOLikeT is a collection of modules for use within the Cobaya cosmological inference and sampling workflow manager. Please see `the Cobaya documentation <https://cobaya.readthedocs.io/en/latest/>`_ for detailed instructions on how to use Cobaya to perform cosmological calculations and generate constraints on cosmological parameters.

SOLikeT examples and explanatory notebooks are under construction, but will be run using standard [yaml](https://en.wikipedia.org/wiki/YAML) format (which can in turn be read in as Python dictionaries). The examples will be run using something similar to::

  cobaya-run examples/example_1.yaml


Developing SOLikeT Theories and Likelihoods
===========================================

If you wish to develop your own Theory and Likelihood codes for use in SOLikeT please see the detailed instructions on the `Developer Guidelines <docs/developers.rst>`_ page.

Running Tests
=============

Tests run a set of SOLikeT calculations with known expected results. There are (at least) two reasons you might want to run tests:

Checking code in development
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
To see if codes you have written when developing SOLikeT are valid and will pass the Continuous Integration (CI) tests which we require for merging on github.

If you are using conda, the easiest way to run tests (and the way we run them) is to use tox-conda::

  pip install tox-conda
  tox -e test

This will create a fresh virtual environment replicating the one which is used for CI then run the tests (i.e. without touching your current environment). Note that any args after a '--' string will be passed to pytest, so::

  tox -e test -- -k my_new_module

will only run tests which have names containing the string 'my_new_model', and ::

  tox -e test -- -pdb

will start a pdb debug instance when (sorry, *if*) a test fails.

Checking environment configuration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Check SOLikeT is working as intended in a python environment of your own specification (i.e. you have installed SOLikeT not using the solike-tests conda environment).


For this you need to make sure all of the required system-level and python dependencies described in `the installation instructions <INSTALL.rst>`_ are working correctly, then run::

  pytest -v soliket

Good luck!

Please raise an issue if you have trouble installing or any of the tests fail.

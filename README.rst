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

SOLikeT can be installed using any of three methods: `uv`, `pip`, or `conda`, with the recommended method being `uv`.

Quick install with uv
---------------------

The recommended method is `uv`, which offers fast, reproducible installations using a lockfile and can manage virtual environments automatically. It is extremely flexible offering complete control over the environment and dependencies. Indeed, `uv` substitutes for `pip` and `conda` in many cases, providing a unified interface for package management. Still, it can be used seamlessly within a `conda` environment and is compatible with existing `pip` and `conda` workflows. 

One of the key features of `uv` is producing a lockfile (`uv.lock`) that captures the exact versions of all installed packages, ensuring reproducibility across different platforms and Python versions. This is particularly useful to avoid the "it works on my machine" problem, as it allows you to recreate the same environment on any machine with a single command.

Installing `uv` is as simple as running:

.. code-block:: bash

  pip install uv

Then, after activating an existing environment (both `conda` or `venv` environments are supported), you can clone the repository and install it via:

.. code-block:: bash

  git clone https://github.com/simonsobs/soliket
  cd soliket
  uv sync --locked

`uv sync --locked` will install all the necessary dependencies to the fixed versions stored in the `uv.lock` file. This ensures that you have a consistent and reproducible environment, identical to the one developed and tested.

At this point, you can forget about `uv`, if you want to, and continue using SOLikeT as desired. If you want to re-sync your environment with the lockfile, you can re-run that command.

If you need to use `CosmoPower` emulator, you can specify the `emulator` extra when running `uv sync`. This will install all the necessary dependencies related to `CosmoPower`:
.. code-block:: bash

  uv sync --locked --extra emulator

If you are less worried about reproducibility, you can also install SOLikeT without the lockfile by using `pip` after cloning the repository:

.. code-block:: bash
  git clone https://github.com/simonsobs/soliket
  cd soliket
  pip install .
  
In this case, you can also specify the `emulator` extra to install the dependencies related to `CosmoPower`:

.. code-block:: bash

  pip install .[emulator]

For further details on `uv` and alternatives ways to install SOLikeT, please refer to `the installation page <INSTALL.rst>`_.

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

If you are using `uv`, the easiest way to run tests (and the way we run them) is to use::

.. code-block:: bash

  uv run pytest -vv --durations=10

`-vv` will give you verbose output, and `--durations=10` will show you the 10 slowest tests, which can help identify performance issues. `uv` will automatically use the current environment, so you don't need to worry about activating a specific virtual environment.

If the current environment does not have the required dependencies, `uv` will install them automatically based on the `uv.lock` file, ensuring that you have all the necessary packages to run the tests.

You can also test a subset of tests or run specific tests by passing additional arguments to `pytest`. For example, if you want to run only the tests in a specific module, you can do

.. code-block:: bash
  uv run pytest -vv --durations=10 -k my_new_module

searching for tests that match the string 'my_new_module'.

If you want to run the tests using `pytest` directly, you can do so by running:

.. code-block:: bash

  pytest -vv soliket

This will run the tests in the same way as `uv`, but without the additional features provided by `uv`. Note that you will need to have all the required dependencies installed in your current environment for this to work.

Indeed, running tests after installing SOLikeT in any environment is a good practice to ensure that everything is working as expected (see `the installation instructions <INSTALL.rst>`_).

Please raise an issue if you have trouble installing or any of the tests fail.

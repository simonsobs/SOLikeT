==============================
General Development Guidelines
==============================

This page describes dow to develop new code within SOLikeT, including changing existing components and adding new ones.

GitHub Workflow
===============

Code development is done via Github. More detailed instructions on each step can be found in the `workflow guidelines <workflow.html>`_, but a brief summary is:

1. Identify development work
2. Open a Github Issue
3. Clone the repository
4. Create a new branch
5. Create a draft Pull Request (PR) from the branch
6. Develop the code including tests and docs
7. Ensure tests pass
8. Convert the PR from draft and request a Code Review
9. Merge the code
10. |:tropical_drink:|

Project Structure and Conventions
=================================

SOLikeT contains two conceptual types of code modules: Likelihoods and Theories.

Likelihoods compute a likelihood value from a comparison between a data vector and a prediction for that data vector at some set of values of input parameters (e.g. cosmological and nuisance parameters). For example a Gaussian likelihood comparing Cosmic Microwave Background (CMB) power spectra with a Lambda-CDM model prediction.

Theories perform calculations necessary to create the predicted data vector given the set of parameters. For instance solving the coupled Einstein-Boltzmann equations necessary to predict the CMB power spectrum within a given model.

General guidance on how to create Theories and Likelihoods can be found within the Cobaya documentation.

Directories and naming
----------------------

Within SOLikeT we request that you create new modules inside a directory of the same name::

 SOLikeT/soliket/my_module/my_module.py

Likelihood classes should be named in CamelCase and have a ``Likelihood`` as a suffix, e.g.::

 class MyModuleLikelihood(GaussianLikelihood)

Default parameters
------------------

Default parameters for a Likelihood or Theory can be stored in a ``.yaml`` file next to the ``.py`` file in which the Likelihood or Theory class is defined, and with the same name as the Likelihood or Theory, i.e.::

 SOLikeT/soliket/my_module/MyModuleLikelihood.yaml

Conventions for Likelihoods
---------------------------

Is your new Likelihood **newlike** Gaussian, Poisson or Cash-C?  If so, great; if not, then we need to do some prep work to implement a generic version of the new likelihood form into **SOLikeT**, alongside ``GaussianLikelihood``, ``PoissonLikelihood`` and ``CashCLikelihood``.

* Write likelihood code so as to inherit from ``GaussianLikelihood``, implementing ``_get_data()``, ``_get_cov()``, ``_get_theory()`` methods, etc.
* Also, if there is substantial data to be used by the likelihood, have it also extend ``_InstallableLikelihood`` (see ``soliket.mflike`` and ``soliket.lensing`` for examples).
* Factor out all cosmological/astrophysical calculations necessary to compute the "theory vector" into separate standalone ``Theory`` objects (current example of this is the ``Foreground`` object in ``soliket.mflike``.)
* If any of your new modules require physical constants, please make use of the ``constants.py`` module in SOLikeT (if you need to add new entries) and import it as needed. This would avoid inconsistent definitions of potentially shared quantities. Don't re-define constants in your own modules.

Conventions for Theories
------------------------

The detailed guidelines for developing new Theory components in SOLikeT can be found in the `theory component guidelines <docs/theory-component-guidelines.rst>`_ page, but a brief summary is:

- Your theory calculator must inherit from the Cobaya theory class.
- It must have 3 main blocks of functions: initialization (`initialize`); requirements (`get_requirement`, `must_provide`); calculations (`get_X`, `calculate`).
- The `initialize` function is where you can assign parameters and perform calculations that need to be done once for all.
- The `get_requirement` function specifies what external elements are needed by your theory block to perform its duties, returning a dictionary with the names of the required elements as keys.
- The `must_provide` function is used to assign values to parameters needed to compute the required elements and can also specify additional requirements for the theory block.
- In each Theory class, you need at least two functions:
  1. A `get_X` function that returns the current state of the required element.
  2. A `calculate` function that performs the actual calculations and updates the state of the required element.

Code Style
==========

All contributions should follow the `PEP8 Style Guide for Python Code <https://www.python.org/dev/peps/pep-0008/>`_. When a PR is created for SOLikeT, a check will be run to make sure your code complies with these recommendations, which are the same as those specified for `Cobaya <https://cobaya.readthedocs.io/>`_. This means the following checks will be made:

.. code-block:: bash

  E713,E704,E703,E714,E741,E10,E11,E20,E22,E23,E25,E27,E301,E302,E304,E9,F405,F406,F5,F6,F7,F8,W1,W2,W3,W6

and a line length limit of 90 characters will be applied.

This will be run automatically for you at commit time if you have `pre-commit <https://pre-commit.com/>`_ installed, which is highly recommended. To install pre-commit, run:

.. code-block:: bash

  pip install pre-commit
  pre-commit install

This will install the pre-commit hooks, and in particular the `ruff <https://docs.astral.sh/ruff/>`_ tool, which will check your code for style issues and formatting before you commit it. You can also run `ruff <https://docs.astral.sh/ruff/>`_ manually with the command:

.. code-block:: bash

  ruff check --fix . --config ./pyproject.toml
  ruff format . --config ./pyproject.toml

Unit Tests
==========

Pull requests will require existing unit tests to pass before they can be merged. Additionally, new unit tests should be written for all new public methods and functions. Unit tests for each Likelihood and Theory should be placed in the tests directory with a name matching that of the python file in which the class is defined

.. code-block:: bash

 SOLikeT/soliket/tests/test_my_module.py


For Likelihoods we request that there is a test which compares the result of a likelihood calculation to a precomputed expected value which is hard coded in the tests file, to a tolerance of ``1.e-3``

.. code-block:: bash

  assert np.isclose(loglike_just_computed, -25.053, rtol=1.e-3)

For more advice on how to write tests see the `Astropy Testing Guidelines <https://docs.astropy.org/en/stable/development/testguide.html>`_.

Tests run a set of SOLikeT calculations with known expected results. There are (at least) two reasons you might want to run tests:

Checking code in development
----------------------------
To see if codes you have written when developing SOLikeT are valid and will pass the Continuous Integration (CI) tests which we require for merging on github.

To run tests, you can use the following command:

.. code-block:: bash

   uv run pytest -vv --durations=10  # using uv
   pytest -vv --durations=10         # using pip or conda

This command will run all tests in the SOLikeT codebase and provide verbose output, including the duration of each test. The `--durations=10` option will show the 10 slowest tests, which can help identify performance bottlenecks.

If the current environment does not have the required dependencies, `uv` will install them automatically based on the `uv.lock` file, ensuring that you have all the necessary packages to run the tests.

You can also test a subset of tests or run specific tests by passing additional arguments to `pytest`. For example, if you want to run only the tests in a specific module, you can do

.. code-block:: bash
  uv run pytest -vv --durations=10 -k my_new_module

searching for tests that match the string 'my_new_module'.

`uv` provides a very easy but powerful way to test your new feature in depth locally (if you really want to). In fact, you can install different Python versions without needing to set up multiple environments manually. You can install multiple Python version and pin the one you want to test with:

.. code-block:: bash

  uv python install 3.10 # install Python 3.10 or any other version you want to test
  uv python pin 3.10 # pin the current environment to Python 3.10

Then, provided that the version is compatible with SOLikeT, you can run the tests with that version:

.. code-block:: bash

  uv run pytest -vv --durations=10

`uv` will automatically use the pinned Python version to run the tests, check the `uv.lock` file for the correct dependencies, and ensure that your code is compatible with that version.

Checking environment configuration
----------------------------------
Check SOLikeT is working as intended in a python environment of your own specification (i.e. you have installed SOLikeT without following our guide).

For this you need to make sure all of the required system-level and python dependencies described in `the installation instructions <install.html>`_ are working correctly, then run

.. code-block:: bash

  uv run pytest -vv soliket # or
  pytest -vv soliket

Skipping tests
--------------

If you want to skip all CI tests, it is possible to do so by using the prefix `[skipci]` or `[skip ci]` in the commit message of your PR. This is useful if you are making changes that do not affect the code, such as documentation updates or minor formatting changes. Still, assuming that some change to the code is made, you should be completely sure that you are not introducing any bugs or issues before skipping the tests, as this can lead to problems down the line.

If you are working on a pure documentation update, or something similar, you can skip tests for all commits in your PR by using the same prefix in the PR title. This will prevent the CI tests from running, which can save time and resources.

Good luck!

Documentation
=============

Along with writing your code and creating tests we also ask that you create documentation for any work you do within SOLikeT, which is then listed on our documentation page `http://soliket.readthedocs.io <http://soliket.readthedocs.io>`_.

Code should be annotated with docstrings which can be automatically parsed by the sphinx tool. See `here for a syntax reference <https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html>`_. You should then create a page in the ``/docs`` folder of the repository on which the code is to be listed, and add the new page to the index.

Detailed instructions and examples on how to do this can be found in our `documentation guide <documentation.html>`_.

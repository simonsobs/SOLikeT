==========================
Information for Developers
==========================

This page describes dow to develop new code within SOLikeT, including changing existing components and adding new ones.

GitHub Workflow
===============

Code development is done via Github. More detailed instructions on each step can be found in the `workflow guidelines <workflow.html>`_, but a brief summary is:

* Identify development work
* Open a Github Issue
* Clone the repository
* Create a new branch
* Create a draft Pull Request (PR) from the branch
* Develop the code including tests and docs
* Ensure tests pass
* Convert the PR from draft and request a Code Review
* Merge the code

Project Structure
================

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


Code Style
==========

All contributions should follow the `PEP8 Style Guide for Python Code <https://www.python.org/dev/peps/pep-0008/>`_. When a PR is created for SOLikeT, a check will be run to make sure your code complies with these recommendations, which are the same as those specified for `Cobaya <https://cobaya.readthedocs.io/>`_. This means the following checks will be made:

::

  E713,E704,E703,E714,E741,E10,E11,E20,E22,E23,E25,E27,E301,E302,E304,E9,F405,F406,F5,F6,F7,F8,W1,W2,W3,W6

and a line length limit of 90 characters will be applied.

You may find it easier to run this check as locally before raising a PR. This can be done by running:

::

  tox -e codestlye

in the SOLikeT root directory.

The `black <https://black.readthedocs.io/en/stable/>`_ tool will also try to automatically format your code to abide by the style guide. It should be used with caution as it is irreversible (without a git revert), and can be run on any python files you create by running:

::

  black <py-file-you-created>

it is usually best to then inspect the file and correct any strange choices `black` has made.

Unit Tests
==========

Pull requests will require existing unit tests to pass before they can be merged. Additionally, new unit tests should be written for all new public methods and functions. Unit tests for each Likelihood and Theory should be placed in the tests directory with a name matching that of the python file in which the class is defined::

 SOLikeT/soliket/tests/test_my_module.py


For Likelihoods we request that there is a test which compares the result of a likelihood calculation to a precomputed expected value which is hard coded in the tests file, to a tolerance of ``1.e-3``::

  assert np.isclose(loglike_just_computed, -25.053, rtol=1.e-3)

For more advice on how to write tests see the `Astropy Testing Guidelines <https://docs.astropy.org/en/stable/development/testguide.html>`_.

Tests run a set of SOLikeT calculations with known expected results. There are (at least) two reasons you might want to run tests:

Checking code in development
----------------------------
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
----------------------------------
Check SOLikeT is working as intended in a python environment of your own specification (i.e. you have installed SOLikeT not using the soliket-tests conda environment).

For this you need to make sure all of the required system-level and python dependencies described in `the installation instructions <INSTALL.rst>`_ are working correctly, then run::

  pytest -v soliket

Good luck!

If your unit tests check the statistical distribution of a random sample, the test outcome itself is a random variable, and the test will fail from time to time. Please mark such tests with the ``@pytest.mark.flaky`` decorator, so that they will be automatically tried again on failure. To prevent non-random test failures from being run multiple times, please isolate random statistical tests and deterministic tests in their own test cases.

Documentation
=============

Along with writing your code and creating tests we also ask that you create documentation for any work you do within SOLikeT, which is then listed on our documentation page `http://soliket.readthedocs.io <http://soliket.readthedocs.io>`_.

Code should be annotated with docstrings which can be automatically parsed by the sphinx tool. See `here for a syntax reference <https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html>`_. You should then create a page in the ``/docs`` folder of the repository on which the code is to be listed, and add the new page to the index.

Detailed instructions and examples on how to do this can be found in our `documentation guide <docs_guidelines.html>`_.
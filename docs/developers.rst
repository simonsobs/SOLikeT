==========================
Information for Developers
==========================

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

Basic ingredients
^^^^^^^^^^^^^^^^^
Your theory calculator must inherit from the cobaya theory class. It must have 3 main blocks of functions: inizialization (``initialize``); requirements (``get_requirement``, ``must_provide``); calculations (``get_X``, ``calculate``).

In what follows, we will use the structure of ``mflike`` as a concrete example of how to build these 3 blocks. The new version of ``mflike`` in SOLikeT splits the original mflike in 4 blocks: one cobaya-likelihood component (``mflike``); three cobaya-theory components.

The three theory components are:
  1. ``TheoryForge``: this is where raw theory (CMB spectra) is mixed and modified with instrumental and non-cosmological effects
  2. ``Foreground``: this is where the foreground (fg) spectra are computed
  3. ``BandPass``: this is where bandpasses are built (either analytically or read from file)


Initialization
^^^^^^^^^^^^^^
You can either assign params in initialize or do that via a dedicated yaml. You can in general do all the calculations that need to be done once for all.

Requirements
^^^^^^^^^^^^
Here you need to write what external elements are needed by your theory block to perform its duties. These external elements will be computed and provided by some other external module (e.g., another Theory class).
In our case, ``mflike`` must tell us that it needs a dictionary of cmb+fg spectra. This is done by letting the get_requirement function return a dictionary which has the name of the needed element as a key. For example, if the cmb+fg spectra dict is called ``cmbfg_dict``, the get_requirement function should::

   return {"cmbfg_dict":{}}

The key is a dict itself. It can be empty, if no params need to be passed to the external Theory in charge of computing cmbfg_dict.
It might be possible that, in order to compute ``cmbfg_dict``, we should pass to the specific Theory component some params known by ``mflike`` (e.g., frequency channel). This is done by filling the above empty dict::

   {"cmbfg_dict": {"param1": param1_value, "param2": param2_value, etc}}

If this happens, then the external Theory block (in this example, ``TheoryForge``) must have a ``must_provide`` function. ``must_provide`` tells the code:
  1. what values should be assigned to the parameters needed to compute the element required from the Theory block. The required elements are stored in the ``**requirements`` dictionary which is the input of ``must_provide``.
   In our example, ``TheoryForge`` will assign to ``param1`` the ``param1_value`` passed from ``mflike`` via the ``get_requirement`` in ``mflike`` (and so on). For example:
   ::

      must_provide(self, **requirements):
         if "cmbfg_dict" in requirements:
            self.param1 = requirements["cmbfg_dict"]["param1"]

   if this is the only job of ``must_provide``, then the function will not return anything

   2. if needed, what external elements are needed by this specific theory block to perform its duties. In this case, the function will return a dictionary of dictionaries which are the requirements of the specific theory block. These dictionaries do not have to necessarily contain content (they can be empty instances of the dictionary), but must be included if expected. Note this can be also done via ``get_requirement``. However, if you need to pass some params read from the block above to the new requirements, this can only be done with ``must_provide``. For example, ``TheoryForge`` needs ``Foreground`` to compute the fg spectra, which we store in a dict called ``fg_dict``. We also want ``TheoryForge`` to pass to ``Foreground`` ``self.param1``. This is done as follows:
   ::

      must_provide(self, **requirements):
         if “cmbfg_dict” etc etc
            ...
         return {“fg_dict”: {“param1_fg”: self.param1}}

   Of course, ``Foreground`` will have a similar call to ``must_provide``, where we assign to ``self.param1_fg`` the value passed from ``TheoryForge`` to ``Foreground``.

Calculation
^^^^^^^^^^^
In each Theory class, you need at least 2 functions:

   1. A get function:
   ::

      get_X(self, any_other_param):
         return self.current_state[“X”]

   where "X" is the name of the requirement computed by that class (in our case, it is ``cmbfg_dict`` in ``TheoryForge``, ``fg_dict`` in ``Foreground``). ``any_other_param`` is an optional param that you may want to apply to ``current_state["X"]`` before returning it. E.g., it could be a rescaling amplitude. This function is called by the Likelihood or Theory class that has ``X`` as its requirement, via the ``self.provider.get_X(any_other_param)`` call.

   2. A calculate function:
   ::

      calculate(self, **state, want_derived=False, **params_values_dict):
         state[“X”] = result of above calculations

   which will do actual calculations, that could involve the use of some of the ``**params_value_dict``, and might also compute derived params (if ``want_derived=True``).

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

For this you need to make sure all of the required system-level and python dependencies described in `the installation instructions <install.html>`_ are working correctly, then run::

  pytest -v soliket

Good luck!

If your unit tests check the statistical distribution of a random sample, the test outcome itself is a random variable, and the test will fail from time to time. Please mark such tests with the ``@pytest.mark.flaky`` decorator, so that they will be automatically tried again on failure. To prevent non-random test failures from being run multiple times, please isolate random statistical tests and deterministic tests in their own test cases.

Documentation
=============

Along with writing your code and creating tests we also ask that you create documentation for any work you do within SOLikeT, which is then listed on our documentation page `http://soliket.readthedocs.io <http://soliket.readthedocs.io>`_.

Code should be annotated with docstrings which can be automatically parsed by the sphinx tool. See `here for a syntax reference <https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html>`_. You should then create a page in the ``/docs`` folder of the repository on which the code is to be listed, and add the new page to the index.

Detailed instructions and examples on how to do this can be found in our `documentation guide <documentation.html>`_.

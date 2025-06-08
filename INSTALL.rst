.. _INSTALL:

Install and run Cobaya+SOLikeT
==============================

SOLikeT can be installed using any of three methods: `uv`, `pip`, or `conda`.

Preferred Way: Using `uv` package manager
-----------------------------------------

SOLikeT can be installed in a fast and reproducible way using the `uv` package manager, which can create a virtual environment for you if you do not already have one, and will install all dependencies from a lockfile for full reproducibility.

The first step is to clone the SOLikeT repository:

.. code-block:: bash
   
   git clone https://github.com/simonsobs/soliket
   cd soliket 

To install SOLikeT using `uv` and following the preferred procedure, you can follow these steps: after installing a `conda` distribution (e.g. as described `here <https://docs.anaconda.com/free/anaconda/install/index.html>`_), you can create the environment and install SOLikeT using:

.. code-block:: bash

   conda create -n my_env python=3.x # create a fresh conda environment with Python 3.x (check supported versions)
   conda activate my_env
   pip install uv  # if you don't have uv already
   uv sync --locked # sync the environment with the uv.lock file

This will ensure full isolation of the environment and reproducibility of the installation, as it will install all the necessary dependencies fixed in the `uv.lock` file. This procedure is test and works on multiple platforms (explicitly tested for latest ubuntu, MacOS and Windows).

If instead you prefer `venv` to be created automatically, you can use:

.. code-block:: bash

   uv venv .venv   # create a virtual environment
   source .venv/bin/activate
   uv sync --locked

In order to use the CosmoPower Theories within SOLikeT you will need to additionally install CosmoPower (and with it tensorflow, which is rather heavy and hence left out of the default installation). To achieve this (see below for M1 Mac specific guide), you can add the `emulator` extra to your install command to install all necessary dependencies:

.. code-block:: bash

   uv sync --locked --extra emulator
   
At this point, you are ready to use SOLikeT!

Install with pip
----------------

Alternatively, you can use pip to install SOLikeT. This method is straightforward but does not provide the same level of reproducibility as `uv` since it does not use a lockfile. Nonetheless, the `pyproject.toml` file specifies the required dependencies, so you can still install a consistent set of packages. The procedure is as follows:

.. code-block:: bash

  git clone https://github.com/simonsobs/soliket
  cd soliket
  pip install .

Soon we will also provide a `soliket` package on PyPI, which will allow you to install the latest released version of SOLikeT using pip without needing to clone the repository:

.. code-block:: bash

  pip install soliket

In these cases, you can also install the `emulator` extra to include CosmoPower support:

.. code-block:: bash

   pip install .[emulator] # without PyPI release after cloning or
   pip install soliket[emulator] # with PyPI release

Install with conda
------------------

Another way to install SOLikeT and its dependents is through using the conda environment defined in `soliket-tests.yml <soliket-tests.yml>`_. After installing an anaconda distribution (e.g. as described `here <https://docs.anaconda.com/free/anaconda/install/index.html>`_), you can create the environment and install a the latest released version of SOLikeT using pip::

.. code-block:: bash

  git clone https://github.com/simonsobs/soliket
  cd soliket
  conda env create -f soliket-tests.yml
  conda activate soliket-tests
  pip install .

Similarly, you can install the latest released version of SOLikeT from PyPI into this environment:

.. code-block:: bash

  conda env create -f soliket-tests.yml
  conda activate soliket-tests
  pip install soliket 

As the `pip` installation, this will not provide the same level of reproducibility as `uv`, since the conda environment will be solved when you create it, but it will still ensure that you have a consistent set of packages installed, based on the `pyproject.toml`.

Installing for development
^^^^^^^^^^^^^^^^^^^^^^^^^^

If you wish to actively develop the code you can instead install the in-development version of SOLikeT from the github repository in editable mode. Also, we prepared an extra set of dependencies for development, which you can install using the `dev` extra. This will allow you to run tests and use the development tools provided by SOLikeT. To do this, you can sunstitute the relevenat commands in the previous section with the following (we assume you are in the `soliket` directory, cloned the repository and you want the `emulator` extra for `CosmoPower`` support):

.. code-block:: bash
   # uv case
   uv sync --locked --extra emulator --extra dev 
   # or pip case without PyPI release
   pip install -e .[emulator,dev]
   # or pip case with PyPI release
   pip install soliket[emulator,dev]

Harder Way: Preparing your own environment
------------------------------------------

**CREATE VIRTUAL CONDA ENV TO RUN COBAYA**
Based on `cobaya documentation <https://cobaya.readthedocs.io/en/latest/cluster_amazon.html>`_.

::

   $ sudo apt update && sudo apt install gcc gfortran g++ openmpi-bin openmpi-common libopenmpi-dev libopenblas-base liblapack3 liblapack-dev make
   $ wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
   $ bash miniconda.sh -b -p $HOME/miniconda
   $ export PATH="$HOME/miniconda/bin:$PATH"
   $ conda config --set always_yes yes --set changeps1 no
   $ conda create -q -n cobaya-env python=3.7 scipy matplotlib cython PyYAML pytest pytest-forked flaky
   $ source activate cobaya-env
   $ pip install mpi4py

**INSTALL COBAYA**

::

   $ pip install cobaya
   $ sudo apt install libcfitsio-bin libcfitsio-dev
   $ cobaya-install cosmo --packages-path cobaya_packages

**INSTALL SOLIKET**

::

   $ conda install -c conda-forge compilers pyccl
   $ git clone https://github.com/simonsobs/soliket
   $ cd soliket
   $ pip install -e .

At NERSC
--------

Based on Perlmutter. Note: you may want to run cobaya in the SCRATCH directory (see thread `here <https://github.com/CobayaSampler/cobaya/issues/219>`_).

**Build mpi4py in your custom conda environment**
Based on `NERSC documentation <https://docs.nersc.gov/development/languages/python/parallel-python/#mpi4py>`_.

::

  module load python
  conda create -n soliket-tests python numpy
  conda activate soliket-tests
  module swap PrgEnv-${PE_ENV,,} PrgEnv-gnu
  MPICC="cc -shared" pip install --force-reinstall --no-cache-dir --no-binary=mpi4py mpi4py

**Load fftw, gsl and cmake (WOULD BE NEEDED BY PYCCL)**

::

   module load cray-fftw
   module load gsl
   module load cmake

Please, make sure to load cmake as your last command, so that the path to cmake bin is prepended to your PATH (see `this issue <https://github.com/LSSTDESC/CCL/issues/542>`_).

**Install additional requirements and soliket**

::

   pip install pytest-cov make swig
   git clone https://github.com/simonsobs/soliket
   cd soliket
   pip install -e .

**BONUS: install cosmopower (optional)**
cosmopower is not automatically installed (see above). However, if you want to use this option, do the following:

::

   pip install cosmopower
   cobaya-install planck_2018_highl_plik.TTTEEE_lite_native

**Run soliket**

Create your job script following `cobaya docs <https://cobaya.readthedocs.io/en/devel/run_job.html>`_.

NOTE: Any time you log in to a new perlmutter shell, remember to always load the relevant modules (in particular, those needed for pyccl). A good option is to create an alias in your bashrc with all the relevant commands.

Please, don't hesitate to open issues and/or be in touch with us, should you find any problems. Many thanks to Luca Pagano, Serena Giardiello, Massimiliano Lattanzi, Pablo Lemos, +++

On M1 Mac
--------
There is an issue with installing tensorflow (needed for cosmopower) on M1 Mac that is likely to be solved in the future. For the moment, if you want to couple SOLikeT to cosmopower, please follow this guidance:

1. Download latest miniconda installer (e.g., `here: Download Conda environment <https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh>`_) and properly rename it (e.g., -> miniconda.sh)
2. Install miniconda and tensor flow-deps

::

   bash ~/miniconda.sh -b -p $HOME/miniconda
   source ~/miniconda/bin/activate
   conda install -c apple tensorflow-deps

3. git clone soliket and create your virtual env

::

   git clone https://github.com/simonsobs/soliket
   cd soliket
   conda env create -n my_env -f soliket-tests.yml
   conda activate my_env 

4. Install tensorflow-macos and metal with correct versioning

::

   pip install tensorflow-macos
   pip install tensorflow-metal

5. Download and install cosmopower manually

::

   git clone https://github.com/alessiospuriomancini/cosmopower
   cd cosmopower
   pip install .

6. Go back to soliket folder and install it

::

   cd path/to/your/soliket
   pip install -e .

Running Tests
=============

Irrespectively of the installation method you choose, you can run tests to ensure that SOLikeT is functioning correctly. These commands will equivalently run the tests using pytest, which is the testing framework used in SOLikeT:

.. code-block:: bash

   uv run pytest -vv  # using uv
   pytest -vv         # using pip or conda

`-vv` will give you verbose output.

Checking code in development
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you are developing new features or making changes to the SOLikeT codebase, running tests is also essential to ensure that your changes do not break existing functionality.

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

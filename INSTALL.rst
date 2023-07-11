.. _INSTALL:

Install and run Cobaya+SOLikeT
==============================

Preferred Way: Using the provided conda environment
---------------------------------------------------

We have provided a conda environment defined in `soliket-tests.yml <https://github.com/simonsobs/SOLikeT/blob/master/soliket-tests.yml>`_ which provides easy set up of a virtual envrinoment with all the dependencies installed in order to run SOLikeT and its tests on multiple platforms (explicitly tested for ubuntu and MacOS-11):

::

   $ conda env create --file soliket-tests.yml

Optional: Install CosmoPower
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
In order to use the CosmoPower Theories within SOLikeT you will need to additionally install CosmoPower (and with it tensorflow, which is rather heavy and hence left out of the default installation).

Unless using an M1 Mac this should be easily achievable with::

  $ pip install cosmopower

If you wish to install it using your own system tools (including new M1 Mac) some useful information is provided below.

Harder Way: Preparing your own conda environment
----------------------------------

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

**Test soliket**

::

   pytest -v soliket

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

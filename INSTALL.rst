.. _INSTALL:

Install and run Cobaya+SOLikeT
==============================

Using conda
-----------

We have provided a conda environment defined in `soliket-tests.yml <https://github.com/simonsobs/SOLikeT/blob/master/soliket-tests.yml>`_ which provides easy set up of a virtual envrinoment with all the dependencies installed in order to run SOLikeT and its tests on multiple platforms (explicitly tested for ubuntu and MacOS-11).

In order to use the CosmoPower Theories within SOLikeT you will need to additionally install CosmoPower (and with it tensorflow, which is rather heavy and hence left out of the default installation).

This should be easily achievable with::

  pip install cosmopower

If you wish to install it using your own system tools some useful information is provided below.

On your own laptop/virtual machine
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

Note: you may want to run cobaya in the SCRATCH directory (see thread `here <https://github.com/CobayaSampler/cobaya/issues/219>`_).

**CREATE A CONDA-ENV COPYING LAZY-MPI4PY AND USING GNU**
Based on `NERSC documentation <https://docs.nersc.gov/development/languages/python/parallel-python/#mpi4py>`_.

::

   module unload PrgEnv-intel
   module load PrgEnv-gnu
   module load python
   conda create --name my_mpi4py_env --clone lazy-mpi4py
   conda activate my_mpi4py_env

**INSTALL COBAYA**

::

   pip install cobaya
   cobaya-install cosmo --packages-path cobaya_packages

**LOAD CMAKE (WOULD BE NEEDED BY PYCCL)**

::

   module load cmake

**INSTALL SOLIKET**

::

   git clone https://github.com/simonsobs/soliket
   cd soliket
   pip install -e .

**RUN SOLIKET**

Create your job script following `cobaya docs <https://cobaya.readthedocs.io/en/devel/run_job.html>`_.

Many thanks to Luca Pagano, Serena Giardiello, Pablo Lemos, +++

On M1 Mac
--------
There is an issue with installing tensorflow (needed for cosmopower) on M1 Mac that is likely to be solved in the future. For the moment, if you want to couple SOLikeT to cosmopower, please follow this guidance:

1. Download latest miniconda installer (e.g., `here: Download Conda environment <https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh>`_) and properly rename it (e.g., -> miniconda.sh)
2. Install miniconda and tensor flow-deps

   bash ~/miniconda.sh -b -p $HOME/miniconda
   source ~/miniconda/bin/activate
   conda install -c apple tensorflow-deps

3. git clone soliket and create your virtual env

   git clone https://github.com/simonsobs/soliket
   cd soliket
   conda env create -n my_env -f soliket-tests.yml
   conda activate my_env 

4. Install tensorflow-macos and metal with correct versioning

   pip install tensorflow-macos
   pip install tensorflow-metal

5. Download and install cosmopower manually

   git clone https://github.com/alessiospuriomancini/cosmopower
   cd cosmopower
   pip install .

7. Go back to soliket folder and install it
   cd path/to/your/soliket
   pip install -e .

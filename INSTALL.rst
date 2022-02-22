INSTALL AND RUN COBAYA+SOLIKET
==============================

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

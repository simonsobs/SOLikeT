==============================
Tutorials
==============================

This page introduces the SOLikeT tutorial notebooks and how to run them locally to get familiar with the package.

Available tutorials
-----------------------

1. First step Notebook — The notebook file is located at ``notebooks/first_step_tutorial.ipynb``. It walks through:

   - Installing SOLikeT and optional extras
   - Loading example data and utilities
   - Running simple likelihoods and theory components
   - Interpreting outputs and plotting basic results

2. Coming soon 

Running the tutorial locally
----------------------------

To open and run the notebook on your machine:

- If you have installed SOLikeT via `uv <https://uv.readthedocs.io/en/latest/>`_, you can simply launch the notebook from the uv environment:

   .. code-block:: bash

      uv run jupyter notebook notebooks/first_step_tutorial.ipynb

- If SOLiket has not been installed via uv:
   1. Ensure you have a Python environment that meets SOLikeT requirements (see :doc:`install`). 
   2. Install Jupyter:

   .. code-block:: bash

      python -m pip install jupyter

  3. Launch Jupyter and open the notebook:

   .. code-block:: bash

      jupyter notebook notebooks/first_step_tutorial.ipynb


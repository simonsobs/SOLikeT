==============================
Tutorials
==============================

This page introduces the SOLikeT tutorial notebook and how to run it locally to get familiar with the package.

Available tutorials
-----------------------

- Getting Started Tutorial Notebook — The notebook file is located at ``notebooks/tutorial_soliket.ipynb``.

   It walks through:

   - Installing SOLikeT and optional extras
   - Loading example data and utilities
   - Running simple likelihoods and theory components
   - Interpreting outputs and plotting basic results

Running the tutorial locally
----------------------------

To open and run the notebook on your machine:

1. Ensure you have a Python environment that meets SOLikeT requirements (see :doc:`install`).
2. Install Jupyter and recommended extras:

   .. code-block:: bash

      python -m pip install jupyter
      python -m pip install .[all]

   The ``[all]`` extra installs common optional dependencies used across tutorials.

3. Launch Jupyter and open the notebook:

   .. code-block:: bash

      jupyter notebook notebooks/tutorial_soliket.ipynb


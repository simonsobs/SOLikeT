.. SOLikeT documentation master file, created by
   sphinx-quickstart on Mon Feb 27 12:10:57 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. image:: images/Sky_UCSD2b.jpg
   :target: https://simonsobservatory.org/
   :alt: Simons Observatory Logo
   :width: 200

====================================
SOLikeT: SO Likelihoods and Theories
====================================

|Read the Docs| |Github| |Codecov|

SOLikeT is a centralized package for Likelihood and Theory codes for the `Simons Observatory <https://simonsobservatory.org/>`_.

You can find the code on our `Github repository <https://github.com/simonsobs/soliket/>`_, where you can also help us develop it. If you have any questions or problems using SOLikeT please `open an Issue <https://github.com/simonsobs/soliket/issues>`_.

The pages here describe how to install and run SOLikeT, and document the functions available within it.

.. toctree::
   :caption: Getting started
   :maxdepth: 1
   
   install

.. toctree::
   :caption: Theory codes
   :maxdepth: 1
   
   ccl
   cosmopower
   bandpass
   foreground
   bias

.. toctree::
   :caption: Likelihood codes
   :maxdepth: 1

   mflike
   fgmarge
   lensing
   clusters
   xcorr
   crosscorrelation

.. toctree::
   :caption: Miscellaneous
   :maxdepth: 1

   utils

.. toctree::
   :caption: Development guidelines
   :maxdepth: 1
   
   developers
   workflow
   documentation


* :ref:`genindex`
* :ref:`search`


.. |Github| image:: https://github.com/simonsobs/soliket/workflows/Testing/badge.svg
   :target: https://github.com/simonsobs/SOLikeT/actions?query=workflow%3ATesting
   :alt: Testing Status

.. |Read the Docs| image:: https://readthedocs.org/projects/soliket/badge/?version=latest&label=Docs&logo=read%20the%20docs
   :target: https://soliket.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation Status

.. |Codecov| image:: https://codecov.io/gh/simonsobs/SOLikeT/branch/master/graph/badge.svg?token=ND945EQDWR
   :target: https://codecov.io/gh/simonsobs/SOLikeT
   :alt: Test Coverage

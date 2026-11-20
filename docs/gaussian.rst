Gaussian Likelihoods
====================

.. automodule:: soliket.gaussian

This module provides the base classes for Gaussian likelihoods in SOLikeT,
including support for combining multiple likelihoods with cross-covariances.

GaussianLikelihood
------------------

The base class for all Gaussian likelihoods. Subclasses should implement
the ``_get_theory()`` method to compute the theory prediction.

.. autoclass:: soliket.gaussian.GaussianLikelihood
    :exclude-members: initialize
    :members:
    :private-members:
    :show-inheritance:

MultiGaussianLikelihood
-----------------------

A likelihood that combines multiple Gaussian likelihoods, optionally
accounting for cross-covariances between them.

.. autoclass:: soliket.gaussian.MultiGaussianLikelihood
    :exclude-members: initialize
    :members:
    :private-members:
    :show-inheritance:

Usage Example
^^^^^^^^^^^^^

To combine multiple likelihoods (e.g., CMB TT and lensing) with cross-covariances:

.. code-block:: yaml

    likelihood:
      soliket.MultiGaussianLikelihood:
        components:
          - soliket.mflike.MFLike
          - soliket.lensing.LensingLikelihood
        options:
          - datapath: /path/to/mflike_data.fits
            use_spectra: all
          - datapath: /path/to/lensing_data.fits
        cross_cov_path: /path/to/cross_covariance.fits

The ``cross_cov_path`` parameter is optional. If not provided, the likelihoods
are assumed to be independent (zero cross-covariance).

Component likelihoods may apply different scale cuts, so their data vectors need
not share the same range: the cross-covariance is automatically trimmed to each
probe's retained bandpowers. Blocks are also aligned to the data by *identity*
(per-bandpower keys) rather than by position, so the order in which the
cross-covariance was built need not match the order produced at run time. See
`Identity alignment`_ below.

CrossCov
--------

A container for storing cross-covariances between likelihood components.
Supports saving and loading in SACC format.

.. autoclass:: soliket.gaussian.CrossCov
    :members:
    :show-inheritance:

Usage Modes
^^^^^^^^^^^

**Mode 1: Full covariance specification**

Use ``add_component()`` to register each component with its auto-covariance,
then ``add_cross_covariance()`` for off-diagonal blocks:

.. code-block:: python

    from soliket.gaussian import CrossCov

    cross_cov = CrossCov()

    # Add auto-covariances
    cross_cov.add_component("mflike", mflike_cov)
    cross_cov.add_component("lensing", lensing_cov)

    # Add cross-covariance
    cross_cov.add_cross_covariance("mflike", "lensing", mflike_lensing_cov)

    # Save to SACC format
    cross_cov.save("cross_covariance.fits")

**Mode 2: Cross-covariance only**

If you only want to specify the cross-covariance (using auto-covariances
from individual likelihoods), just use ``add_cross_covariance()``:

.. code-block:: python

    cross_cov = CrossCov()
    cross_cov.add_cross_covariance("mflike", "lensing", mflike_lensing_cov)
    cross_cov.save("cross_covariance.fits")

When loaded by ``MultiGaussianLikelihood``, the auto-covariances will be
taken from each individual likelihood's SACC file.

Loading CrossCov
^^^^^^^^^^^^^^^^

To load a previously saved cross-covariance:

.. code-block:: python

    from soliket.gaussian import CrossCov

    cross_cov = CrossCov.load("cross_covariance.fits")

    # Access blocks
    mflike_lensing_block = cross_cov[("mflike", "lensing")]

Identity alignment
^^^^^^^^^^^^^^^^^^

Each component may carry per-bandpower *identity keys* -- a unique label for
every data point (e.g. ``(field, channel pair, ell)``). When present,
``MultiGaussianData`` aligns each covariance block to the data by matching these
keys instead of relying on position, so a cross-covariance keeps working even if
the data is reordered or scale-cut differently at run time. Missing or duplicate
keys raise a clear error rather than silently corrupting the joint covariance.

Identity keys are optional, supplied via ``ids`` and persisted by ``save()`` /
``load()``:

.. code-block:: python

    cross_cov.add_component("mflike", mflike_cov, ids=mflike_ids)
    cross_cov.add_component("lensing", lensing_cov, ids=lensing_ids)
    cross_cov.add_cross_covariance("mflike", "lensing", mflike_lensing_cov)

SOLikeT ``GaussianLikelihood`` subclasses populate these keys automatically from
their SACC file, and ``MultiGaussianLikelihood`` also sources them for external
components such as ``mflike`` (via ``soliket.gaussian.gaussian.bandpower_ids``).
If keys are omitted, blocks fall back to positional alignment.

GaussianData
------------

Low-level data container for named multivariate Gaussian data.

.. autoclass:: soliket.gaussian.GaussianData
    :members:
    :show-inheritance:

MultiGaussianData
-----------------

Assembles multiple ``GaussianData`` objects into a joint data vector
with combined covariance matrix.

.. autoclass:: soliket.gaussian.MultiGaussianData
    :members:
    :show-inheritance:

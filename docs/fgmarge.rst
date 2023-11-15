Foreground Marginalization (Primary CMB)
======

.. automodule:: soliket.mflike.foreground_marginalization

Performing a Foreground Marginalization
---------------------------------------

To perform a foreground marginalization, the script
``scripts/foreground_marginalization.py`` is provided. It performs a Gibbs sampling of
the CMB bandpowers and foreground parameters, where the former is estimated directly from
the data, and the latter is estimated with the Metropolis-Hastings algorithm.

To perform a foreground marginalization, simply run the script. It will write the data
to a couple of plain text files:

* ``leff.txt`` contains the l ranges of the extracted bandpower bins.
* ``samples.txt`` contains the weights, log-posterior, and values of foreground parameters
of the MH chain.
* ``progress.txt`` contains convergence statistics as the chain progresses.
* ``covmat.txt`` contains the latest estimate for the foreground parameter covariance.
* ``extracted.txt`` contains the individual samples of the extracted CMB bandpowers.

There are a variety of parameters that can be modified by the user if you so desire, or
customized if you want to apply the code to a different dataset.


Customizing the dataset
^^^^^^^^^^^^^^^^^^^^^^^

The code uses the ``soliket.ForegroundMarginalizer`` class, which is a slightly customized
version of ``soliket.MFLike`` that adds a few utility functions for the gibbs sampling. If
you wish to customize the dataset or foreground/systematics model that is used, you can
do so similar to the customization options that exist in ``MFLike``. You can provide the
``ForegroundMarginalizer`` with:

* ``bandpass`` a ``soliket.BandPass`` class that performs the bandpass integration.
* ``foregrounds`` a ``soliket.Foreground`` model that computes the foreground model.
* ``theoryforge`` a ``soliket.TheoryForge_MFLike`` class that combines the previous two
and includes instrumental systematics into the data vector.


Customizing the parameter sampling
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The parameters are split into two groups: sampled parameters and fixed parameters. By
default, the former are sampled from a multivariate gaussian distribution that is
re-estimated every few steps; the latter are kept fixed throughout the entire chain.

If you want to add your own parameter, you can do so by:

1. Adding the name of the parameter to the ``param_names`` list;
2. Adding the starting point of the parameter to the ``starting_point`` array;
3. Providing an initial proposal covariance to the ``proposal`` array;
4. You can optionally provide a hard prior range in the ``param_ranges`` array, or you
can include a log-prior function in the ``logprior()`` function.


By default, the code starts with a diagonal covariance matrix with very small step sizes,
with the initial point in a region of parameter space where we expect the chain to
converge to. This is done so that the chain initially accepts a lot of steps, and it can
better re-estimate the proposal matrix as it goes. If you want to, you can load your own
matrix into the ``proposal`` matrix, for example, by doing::

    covmat = np.loadtxt("covmat.txt")
    proposal = np.linalg.cholesky(covmat, lower = True)

The ``update_proposal_every`` parameter is the number of accepted steps in the chain
between two (attempted) updates of the proposal matrix. You can set this number to zero
if you want to disable this.


Foreground Marginalizer
-----------------------

.. autoclass:: soliket.mflike.ForegroundMarginalizer
    :exclude-members: initialize
    :members:
    :private-members:
    :show-inheritance:


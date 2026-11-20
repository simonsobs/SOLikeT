"""CAMB lensing-derivative computation feeding the cross-covariance kernels."""


def camb_lensing_derivatives(cosmo, accuracy, lmax):
    """Compute the CAMB lensed-Cl derivative w.r.t. the lensing potential.

    Wraps the ``camb.set_params`` -> ``get_results`` -> ``lensed_cl_derivatives``
    chain. ``cosmo`` and ``accuracy`` are passed straight through to
    ``camb.set_params`` (so ``cosmo`` may use either ``H0`` or ``cosmomc_theta``,
    and ``accuracy`` carries keys like ``lens_potential_accuracy``).

    Returns ``(cls, clp, dCllens)``:

    - ``cls`` -- unlensed total CMB spectra (muK^2), shape ``(lmax+1, 4)``,
    - ``clp`` -- lensing-potential power (muK), shape ``(lmax+1,)``,
    - ``dCllens`` -- ``∂ C_ell^XY / ∂ C_L^φφ``, shape ``(4, lmax+1, lmax+1)``.
    """
    import camb
    from camb.correlations import lensed_cl_derivatives

    pars = camb.set_params(lmax=lmax, **cosmo, **accuracy)
    pars.set_for_lmax(lmax)
    results = camb.get_results(pars)
    cls = results.get_unlensed_total_cls(CMB_unit="muK")[: lmax + 1, :]
    clp = results.get_lens_potential_cls(CMB_unit="muK")[: lmax + 1, 0]
    return cls, clp, lensed_cl_derivatives(cls, clp)

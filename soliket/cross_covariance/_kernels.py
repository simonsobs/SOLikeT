"""Pure cross-covariance kernels.

These functions take the CAMB lensing derivative ``dCllens`` (``∂ C_ell^XY /
∂ C_L^φφ`` from :func:`camb.correlations.lensed_cl_derivatives`) and fiducial
spectra as arrays, and return covariance blocks. They are deliberately free of
SACC / likelihood coupling so they can be unit-tested in isolation; the
extraction of windows/fsky/spectra from a model lives in the ``CrossCov``
convenience layer.

A CMB tracer combination is described by a tuple ``(ind_camb, support, weight)``:

- ``ind_camb`` -- row of ``dCllens`` for the spectrum (0=TT, 1=EE, 3=TE),
- ``support`` -- the multipoles the combination's bandpower window spans,
- ``weight`` -- the bandpower weights, shape ``(n_bins, len(support))``.
"""

import numpy as np


def cmb_lensing_block(dCllens, clp, cl_kk, fsky, cmb_combs, kk_binning):
    """Cross-covariance between binned CMB spectra and binned ``C_L^kk``.

    Parameters
    ----------
    dCllens : ndarray, shape (n_spec, lmax+1, lmax+1)
        CAMB lensed-Cl derivative w.r.t. the lensing-potential power.
    clp : ndarray
        Fiducial lensing-potential power ``C_L^φφ`` (indexed by multipole).
    cl_kk : ndarray, shape (lmax_kk,)
        Fiducial convergence power ``C_L^kk`` on the kappa side.
    fsky : float
        Sky fraction.
    cmb_combs : list of (int, ndarray, ndarray)
        One ``(ind_camb, support, weight)`` per CMB tracer combination.
    kk_binning : ndarray, shape (n_kk_bins, lmax_kk)
        Binning matrix on the kappa side.

    Returns
    -------
    ndarray, shape (n_cmb_data, n_kk_bins)
    """
    lmax_kk = kk_binning.shape[1]
    ell_kk = np.arange(lmax_kk)
    cl_kk = np.asarray(cl_kk)
    kk_weight = 2.0 * cl_kk**2 / (2 * ell_kk + 1) / fsky

    n_cmb = sum(weight.shape[0] for _, _, weight in cmb_combs)
    out = np.zeros((n_cmb, kk_binning.shape[0]))

    row = 0
    for ind_camb, support, weight in cmb_combs:
        deriv = dCllens[ind_camb][np.asarray(support)][:, :lmax_kk]
        xcov = (2.0 / np.pi) * deriv / clp[:lmax_kk] * kk_weight
        xcov[:, 0:2] = 0.0
        out[row : row + weight.shape[0], :] = weight @ (xcov @ kk_binning.T)
        row += weight.shape[0]
    return out


def lensing_induced_block(dCllens, fsky, cmb_combs):
    """Lensing-induced covariance mixing the CMB spectra among themselves.

    Captures the covariance the lensing potential induces between (binned) CMB
    bandpowers across tracer combinations. Returns the symmetric joint matrix.

    Parameters
    ----------
    dCllens : ndarray, shape (n_spec, lmax+1, lmax+1)
        CAMB lensed-Cl derivative w.r.t. the lensing-potential power.
    fsky : float
        Sky fraction.
    cmb_combs : list of (int, ndarray, ndarray)
        One ``(ind_camb, support, weight)`` per CMB tracer combination.

    Returns
    -------
    ndarray, shape (n_cmb_data, n_cmb_data)
    """
    lmax = dCllens.shape[1]
    ell = np.arange(lmax)
    factor = 2.0 / (2 * ell + 1) / fsky

    # Bandpower-weighted derivatives per combination, shape (n_bins, lmax).
    binned_deriv = [
        weight @ dCllens[ind_camb][np.asarray(support)]
        for ind_camb, support, weight in cmb_combs
    ]
    offsets = np.cumsum([0] + [a.shape[0] for a in binned_deriv])
    out = np.zeros((offsets[-1], offsets[-1]))

    for i, a_i in enumerate(binned_deriv):
        for j in range(i, len(binned_deriv)):
            a_j = binned_deriv[j]
            block = (a_i[:, None, :] * a_j[None, :, :] * factor).sum(axis=2)
            out[offsets[i] : offsets[i + 1], offsets[j] : offsets[j + 1]] = block

    return np.triu(out, 0) + np.triu(out, 1).T


def shear_kappa_block(dCllens, clp, lss_spectra, lss_binnings, fsky, cmb_combs):
    """Cross-covariance between binned CMB spectra and binned shear/galaxy-kappa.

    Each LSS tracer contributes the same kernel as :func:`cmb_lensing_block` with
    its own theory spectrum and bandpower windows; the per-tracer blocks are
    concatenated along the LSS axis.

    Parameters
    ----------
    dCllens : ndarray, shape (n_spec, lmax+1, lmax+1)
        CAMB lensed-Cl derivative w.r.t. the lensing-potential power.
    clp : ndarray
        Fiducial lensing-potential power.
    lss_spectra : list of ndarray
        Unbinned theory ``C_ell^{shear x kappa}`` per LSS tracer.
    lss_binnings : list of ndarray
        Bandpower-window matrix per LSS tracer, shape ``(n_bins, len(spectrum))``.
    fsky : float
        Sky fraction.
    cmb_combs : list of (int, ndarray, ndarray)
        One ``(ind_camb, support, weight)`` per CMB tracer combination.

    Returns
    -------
    ndarray, shape (n_cmb_data, sum_of_lss_bins)
    """
    blocks = [
        cmb_lensing_block(dCllens, clp, spectrum, fsky, cmb_combs, binning)
        for spectrum, binning in zip(lss_spectra, lss_binnings)
    ]
    return np.hstack(blocks)


def n1_crosscov_block(dCllens, clp, n1_normed_mat, fsky, cmb_combs, kk_binning):
    """N1-bias contribution to the CMB-spectra x ``C_L^kk`` cross-covariance.

    Applies a precomputed, normalised N1 transfer matrix (produced externally via
    ``lensitbiases``; see the create_cross_covariance notebook) to the lensing
    derivative. Pure given ``n1_normed_mat`` -- the lensitbiases dependency lives
    only in generating that matrix.

    This is a **diagnostic**, not part of the assembled covariance: it sizes the N1
    contribution to the kappa cross-block, and no covariance builder (including
    :meth:`CrossCov.from_cmb_lensing`) calls it. The dev notebook computes, saves and
    plots it, then stops there.

    Parameters
    ----------
    dCllens : ndarray, shape (n_spec, lmax+1, lmax+1)
        CAMB lensed-Cl derivative w.r.t. the lensing-potential power.
    clp : ndarray
        Fiducial lensing-potential power.
    n1_normed_mat : ndarray, shape (>= lmax_kk, n_ell)
        Normalised, smoothed N1 transfer matrix. **Rows must be indexed by lensing
        multipole from L=0**, so that row ``L`` aligns with column ``L`` of
        ``kk_binning``; only the first ``lmax_kk`` rows are read. Note the
        lensitbiases recipe in the notebook builds its rows over
        ``Ls_n1 = arange(lminbox, ...)`` (``lminbox=20``), i.e. row ``i`` is
        ``L = lminbox + i`` -- such a matrix must be zero-padded up to L=0 before it
        is passed here, or every row lands ``lminbox`` multipoles off.
    fsky : float
        Sky fraction.
    cmb_combs : list of (int, ndarray, ndarray)
        One ``(ind_camb, support, weight)`` per CMB tracer combination.
    kk_binning : ndarray, shape (n_kk_bins, lmax_kk)
        Binning matrix on the kappa side.

    Returns
    -------
    ndarray, shape (n_cmb_data, n_kk_bins)
    """
    n_ell = n1_normed_mat.shape[1]
    ell = np.arange(n_ell)
    factor = 2 * clp[:n_ell] / (2 * ell + 1) / fsky
    factor[1:] *= 2 * np.pi / (ell[1:] * (ell[1:] + 1)) ** 2

    lmax_kk = kk_binning.shape[1]
    binned_n1 = kk_binning @ n1_normed_mat[:lmax_kk, :] * np.pi / 2.0

    n_cmb = sum(weight.shape[0] for _, _, weight in cmb_combs)
    out = np.zeros((n_cmb, kk_binning.shape[0]))
    row = 0
    for ind_camb, support, weight in cmb_combs:
        a1 = weight @ dCllens[ind_camb][np.asarray(support)]
        product = a1[:, None, :n_ell] * binned_n1[None, :, :] * factor
        out[row : row + weight.shape[0], :] = product.sum(axis=2)
        row += weight.shape[0]
    return out

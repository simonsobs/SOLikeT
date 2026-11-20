"""Extract kernel inputs from a model and assemble cross-covariance blocks.

This is the convenience layer: it pulls fsky, the fiducial cosmology/accuracy,
the CMB bandpower windows and the kappa-side binning from a likelihood/Session,
runs :func:`camb_lensing_derivatives`, and calls the pure kernels. The physics
lives in ``_kernels``; this module only does the wiring.
"""

import ast

import numpy as np

from ._derivatives import camb_lensing_derivatives
from ._kernels import cmb_lensing_block, lensing_induced_block, shear_kappa_block


def _cosmo_camb_kwargs(cosmo):
    """Map a SACC-metadata cosmology dict to ``camb.set_params`` keywords."""
    return {
        "cosmomc_theta": cosmo["cosmomc_theta"],
        "As": 1e-10 * np.exp(cosmo["logA"]),
        "ombh2": cosmo["ombh2"],
        "omch2": cosmo["omch2"],
        "ns": cosmo["ns"],
        "Alens": cosmo["Alens"],
        "tau": cosmo["tau"],
    }


# spec_meta polarisation -> CAMB lensed-Cl derivative row (TT=0, EE=1, BB=2, TE=3).
_POL_TO_CAMB = {"tt": 0, "ee": 1, "bb": 2, "te": 3}


def cmb_combs_from_spec_meta(spec_meta):
    """Per spectrum, the ``(ind_camb, support, weight)`` triple, in MFLike's own
    data-vector (auto-covariance) order.

    Driven by ``mflike.spec_meta`` -- the *same* per-spectrum windows and ordering
    MFLike uses to build its auto-covariance (and the smooth-data binner) -- so a
    cross-covariance block built from these is aligned with the auto-covariance
    **by construction**: no reliance on the SACC's tracer-combination order, robust
    to a reordered cov_Bbl file or TE/ET symmetrization (the lensed-Cl derivative is
    per spectrum type, so the TE window is the right row even when ET is folded in).
    The rows are already scale-cut, so no further trimming is needed.
    """
    return [
        (_POL_TO_CAMB[m["pol"]], np.asarray(m["bpw"].values), m["bpw"].weight.T)
        for m in spec_meta
    ]


def cmb_lensing_crosscov(
    mflike_sacc,
    lensing,
    combs,
    *,
    fsky=None,
    cosmo=None,
    accuracy=None,
    lmax=None,
    derivatives=None,
):
    """Compute the CMB-primary x CMB-lensing cross-covariance block.

    Low-level entry point. ``mflike_sacc`` is the MFLike SACC (with covariance
    metadata); ``lensing`` is an evaluated ``LensingLikelihood``. ``combs`` are the
    per-spectrum ``(ind_camb, support, weight)`` triples for the CMB rows -- build
    them with :func:`cmb_combs_from_spec_meta`, which keeps the block rows in
    MFLike's own data-vector order. ``fsky``, ``cosmo``, ``accuracy`` and ``lmax``
    default to the values stored in the MFLike SACC metadata and may be overridden.
    ``derivatives`` optionally supplies a precomputed ``camb_lensing_derivatives``
    bundle (see :func:`camb_lensing_derivatives_from_sacc`) so a joint covariance
    shares one CAMB run across blocks; when omitted it is computed here.
    """
    fsky, cosmo, accuracy, lmax = _resolve_inputs(
        mflike_sacc.metadata, fsky, cosmo, accuracy, lmax
    )
    _, clp, dCllens = (
        derivatives
        if derivatives is not None
        else camb_lensing_derivatives(cosmo, accuracy, lmax)
    )

    lmax_kk = lensing.binning_matrix.shape[1]
    cl_kk = np.pi / 2 * lensing.provider.get_Cl(ell_factor=True)["pp"][:lmax_kk]
    return cmb_lensing_block(dCllens, clp, cl_kk, fsky, combs, lensing.binning_matrix)


def _resolve_inputs(md, fsky, cosmo, accuracy, lmax):
    """Fill fsky/cosmo/accuracy/lmax from MFLike SACC metadata where not given."""

    def meta(key):
        if key not in md:
            raise KeyError(
                f"MFLike SACC metadata has no {key!r}; pass the corresponding "
                "argument to the cross-covariance call explicitly."
            )
        return md[key]

    if fsky is None:
        fsky = float(meta("f_sky_LAT"))
    if accuracy is None:
        accuracy = ast.literal_eval(meta("accuracy_params"))
    if lmax is None:
        lmax = int(meta("lmax")) + 1
    if cosmo is None:
        cosmo = _cosmo_camb_kwargs(ast.literal_eval(meta("cosmo_params")))
    return fsky, cosmo, accuracy, lmax


def camb_lensing_derivatives_from_sacc(
    mflike_sacc, *, cosmo=None, accuracy=None, lmax=None
):
    """CAMB lensed-Cl derivative bundle for an MFLike SACC, computed once.

    Resolves ``cosmo``/``accuracy``/``lmax`` from the MFLike SACC metadata (each
    overridable) and runs CAMB a single time. Pass the returned ``(cls, clp,
    dCllens)`` bundle as ``derivatives=`` to :func:`cmb_lensing_crosscov`,
    :func:`lensing_induced_cov` and :func:`shear_kappa_crosscov` so a joint
    covariance shares one CAMB run instead of recomputing the (expensive)
    derivative per block.
    """
    _, cosmo, accuracy, lmax = _resolve_inputs(
        mflike_sacc.metadata, None, cosmo, accuracy, lmax
    )
    return camb_lensing_derivatives(cosmo, accuracy, lmax)


def lensing_induced_cov(
    mflike_sacc,
    combs,
    *,
    fsky=None,
    cosmo=None,
    accuracy=None,
    lmax=None,
    derivatives=None,
):
    """Compute the lensing-induced covariance within the MFLike CMB block.

    Low-level entry point mirroring :func:`cmb_lensing_crosscov`; ``combs`` are the
    per-spectrum triples (see :func:`cmb_combs_from_spec_meta`), and ``fsky``,
    ``cosmo``, ``accuracy`` and ``lmax`` default to the MFLike SACC metadata.
    ``derivatives`` optionally supplies a precomputed ``camb_lensing_derivatives``
    bundle to share one CAMB run across blocks. Returns the symmetric
    ``(n_cmb_data, n_cmb_data)`` matrix.
    """
    fsky, cosmo, accuracy, lmax = _resolve_inputs(
        mflike_sacc.metadata, fsky, cosmo, accuracy, lmax
    )
    _, _, dCllens = (
        derivatives
        if derivatives is not None
        else camb_lensing_derivatives(cosmo, accuracy, lmax)
    )
    return lensing_induced_block(dCllens, fsky, combs)


def shear_kappa_limber(shearkappa_like, params_values):
    """Unbinned shear x CMB-lensing spectra and bandpower windows per LSS tracer.

    Thin wrapper over the likelihood's own theory: ``get_unbinned_theory`` returns
    the Limber spectra (including IA, redshift- and multiplicative-bias nuisance
    handling) and ``get_binning`` the bandpower windows. Returns
    ``(cl_unbinned_list, w_bins_list)``.
    """
    sklike = shearkappa_like
    cl_unbinned_list = sklike.get_unbinned_theory(**params_values)
    w_bins_list = [
        sklike.get_binning(comb)[1] for comb in sklike.sacc_data.get_tracer_combinations()
    ]
    return cl_unbinned_list, w_bins_list


def shear_kappa_crosscov(
    mflike_sacc,
    shearkappa_like,
    params_values,
    combs,
    *,
    fsky=None,
    cosmo=None,
    accuracy=None,
    lmax=None,
    derivatives=None,
):
    """Compute the CMB-primary x shear/galaxy-kappa cross-covariance block.

    Low-level entry point; ``fsky``/``cosmo``/``accuracy``/``lmax`` default to the
    MFLike SACC metadata. ``params_values`` are the nuisance parameters the LSS
    Limber spectra depend on. ``combs`` are the per-spectrum triples for the CMB
    (row) side -- see :func:`cmb_combs_from_spec_meta`. ``derivatives`` optionally
    supplies a precomputed ``camb_lensing_derivatives`` bundle to share one CAMB run
    across blocks.
    """
    fsky, cosmo, accuracy, lmax = _resolve_inputs(
        mflike_sacc.metadata, fsky, cosmo, accuracy, lmax
    )
    _, clp, dCllens = (
        derivatives
        if derivatives is not None
        else camb_lensing_derivatives(cosmo, accuracy, lmax)
    )
    cl_list, w_list = shear_kappa_limber(shearkappa_like, params_values)
    return shear_kappa_block(dCllens, clp, cl_list, w_list, fsky, combs)

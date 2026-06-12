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


# SACC CMB data type -> CAMB lensed_cl_derivatives row (0=TT, 1=EE, 2=BB, 3=TE).
_CAMB_SPECTRUM_ROW = {"cl_00": 0, "cl_ee": 1, "cl_bb": 2, "cl_0e": 3}


def _camb_spectrum_index(sacc_data, tracer_comb):
    """Row of the CAMB lensed-Cl derivative for a CMB tracer pair (TT=0/EE=1/BB=2/TE=3).

    Resolved from the SACC ``data_type`` rather than the tracer-name spin suffix:
    EE and BB share the ``_s2 x _s2`` tracer pair and are indistinguishable by name.
    The rest of this module assumes one spectrum per tracer pair, so a pair carrying
    several data types -- or a type CAMB has no lensed-Cl derivative row for (e.g.
    TB/EB) -- is rejected here rather than silently mapped to the wrong row.
    """
    dtypes = sacc_data.get_data_types(tracers=tracer_comb)
    if len(dtypes) != 1:
        raise ValueError(
            f"tracer pair {tracer_comb} carries data types {dtypes}; the "
            "cross-covariance assumes exactly one CMB spectrum per tracer pair."
        )
    dtype = dtypes[0]
    if dtype not in _CAMB_SPECTRUM_ROW:
        raise ValueError(
            f"data type {dtype!r} for tracers {tracer_comb} has no CAMB lensed-Cl "
            "derivative row (supported: TT=cl_00, EE=cl_ee, BB=cl_bb, TE=cl_0e)."
        )
    return _CAMB_SPECTRUM_ROW[dtype]


def cmb_combs_from_sacc(sacc_data):
    """Per CMB tracer combination, the ``(ind_camb, support, weight)`` triple.

    The bandpower windows are fetched the same way every likelihood does
    (``indices(tracers=comb)``); only the CAMB-spectrum row is cross-covariance
    specific. For MFLike each tracer pair carries a single spectrum, so the
    tracer-only window lookup is unambiguous.
    """
    combs = []
    flat = []
    for comb in sacc_data.get_tracer_combinations():
        idx = np.asarray(sacc_data.indices(tracers=comb))
        flat.append(idx)
        bpw = sacc_data.get_bandpower_windows(idx)
        combs.append(
            (_camb_spectrum_index(sacc_data, comb), np.asarray(bpw.values), bpw.weight.T)
        )
    # The block rows below are laid out in this per-combination order, but the
    # cross-covariance is later trimmed with mflike's mask in the SACC's NATURAL
    # flat order. The two coincide only when each spectrum is stored contiguously;
    # guard it so a reordered SACC fails loudly instead of silently mis-ordering
    # the cross-covariance rows. (The auto-cov side is checked in from_cmb_lensing.)
    flat = np.concatenate(flat) if flat else np.zeros(0, dtype=int)
    if not np.array_equal(flat, np.arange(flat.size)):
        raise ValueError(
            "MFLike SACC bandpowers are not contiguous in tracer-combination order; "
            "the cross-covariance block rows would not match the SACC's natural "
            "bandpower order. Re-save the SACC with each spectrum stored contiguously."
        )
    return combs


def bandpower_ell_natural(sacc_data):
    """Effective multipole of every SACC bandpower, in the SACC's natural flat order.

    Used to verify, by bandpower identity, that a cross-covariance block (laid out
    in natural order) lines up with a likelihood's auto-covariance before trimming.
    """
    ell = np.empty(sacc_data.mean.size)
    for comb in sacc_data.get_tracer_combinations():
        dtype = sacc_data.get_data_types(tracers=comb)[0]
        comb_ell, _, ind = sacc_data.get_ell_cl(dtype, *comb, return_ind=True)
        ell[ind] = comb_ell
    return ell


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
    mflike_sacc, lensing, *, fsky=None, cosmo=None, accuracy=None, lmax=None,
    derivatives=None, combs=None,
):
    """Compute the CMB-primary x CMB-lensing cross-covariance block.

    Low-level entry point. ``mflike_sacc`` is the MFLike SACC (with covariance
    metadata); ``lensing`` is an evaluated ``LensingLikelihood``. ``fsky``,
    ``cosmo``, ``accuracy`` and ``lmax`` default to the values stored in the
    MFLike SACC metadata and may be overridden. ``derivatives`` optionally supplies
    a precomputed ``camb_lensing_derivatives`` bundle (see
    :func:`camb_lensing_derivatives_from_sacc`) so a joint covariance shares one
    CAMB run across blocks; when omitted it is computed here. ``combs`` optionally
    supplies the per-spectrum ``(ind_camb, support, weight)`` triples -- pass
    :func:`cmb_combs_from_spec_meta` so the block rows match the MFLike
    auto-covariance order; by default they are derived from the SACC's tracer
    combinations (full, natural order).
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
    if combs is None:
        combs = cmb_combs_from_sacc(mflike_sacc)
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
    *,
    fsky=None,
    cosmo=None,
    accuracy=None,
    lmax=None,
    derivatives=None,
    combs=None,
):
    """Compute the lensing-induced covariance within the MFLike CMB block.

    Low-level entry point mirroring :func:`cmb_lensing_crosscov`; ``fsky``,
    ``cosmo``, ``accuracy`` and ``lmax`` default to the MFLike SACC metadata.
    ``derivatives`` optionally supplies a precomputed ``camb_lensing_derivatives``
    bundle to share one CAMB run across blocks. Returns the symmetric
    ``(n_cmb_data, n_cmb_data)`` matrix. ``combs`` optionally supplies the
    per-spectrum triples -- pass :func:`cmb_combs_from_spec_meta` so this block
    matches the MFLike auto-covariance order it is added to; by default they are
    the SACC's tracer combinations (full, natural order).
    """
    fsky, cosmo, accuracy, lmax = _resolve_inputs(
        mflike_sacc.metadata, fsky, cosmo, accuracy, lmax
    )
    _, _, dCllens = (
        derivatives
        if derivatives is not None
        else camb_lensing_derivatives(cosmo, accuracy, lmax)
    )
    if combs is None:
        combs = cmb_combs_from_sacc(mflike_sacc)
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
        sklike.get_binning(comb)[1]
        for comb in sklike.sacc_data.get_tracer_combinations()
    ]
    return cl_unbinned_list, w_bins_list


def shear_kappa_crosscov(
    mflike_sacc,
    shearkappa_like,
    params_values,
    *,
    fsky=None,
    cosmo=None,
    accuracy=None,
    lmax=None,
    derivatives=None,
    combs=None,
):
    """Compute the CMB-primary x shear/galaxy-kappa cross-covariance block.

    Low-level entry point; ``fsky``/``cosmo``/``accuracy``/``lmax`` default to the
    MFLike SACC metadata. ``params_values`` are the nuisance parameters the LSS
    Limber spectra depend on. ``derivatives`` optionally supplies a precomputed
    ``camb_lensing_derivatives`` bundle to share one CAMB run across blocks.
    ``combs`` optionally supplies the per-spectrum triples for the CMB (row) side --
    pass :func:`cmb_combs_from_spec_meta` so the rows match the MFLike
    auto-covariance order; by default they are the SACC's tracer combinations.
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
    if combs is None:
        combs = cmb_combs_from_sacc(mflike_sacc)
    return shear_kappa_block(dCllens, clp, cl_list, w_list, fsky, combs)

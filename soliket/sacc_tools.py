"""Dataset-creation utilities distilled from the simulation notebooks.

General, likelihood-agnostic helpers for building SACC datasets: top-hat
bandpower windows and the Knox (Gaussian) covariance for an arbitrary set of
tracers. The per-tracer ``add_tracer``/``add_ell_cl`` calls stay in the notebooks
-- they are thin SACC-API calls that differ by tracer type and gain nothing from
wrapping.
"""

import numpy as np


def top_hat_windows(ell_max, n_bins):
    """Uniform top-hat bandpower windows partitioning ``[0, ell_max]``.

    Returns ``(ells, window)`` where ``ells`` are the bin-centre multipoles and
    ``window`` is a :class:`sacc.BandpowerWindow` with one unit-weight top-hat per
    bin. The bins partition every multipole in ``0..ell_max`` inclusive with no
    gaps; when ``ell_max + 1`` is not divisible by ``n_bins`` the bin widths differ
    by at most one multipole rather than leaving the high-ell tail unbinned.
    """
    import sacc

    ells_win = np.arange(ell_max + 1)
    segments = np.array_split(ells_win, n_bins)
    ells = np.array([seg.mean() for seg in segments])
    weights = np.zeros((n_bins, len(ells_win)))
    for i, seg in enumerate(segments):
        weights[i, seg] = 1.0
    return ells, sacc.BandpowerWindow(ells_win, weights.T)


def gaussian_covariance(cls, ells, delta_ell, fsky):
    """Knox (Gaussian) covariance of the auto/cross bandpowers of N maps.

    Parameters
    ----------
    cls : ndarray, shape (n_maps, n_maps, n_ell)
        Auto- and cross-spectra (including any noise on the autos), symmetric in
        the first two axes.
    ells : ndarray, shape (n_ell,)
        Bin-centre multipoles.
    delta_ell : float
        Bin width.
    fsky : float
        Sky fraction.

    Returns
    -------
    ndarray, shape (n_cross * n_ell, n_cross * n_ell)
        Joint covariance over the ``n_cross = n_maps (n_maps + 1) / 2`` unique
        spectra, ordered as ``(0,0), (0,1), ..., (1,1), ...``.
    """
    cls = np.asarray(cls)
    n_maps = cls.shape[0]
    n_ell = len(ells)
    pairs = [(i, j) for i in range(n_maps) for j in range(i, n_maps)]
    n_cross = len(pairs)

    covar = np.zeros((n_cross, n_ell, n_cross, n_ell))
    knox_norm = delta_ell * fsky * (2 * np.asarray(ells) + 1)
    for id_i, (i1, i2) in enumerate(pairs):
        for id_j, (j1, j2) in enumerate(pairs):
            cov = (cls[i1, j1] * cls[i2, j2] + cls[i1, j2] * cls[i2, j1]) / knox_norm
            covar[id_i, :, id_j, :] = np.diag(cov)
    return covar.reshape(n_cross * n_ell, n_cross * n_ell)


def smooth_twin_sacc(src, data_type, tracer1, tracer2, theory, *, out_path=None):
    """Smooth (theory) twin of a single-spectrum SACC.

    Reuses the tracers, bandpower windows and covariance of ``src`` for the
    ``(data_type, tracer1, tracer2)`` spectrum but replaces the measured bandpowers
    with ``theory``. Evaluating a likelihood on the result at the cosmology
    ``theory`` was computed for gives chi^2 = 0 -- the standard way to make a
    noiseless twin of a real dataset (e.g. a CMB-lensing reconstruction). If
    ``out_path`` is given the SACC is saved there. Returns the :class:`sacc.Sacc`.
    """
    import sacc

    ell, _, cov, ind = src.get_ell_cl(
        data_type, tracer1, tracer2, return_cov=True, return_ind=True
    )
    windows = src.get_bandpower_windows(ind)

    out = sacc.Sacc()
    for name in dict.fromkeys((tracer1, tracer2)):  # de-dup, keep order
        out.add_tracer_object(src.tracers[name])
    out.add_ell_cl(data_type, tracer1, tracer2, ell, np.asarray(theory), window=windows)
    out.add_covariance(cov)

    if out_path is not None:
        out.save_fits(str(out_path), overwrite=True)
    return out


# Multipole/spin spelling MFLike uses in SACC: T is a spin-0 (s0) map, E/B are
# spin-2 (s2); the data_type code spells each field as 0 (T), e (E), b (B).
_MAP_TYPE = {"T": "0", "E": "e", "B": "b"}
_POLS = ("T", "E", "B")


def _bin_modified_theory(mflike, dls, fg_totals, params):
    """Bin the CMB+foreground+systematics theory through MFLike's bandpower windows.

    Returns ``(ps_dic, ps_vec)``: ``ps_dic[f"{t1}x{t2}"][pol]`` holds the binned
    bandpowers per frequency pair and polarisation (``"tt"``/``"te"``/``"ee"``)
    plus ``"lbin"``; ``ps_vec`` is the flat data vector in MFLike's own ordering.
    """
    dls_cut = {s: dls[s][mflike.l_bpws] for s in mflike.lcuts}
    obs = mflike.get_modified_theory(dls_cut, fg_totals, **params)

    ps_vec = np.zeros_like(mflike.data_vec)
    ps_dic = {}
    for m in mflike.spec_meta:
        pol, ids, window = m["pol"], m["ids"], m["bpw"]
        key = f"{m['t1']}x{m['t2']}"
        ps_dic.setdefault(key, {"lbin": m["leff"]})
        t1, t2 = (m["t2"], m["t1"]) if m["hasYX_xsp"] else (m["t1"], m["t2"])
        spec = obs[pol, t1, t2]
        for i, nonzero, weights in zip(ids, window.nonzeros, window.sliced_weights):
            ps_vec[i] = weights @ spec[nonzero]
        ps_dic[key][pol] = ps_vec[ids]
    return ps_dic, ps_vec


def smooth_mflike_sacc(mflike, dls, fg_totals, params, *, out_path=None, beam_lmax=10000):
    """Build a smooth (theory) MFLike data SACC and return it.

    Bins the CMB + foreground + systematics theory through MFLike's own bandpower
    windows and writes one ``NuMap`` tracer per ``(frequency, spin)`` channel plus
    an ``add_ell_cl`` per cross-spectrum -- the ``input_file`` format MFLike reads.
    Evaluating the likelihood on it at the same fiducial gives chi^2 = 0.

    The intricate per-frequency plumbing is MFLike-specific, so this stays a helper
    rather than living in the notebook; it takes the concrete handles (not a
    ``Session``) so the caller keeps the model in view:

    Parameters
    ----------
    mflike : an evaluated MFLike likelihood (e.g. ``resolve_aliases(model).mflike``),
        exposing ``spec_meta``, ``bands``, ``l_bpws``, ``lcuts``,
        ``get_modified_theory`` and ``data_vec``.
    dls : the lensed CMB :math:`D_\\ell` dict, ``model.provider.get_Cl(ell_factor=True)``.
    fg_totals : the foreground bandpowers, the foreground theory's ``get_fg_totals()``.
    params : flat ``{name: value}`` of the cosmo + foreground + systematics fiducial,
        forwarded to ``get_modified_theory`` for calibrations and bandpass shifts.
    out_path : if given, save the SACC there as FITS.
    beam_lmax : length of the unit beam attached to each tracer.

    Notes
    -----
    The covariance and bandpower-window (Bbl) matrices are *not* written here;
    MFLike reads them separately via ``cov_Bbl_file``, so reuse the shipped one.
    Cross-frequency ``ET`` reuses the pair's ``TE`` and any B-mode spectrum is
    written as zeros; MFLike selects only the requested TT/TE/ET/EE within its
    scale cuts, so these extras are ignored.
    """
    import sacc

    ps_dic, _ = _bin_modified_theory(mflike, dls, fg_totals, params)
    freqs = sorted(
        {t for key in ps_dic for t in key.split("x")},
        key=lambda name: int(name.split("_")[1]),
    )

    s = sacc.Sacc()
    beam = {"ell": np.arange(beam_lmax), "beam": np.ones(beam_lmax)}
    for freq in freqs:
        for spin, quantity in (("s0", "cmb_temperature"), ("s2", "cmb_polarization")):
            band = mflike.bands[f"{freq}_{spin}"]
            s.add_tracer(
                "NuMap", f"{freq}_{spin}", quantity=quantity,
                spin=0 if spin == "s0" else 2,
                nu=band["nu"], bandpass=band["bandpass"], **beam,
            )

    for ia, fa in enumerate(freqs):
        for ib, fb in enumerate(freqs):
            if ia > ib:
                continue
            for ipa, pa in enumerate(_POLS):
                for pb in (_POLS[ipa:] if fa == fb else _POLS):
                    ta = f"{fa}_s0" if pa == "T" else f"{fa}_s2"
                    tb = f"{fb}_s0" if pb == "T" else f"{fb}_s2"
                    cl_type = "cl_" + (
                        _MAP_TYPE[pb] + _MAP_TYPE[pa] if pb == "T"
                        else _MAP_TYPE[pa] + _MAP_TYPE[pb]
                    )
                    pair = ps_dic[f"{fa}x{fb}"]
                    lbin = pair["lbin"]
                    values = pair.get(
                        (pa + pb).lower(),
                        pair.get((pb + pa).lower(), np.zeros(len(lbin))),
                    )
                    s.add_ell_cl(cl_type, ta, tb, lbin, values)

    if out_path is not None:
        s.save_fits(str(out_path), overwrite=True)
    return s

"""Dataset-creation utilities distilled from the simulation notebooks.

General, likelihood-agnostic helpers for building SACC datasets: top-hat
bandpower windows and the Knox (Gaussian) covariance for an arbitrary set of
tracers. The per-tracer ``add_tracer``/``add_ell_cl`` calls stay in the notebooks
-- they are thin SACC-API calls that differ by tracer type and gain nothing from
wrapping.
"""

import numpy as np


def top_hat_windows(ell_max, n_bins):
    """Normalised top-hat bandpower windows partitioning ``[0, ell_max]``.

    Returns ``(ells, window)`` where ``ells`` are the bin-centre multipoles and
    ``window`` is a :class:`sacc.BandpowerWindow` whose per-bin weights **sum to
    one**, so binning a spectrum through it gives the bin *mean*. The bins partition
    every multipole in ``0..ell_max`` inclusive with no gaps; when ``ell_max + 1`` is
    not divisible by ``n_bins`` the bin widths differ by at most one multipole rather
    than leaving the high-ell tail unbinned.

    The normalisation matters: a likelihood bins theory as ``w_bins @ cl`` (a plain
    contraction, no implicit averaging), so the data stored against this window must
    be in the same units the window produces. Mean-normalised weights let a dataset
    store ``C_ell`` at the bin centres -- which is what the simulation notebooks
    compute and plot against unbinned theory. Real datasets whose windows sum to
    ``delta_ell`` instead store bandpower *sums*; both are self-consistent, and the
    likelihood is agnostic as long as ``data == w_bins @ cl_true``.
    """
    import sacc

    ells_win = np.arange(ell_max + 1)
    segments = np.array_split(ells_win, n_bins)
    ells = np.array([seg.mean() for seg in segments])
    weights = np.zeros((n_bins, len(ells_win)))
    for i, seg in enumerate(segments):
        weights[i, seg] = 1.0 / len(seg)
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
    delta_ell : float or ndarray, shape (n_ell,)
        Bin width, broadcast against ``ells``. Pass a per-bin array when the bins
        are not uniform -- Knox variance goes as ``1 / delta_ell``, so a single bin
        one multipole wider than the rest is a real (few-percent) error, not a wash.
        Reading the widths off the bandpower window keeps them honest:
        ``(window.weight.T != 0).sum(axis=1)``.
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


def build_lensing_sacc(ell, bandpowers, cov, *, windows=None, tracer_name="ck",
                       beam_lmax=3000, out_path=None):
    """Assemble a CMB-lensing (``ck x ck``) data SACC from binned bandpowers.

    Stores ``bandpowers`` (already binned to the bin-centre multipoles ``ell``) as
    the ``cl_00`` spectrum of a single ``cmb_convergence`` tracer, with covariance
    ``cov`` and optional bandpower ``windows``. This is the ``data_filename`` the
    lensing likelihoods read -- the "bring your own bandpowers" entry point.

    The caller bins the theory (``w_bins @ clkk``); for the *full*
    ``LensingLikelihood``, which compares against binned-theory *plus* the N0/N1
    correction, add that correction to ``bandpowers`` before calling (e.g. via
    :meth:`~soliket.lensing._corrections.LensingCorrections.compute`) so the stored
    vector matches what the likelihood recomputes and chi^2 = 0 at the imprint
    cosmology. The bare binned spectrum (no correction) is the ``LensingLite``
    target. Returns the :class:`sacc.Sacc`.
    """
    import sacc

    s = sacc.Sacc()
    s.add_tracer(
        "Map", tracer_name, quantity="cmb_convergence", spin=0,
        ell=np.arange(beam_lmax), beam=np.ones(beam_lmax),
    )
    s.add_ell_cl("cl_00", tracer_name, tracer_name, ell, np.asarray(bandpowers),
                 window=windows)
    s.add_covariance(np.asarray(cov))
    if out_path is not None:
        s.save_fits(str(out_path), overwrite=True)
    return s


# Tracer/data-type layout the full LensingLikelihood reads its fiducial spectra and
# N0/N1 correction matrices from (mirrors LensingLikelihood._set_fiducial_Cls and
# LensingCorrections.from_sacc). Kept local so sacc_tools stays import-light.
_LENS_FID = (  # (key, tracer1, tracer2, data_type) for the fiducial spectra
    ("tt", "ct", "ct", "cl_00"),
    ("te", "ct", "ce", "cl_0e"),
    ("ee", "ce", "ce", "cl_ee"),
    ("bb", "cb", "cb", "cl_bb"),
    ("kk", "ck", "ck", "cl_00"),
)
_LENS_N0 = {"tt": ("ct", "ct", "N0_00"), "te": ("ct", "ce", "N0_0e"),
            "ee": ("ce", "ce", "N0_ee"), "bb": ("cb", "cb", "N0_bb")}
_LENS_N1 = {"tt": ("ct", "ct", "N1_00"), "te": ("ct", "ce", "N1_0e"),
            "ee": ("ce", "ce", "N1_ee"), "bb": ("cb", "cb", "N1_bb")}
_LENS_Q = {"ct": "cmb_temperature", "ce": "cmb_polarization",
           "cb": "cmb_polarization", "cp": "cmb_lens_potential",
           "ck": "cmb_convergence", "n0": "cmb_convergence"}


def build_lensing_corrections_sacc(*, fiducial, n0_response, n1_response, n1_clpp,
                                   n0, fiducial_out=None, corrections_out=None):
    """Write the fiducial + N0/N1 correction SACC files the full LensingLikelihood loads.

    The inverse of ``LensingLikelihood._set_fiducial_Cls`` /
    ``LensingCorrections.from_sacc``: lays a user's own estimator inputs into the
    exact tracer/data-type layout those readers expect, so the full likelihood can
    be pointed at custom ``fiducial_filename`` / ``correction_filename``. Use when
    you have your own reconstruction's biases (e.g. from ``so-lenspipe``); most
    users bringing only a spectrum want the correction-free ``LensingLite`` instead.

    Parameters
    ----------
    fiducial : dict
        Unbinned fiducial spectra to ``lmax``, keyed ``"tt"/"te"/"ee"/"bb"/"kk"``.
    n0_response, n1_response : dict
        ``(lmax, lmax)`` response matrices per spectrum (``"tt"/"te"/"ee"/"bb"``).
    n1_clpp : ndarray, shape (lmax, lmax)
        N1 response to ``(Clkk - fiducial Clkk)``.
    n0 : ndarray, shape (lmax,)
        Estimator normalisation. Stored as a matrix whose row 0 is ``n0`` -- the
        shipped file's layout, which ``from_sacc`` reads back via ``[0]``.
    fiducial_out, corrections_out : path-like, optional
        Save destinations for the two files.

    Returns ``(fiducial_sacc, corrections_sacc)``.
    """
    import sacc

    lmax = len(np.asarray(fiducial["kk"]))
    ell = np.arange(lmax)

    def _add_tracer(s, name):
        s.add_tracer("Map", name, quantity=_LENS_Q[name], spin=0,
                     ell=ell, beam=np.ones(lmax))

    fid = sacc.Sacc()
    for name in ("ct", "ce", "cb", "ck"):
        _add_tracer(fid, name)
    for key, t1, t2, dtype in _LENS_FID:
        fid.add_ell_cl(dtype, t1, t2, ell, np.asarray(fiducial[key]))
    if fiducial_out is not None:
        fid.save_fits(str(fiducial_out), overwrite=True)

    cor = sacc.Sacc()
    for name in ("ct", "ce", "cb", "cp", "n0"):
        _add_tracer(cor, name)
    for key in ("tt", "te", "ee", "bb"):
        t1, t2, dtype = _LENS_N0[key]
        cor.add_ell_cl(dtype, t1, t2, ell, np.asarray(n0_response[key]))
        t1, t2, dtype = _LENS_N1[key]
        cor.add_ell_cl(dtype, t1, t2, ell, np.asarray(n1_response[key]))
    cor.add_ell_cl("N1_00", "cp", "cp", ell, np.asarray(n1_clpp))
    # from_sacc reads the normalisation as spec("n0","n0","N0_00")[0], i.e. row 0.
    cor.add_ell_cl("N0_00", "n0", "n0", ell, np.tile(np.asarray(n0), (lmax, 1)))
    if corrections_out is not None:
        cor.save_fits(str(corrections_out), overwrite=True)
    return fid, cor


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
                "NuMap",
                f"{freq}_{spin}",
                quantity=quantity,
                spin=0 if spin == "s0" else 2,
                nu=band["nu"],
                bandpass=band["bandpass"],
                **beam,
            )

    for ia, fa in enumerate(freqs):
        for ib, fb in enumerate(freqs):
            if ia > ib:
                continue
            for ipa, pa in enumerate(_POLS):
                for pb in _POLS[ipa:] if fa == fb else _POLS:
                    ta = f"{fa}_s0" if pa == "T" else f"{fa}_s2"
                    tb = f"{fb}_s0" if pb == "T" else f"{fb}_s2"
                    cl_type = "cl_" + (
                        _MAP_TYPE[pb] + _MAP_TYPE[pa]
                        if pb == "T"
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

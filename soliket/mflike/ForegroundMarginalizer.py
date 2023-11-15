r"""
.. module:: foreground_marginalization

:Synopsis: Perform the foreground marginalization for the TTTEEE CMB power spectra for SO.
:Authors: Hidde Jense.

This file contains the utlity class for the Foreground Marginalization of the
multifrequency primary CMB data from the Simons Observatory.
"""
from typing import Any, Iterable, Optional
import numpy as np
from scipy import linalg
from .mflike import MFLike


class ForegroundMarginalizer(MFLike):
    """
    Utility class for the foreground marginalization.
    """
    lmax_extract: Optional[int] = None
    requested_cls: Iterable[str] = ["tt", "te", "ee"]
    foregrounds: Any
    theoryforge: Any
    bandpass: Any

    def initialize(self):
        super().initialize()

        self.lmax_extract = self.lmax_extract or self.lmax_theory
        if type(self.lmax_extract) == int:
            self.lmax_extract = {k: self.lmax_extract for k in self.requested_cls}

        self.last_extract = None
        self.current_extract = None

        if self.bandpass is not None:
            self.bandpass.exp_ch = self.experiments
            self.bandpass.bands = self.bands

        if self.theoryforge is not None:
            self.theoryforge.lmin = self.l_bpws.min()
            self.theoryforge.lmax = self.l_bpws.max()
            self.theoryforge.ell = self.l_bpws

        if self.foregrounds is not None:
            self.foregrounds.lmin = 0
            self.foregrounds.lmax = self.l_bpws.max()
            self.foregrounds.ell = np.arange(0, self.l_bpws.max() + 1)

    def make_mapping_matrix(self, **params_nuisance):
        """We build an overview of how many bins we are extracting per """
        self.extract_bins = {spec: 0 for spec in self.lcuts.keys()}
        self.lbins = {spec: [] for spec in self.lcuts.keys()}

        for m in self.spec_meta:
            i = m["ids"][m["leff"] <= self.lmax_extract.get(m["pol"], 0)]
            p = m["pol"]

            if len(i) > self.extract_bins.get(p, 0):
                self.extract_bins[p] = len(i)
                self.lbins[p] = m["leff"][m["leff"] <= self.lmax_extract.get(m["pol"], 0)]

        """Save the zero-point of each spectrum."""
        self.extract_zero = {}
        i = 0
        for k in self.extract_bins:
            self.extract_zero[k] = i
            i += self.extract_bins[k]
        self.total_bins = i

        """Prepare the mapping and projection matrix."""
        self.mapping_matrix = np.zeros((len(self.data_vec), self.total_bins))
        self.projection_matrix = np.zeros((len(self.data_vec), self.total_bins))
        self.mapping_ls = np.zeros((self.total_bins))

        for m in self.spec_meta:
            ls = m["leff"][m["leff"] <= self.lmax_extract.get(m["pol"], 0)]

            # If we do not extract any bins, ignore this spectrum.
            if len(ls) == 0:
                continue

            i = m["ids"][m["leff"] <= self.lmax_extract.get(m["pol"], 0)]
            p = m["pol"]

            e1, e2 = m["t1"], m["t2"]
            p1, p2 = p.upper()
            cal1 = params_nuisance["calG_all"] * params_nuisance[f"cal_{e1}"] * \
                   params_nuisance[f"cal{p1}_{e1}"]
            cal2 = params_nuisance["calG_all"] * params_nuisance[f"cal_{e2}"] * \
                   params_nuisance[f"cal{p2}_{e2}"]

            j = self.extract_zero[p] + np.where(np.in1d(self.lbins[p], ls))[0]

            # Now we can populate the matrices.
            self.projection_matrix[i, j] = 1.0
            self.mapping_matrix[i, j] = 1.0 / (cal1 * cal2)
            binned_ls = m["bpw"].weight.T @ m["bpw"].values
            self.mapping_ls[j] = binned_ls[m["leff"] <= self.lmax_extract[m["pol"]]]

        # Now we can build the extract & step matrix.
        self.sampling_matrix = self.mapping_matrix.T @ self.inv_cov
        self.extract_matrix = linalg.inv(self.sampling_matrix @ self.mapping_matrix)
        self.step_matrix = linalg.cholesky(self.extract_matrix, lower=True)

    def extract(self, fg_vec):
        bb = self.sampling_matrix @ (self.data_vec - fg_vec)
        wb = self.extract_matrix @ bb
        nb = np.random.normal(size=(self.total_bins,))
        gn = self.step_matrix @ nb

        if self.current_extract is not None:
            self.last_extract = self.current_extract.copy()
        self.current_extract = wb + gn

        return self.current_extract

    def get_modified_theory(self, Dls: Optional[dict] = None, **params):
        if Dls is None:
            Dls = {s: np.zeros(self.l_bpws.max() + 1) for s in self.lcuts}
        else:
            Dls = {s: Dls[s][self.l_bpws] for s in self.lcuts}

        bandint_freqs = self.bandpass._bandpass_construction(**params)
        fg_dict = self.foregrounds._get_foreground_model(
            requested_cls=self.requested_cls,
            exp_ch=self.experiments,
            bandint_freqs=bandint_freqs,
            **params)
        DlsObs = self.theoryforge.get_modified_theory(Dls, fg_dict, **params)

        return DlsObs

    def get_theory_vector(self, Dls: Optional[dict] = None, **params):
        DlsObs = self.get_modified_theory(Dls, **params)

        ps_vec = np.zeros_like(self.data_vec)
        for m in self.spec_meta:
            p = m["pol"]
            i = m["ids"]
            w = m["bpw"].weight.T
            clt = w @ DlsObs[p, m["t1"], m["t2"]][self.l_bpws - 2]
            ps_vec[i] = clt

        return ps_vec

    def logp(self, dls, **params):
        self.make_mapping_matrix(**params)

        fg_vec = self.get_theory_vector(dls, **params)
        ps_vec = self.extract(fg_vec)

        return self.loglike(dls, ps_vec, **params)

    def loglike(self, dls, cl_vec, **params):
        ps_vec = self.get_theory_vector(dls, **params) + self.projection_matrix @ cl_vec

        delta = self.data_vec - ps_vec
        logp = -0.5 * (delta @ self.inv_cov @ delta)
        logp += self.logp_const
        self.log.debug(
            f"Log-likelihood value computed = {logp} \
              (Χ² = {-2 * (logp - self.logp_const)})"
        )
        return logp


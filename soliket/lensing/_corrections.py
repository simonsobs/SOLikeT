"""N0/N1 lensing bias correction, decoupled from the likelihood.

The full ``LensingLikelihood`` corrects its binned :math:`C_L^{\\kappa\\kappa}`
theory for the response of the lensing estimator to deviations of the CMB spectra
(:math:`N_0`) and of :math:`C_L^{\\kappa\\kappa}` (:math:`N_1`) away from the
fiducial spectra used to compute its normalisation (see Sec. 5.9 / App. E of
`Qu et al. 2023 <https://arxiv.org/abs/2304.05202>`_).

:class:`LensingCorrections` bundles the static, cosmology-independent inputs (the
:math:`N_0`/:math:`N_1` response matrices keyed by spectrum, the fiducial spectra
and the normalisation) and exposes :meth:`compute`, which applies them to a
per-step theory vector. :meth:`compute` is free of SACC / cobaya coupling so it
can be built and called externally (e.g. from the simulation notebooks) and
unit-tested in isolation; :meth:`from_sacc` is the adapter that reads the shipped
correction file. The likelihood builds one with ``from_sacc`` and calls
``compute`` each step.
"""

from dataclasses import dataclass

import numpy as np

# Spectra carrying both an N0 and an N1 response, and the (tracer1, tracer2,
# data_type) triple naming each in the correction SACC.
_N0_SACC = {
    "tt": ("ct", "ct", "N0_00"),
    "te": ("ct", "ce", "N0_0e"),
    "ee": ("ce", "ce", "N0_ee"),
    "bb": ("cb", "cb", "N0_bb"),
}
_N1_SACC = {
    "tt": ("ct", "ct", "N1_00"),
    "te": ("ct", "ce", "N1_0e"),
    "ee": ("ce", "ce", "N1_ee"),
    "bb": ("cb", "cb", "N1_bb"),
}
_SPECS = tuple(_N0_SACC)


@dataclass(frozen=True)
class LensingCorrections:
    fiducial: dict[str, np.ndarray]  # fiducial CMB spectra, keyed "tt"/"te"/"ee"/"bb"
    n0_response: dict[str, np.ndarray]  # N0 response matrix per spectrum
    n1_response: dict[str, np.ndarray]  # N1 response matrix per spectrum
    n1_clpp: np.ndarray  # N1 response to (Clkk - fiducial Clkk)
    thetaclkk: np.ndarray  # fiducial C_L^kk
    n0: np.ndarray  # estimator normalisation

    def compute(
        self,
        *,
        cls: dict[str, np.ndarray],
        clkk_theo: np.ndarray,
        binning_matrix: np.ndarray,
    ) -> np.ndarray:
        r"""Binned :math:`N_0`/:math:`N_1` correction for a theory step.

        :param cls: theory CMB spectra (unbinned, to ``lmax``) keyed "tt"/"te"/"ee"/"bb".
        :param clkk_theo: theory :math:`C_L^{\kappa\kappa}` (unbinned, to ``lmax``).
        :param binning_matrix: bandpower binning matrix applied to the correction.
        :return: the binned correction to add to the binned :math:`C_L^{\kappa\kappa}`.
        """
        delta = {s: cls[s] - self.fiducial[s] for s in self.fiducial}
        n0_term = sum(self.n0_response[s] @ delta[s] for s in delta)
        n1_term = self.n1_clpp @ (clkk_theo - self.thetaclkk) + sum(
            self.n1_response[s] @ delta[s] for s in delta
        )
        correction = 2 * (self.thetaclkk / self.n0) * n0_term + n1_term
        return binning_matrix @ correction

    @classmethod
    def from_sacc(
        cls, s, *, fiducial: dict[str, np.ndarray]
    ) -> "LensingCorrections":
        """Build from a correction SACC and the (lmax-sliced) fiducial spectra.

        :param s: a loaded :class:`sacc.Sacc` holding the N0/N1 response spectra.
        :param fiducial: fiducial spectra keyed "tt"/"te"/"ee"/"bb"/"kk".
        """

        def spec(tracer1, tracer2, data_type):
            _, cl = s.get_ell_cl(data_type, tracer1, tracer2, return_cov=False)
            return cl

        return cls(
            fiducial={key: fiducial[key] for key in _SPECS},
            n0_response={key: spec(*_N0_SACC[key]) for key in _SPECS},
            n1_response={key: spec(*_N1_SACC[key]) for key in _SPECS},
            n1_clpp=spec("cp", "cp", "N1_00"),
            thetaclkk=fiducial["kk"],
            n0=spec("n0", "n0", "N0_00")[0],
        )

"""
.. module:: lensing

:Synopsis: Gaussian Likelihood for CMB Lensing for Simons Observatory
:Authors: Frank Qu, Mat Madhavacheril.

This is a simple likelihood which inherits from generic binned power spectrum (PS)
likelihood. It comes in two forms: the full ``LensingLikelihood`` which requires
(automated) downloading of external data and a more lightweight ``LensingLiteLikelihood``
which is less accurate (and should only be used for testing) but does not require the
data download.
"""

import os
from typing import ClassVar

import numpy as np
import sacc
from cobaya.likelihoods.base_classes import InstallableLikelihood
from cobaya.log import LoggedError
from cobaya.model import get_model
from cobaya.theory import Provider

from soliket.ccl import CCL
from soliket.gaussian import GaussianLikelihood


class LensingLikelihood(GaussianLikelihood, InstallableLikelihood):
    r"""
    The full ``LensingLikelihood`` makes use of a *fiducial* lensing power spectrum which
    is calculated at a hard-coded set of fiducial cosmological parameters. This fiducial
    spectrum is combined with noise power spectra correction terms
    (:math:`N_0` and :math:`N_1` terms calculated using
    `this code <https://github.com/simonsobs/so-lenspipe/blob/master/bin/n1so.py>`_)
    appropriate for SO accounting for known biases in
    the lensing estimators. These correction terms are then combined with the power
    spectrum calculated at each Monte Carlo step. For more details on the calculation of
    the corrected power spectrum see e.g. Section 5.9 and Appendix E of
    `Qu et al (2023) <https://arxiv.org/abs/2304.05202>`_.

    Noise power spectra are downloaded as part of the ``LensingLikelihood`` installation.
    This is an `Installable Likelihood
    <https://cobaya.readthedocs.io/en/latest/installation_cosmo.html>`_
    with necessary data files stored on NERSC. You can install these data files either by
    running ``cobaya-install`` on the yaml file specifying your run, or letting the
    Likelihood install itself at run time. Please see the cobaya documentation for more
    information about installable likelihoods.
    """

    name: str = "CMB Lensing"
    _url: str = (
        "https://portal.nersc.gov/project/act/jia_qu/lensing_like/likelihood.tar.gz"
    )
    install_options: ClassVar = {"download_url": _url}
    data_folder: str = "LensingLikelihood/"
    data_filename: str = "lensing.sacc.fits"

    use_spectra: str | tuple[str, str] = ("ck", "ck")
    sim_number: int = 0
    lmax: int = 3000
    theory_lmax: int = 10000
    # flag about whether CCL should be used to compute the cmb lensing power spectrum
    pp_ccl: bool = False
    provider: Provider

    fiducial_from_file: bool = True
    fiducial_filename: str | None = "fiducial_lensing.sacc.fits"
    correction_filename: str | None = "corrections_lensing.sacc.fits"

    fiducial_params: ClassVar = {
        "ombh2": 0.02219218,
        "omch2": 0.1203058,
        "H0": 67.02393,
        "tau": 0.6574325e-01,
        "nnu": 3.046,
        "As": 2.15086031154146e-9,
        "ns": 0.9625356e00,
    }

    _allowable_tracers: ClassVar[list[str]] = ["cmb_convergence"]

    def initialize(self):
        self.datapath = self._get_datapath()
        super().initialize()
        _, self.binning_matrix = self.get_binning(self.tracer_comb)

        # Set the fiducial spectra
        self.ls = np.arange(0, self.lmax, dtype=np.longlong)
        self._set_fiducial_Cls()

        # set the correction terms generate from the script n1so.py
        self._set_correction_factors()

    def _get_datapath(self) -> str:
        if (not getattr(self, "path", None)) and (
            not getattr(self, "packages_path", None)
        ):
            raise LoggedError(
                self.log,
                "No path given to LensingLikelihood data. "
                "Set the likelihood property "
                "'path' or 'packages_path'",
            )

        # If no path specified, use the modules path
        data_file_path = os.path.normpath(
            getattr(self, "path", None) or os.path.join(self.packages_path, "data")
        )

        self.data_folder = os.path.join(data_file_path, self.data_folder)
        if not os.path.exists(self.data_folder):
            if not getattr(self, "path", None):
                self.install(path=self.packages_path)
            else:
                raise LoggedError(
                    self.log,
                    "The 'data_folder' directory does not exist. "
                    "Check the given path [%s].",
                    self.data_folder,
                )
        return os.path.join(self.data_folder, self.data_filename)

    def _set_correction_factors(self):
        if self.correction_filename is not None:
            assert self.correction_filename.endswith((".fits", ".sacc")), (
                "Passing 'correction_filepath' LensingLikelihood tries to load it as a "
                "'sacc file'. Remove it to use default correction factors."
            )
            self.log.info(
                f"Loading correction factors from file: {self.correction_filename}"
            )
            s = sacc.Sacc.load_fits(
                os.path.join(self.data_folder, self.correction_filename)
            )

            _, self.N0cltt = self._get_spectrum_from_sacc(
                s, "ct", "ct", data_type="N0_00"
            )
            _, self.N0clte = self._get_spectrum_from_sacc(
                s, "ct", "ce", data_type="N0_0e"
            )
            _, self.N0clee = self._get_spectrum_from_sacc(
                s, "ce", "ce", data_type="N0_ee"
            )
            _, self.N0clbb = self._get_spectrum_from_sacc(
                s, "cb", "cb", data_type="N0_bb"
            )
            _, self.N1clpp = self._get_spectrum_from_sacc(
                s, "cp", "cp", data_type="N1_00"
            )
            _, self.N1cltt = self._get_spectrum_from_sacc(
                s, "ct", "ct", data_type="N1_00"
            )
            _, self.N1clte = self._get_spectrum_from_sacc(
                s, "ct", "ce", data_type="N1_0e"
            )
            _, self.N1clee = self._get_spectrum_from_sacc(
                s, "ce", "ce", data_type="N1_ee"
            )
            _, self.N1clbb = self._get_spectrum_from_sacc(
                s, "cb", "cb", data_type="N1_bb"
            )
            _, self.n0 = self._get_spectrum_from_sacc(s, "n0", "n0", data_type="N0_00")
            self.n0 = self.n0[0]
        else:
            raise LoggedError(
                self.log,
                "No correction file provided. "
                "Set the 'correction_filename' property of LensingLikelihood.",
            )

    def _get_spectrum_from_sacc(
        self, s: sacc.Sacc, name1: str, name2: str, data_type: str | None = None
    ) -> np.ndarray:
        ls, cl = s.get_ell_cl(data_type, name1, name2, return_cov=False)
        return ls, cl

    def _set_fiducial_Cls(self) -> dict:
        """
        Obtain a set of fiducial ``Cls`` from theory provider (e.g. ``camb``).
        Fiducial ``Cls`` are used to compute correction terms for the theory vector.

        :return: Fiducial ``Cls``
        """
        if self.fiducial_from_file:
            assert self.fiducial_filename is not None
            self.log.info(f"Loading fiducial Cls from file: {self.fiducial_filename}")
            if self.fiducial_filename.endswith((".fits", ".sacc")):
                s = sacc.Sacc.load_fits(
                    os.path.join(self.data_folder, self.fiducial_filename)
                )

                _, Cls_tt = self._get_spectrum_from_sacc(s, "ct", "ct")
                _, Cls_ee = self._get_spectrum_from_sacc(s, "ce", "ce")
                _, Cls_bb = self._get_spectrum_from_sacc(s, "cb", "cb")
                _, Cls_te = self._get_spectrum_from_sacc(s, "ct", "ce")

                try:
                    _, Cls_kk = self._get_spectrum_from_sacc(s, "ck", "ck")
                except Exception:
                    ls, Cls_pp = self._get_spectrum_from_sacc(s, "cp", "cp")
                    Cls_kk = Cls_pp * (ls * (ls + 1)) ** 2 * 0.25

                Cls = {
                    "kk": Cls_kk,
                    "tt": Cls_tt,
                    "ee": Cls_ee,
                    "bb": Cls_bb,
                    "te": Cls_te,
                }
            else:
                raise LoggedError(
                    self.log,
                    "Fiducial Cls file not recognized. "
                    "Please provide a .fits or .sacc file.",
                )
        else:
            info_fiducial = {
                "params": self.fiducial_params,
                "likelihood": {"soliket.utils.OneWithCls": {"lmax": self.theory_lmax}},
                "theory": {"camb": {"extra_args": {"kmax": 0.9}}},
            }
            model_fiducial = get_model(info_fiducial)
            model_fiducial.logposterior({})
            Cls = model_fiducial.provider.get_Cl(ell_factor=False)
            Cls["kk"] = Cls["pp"][0 : self.lmax] * (self.ls * (self.ls + 1)) ** 2 * 0.25

        self.fcltt = Cls["tt"][0 : self.lmax]
        self.fclee = Cls["ee"][0 : self.lmax]
        self.fclte = Cls["te"][0 : self.lmax]
        self.fclbb = Cls["bb"][0 : self.lmax]
        self.thetaclkk = Cls["kk"][0 : self.lmax]
        return Cls

    def get_requirements(self) -> dict:
        """
        Set ``lmax`` for theory ``Cls``

        :return: Dictionary ``Cl`` of lmax for each spectrum type.
        """
        if self.pp_ccl is False:
            return {
                "Cl": {
                    "pp": self.theory_lmax,
                    "tt": self.theory_lmax,
                    "te": self.theory_lmax,
                    "ee": self.theory_lmax,
                    "bb": self.theory_lmax,
                }
            }
        else:
            return {
                "Cl": {
                    "pp": self.theory_lmax,
                    "tt": self.theory_lmax,
                    "te": self.theory_lmax,
                    "ee": self.theory_lmax,
                    "bb": self.theory_lmax,
                },
                "CCL": {"kmax": 10, "nonlinear": True},
                "zstar": None,
            }

    def _get_CCL_results(self) -> tuple[CCL, dict]:
        cosmo_dict = self.provider.get_CCL()
        return cosmo_dict["ccl"], cosmo_dict["cosmo"]

    def _get_theory(self, **params_values) -> np.ndarray:
        r"""
        Generate binned theory vector of :math:`\kappa \kappa` with correction terms.

        :param params_values: Dictionary of cosmological parameters.

        :return: Array ``Clkk``.
        """
        cl = self.provider.get_Cl(ell_factor=False)

        if self.pp_ccl is False:
            Cl_theo = cl["pp"][0 : self.lmax]
            ls = self.ls
            Clkk_theo = (ls * (ls + 1)) ** 2 * Cl_theo * 0.25
        else:
            ccl, cosmo = self._get_CCL_results()
            zstar = self.provider.get_param("zstar")
            cmbk = ccl.CMBLensingTracer(cosmo, z_source=zstar)
            Clkk_theo = ccl.angular_cl(cosmo, cmbk, cmbk, self.ls)

        Cl_tt = cl["tt"][0 : self.lmax]
        Cl_ee = cl["ee"][0 : self.lmax]
        Cl_te = cl["te"][0 : self.lmax]
        Cl_bb = cl["bb"][0 : self.lmax]

        Clkk_binned = self.binning_matrix.dot(Clkk_theo)

        correction = (
            2
            * (self.thetaclkk / self.n0)
            * (
                np.dot(self.N0cltt, Cl_tt - self.fcltt)
                + np.dot(self.N0clee, Cl_ee - self.fclee)
                + np.dot(self.N0clbb, Cl_bb - self.fclbb)
                + np.dot(self.N0clte, Cl_te - self.fclte)
            )
            + np.dot(self.N1clpp, Clkk_theo - self.thetaclkk)
            + np.dot(self.N1cltt, Cl_tt - self.fcltt)
            + np.dot(self.N1clee, Cl_ee - self.fclee)
            + np.dot(self.N1clbb, Cl_bb - self.fclbb)
            + np.dot(self.N1clte, Cl_te - self.fclte)
        )

        # put the correction term into bandpowers
        correction = self.binning_matrix.dot(correction)

        return Clkk_binned + correction


class LensingLiteLikelihood(GaussianLikelihood):
    """
    Lite version of Lensing Likelihood for quick tests, which does not make any of the
    bias corrections requiring fiducial spectra calculations or downloads of external
    data. Simply a Gaussian likelihood between a provided binned ``pp`` data vector
    and covariance matrix, and the appropriate theory vector.
    """

    lmax: int = 3000
    data_filename: str = "lensing.sacc.fits"
    use_spectra: str | tuple[str, str] = ("ck", "ck")

    _allowable_tracers: ClassVar[list[str]] = ["cmb_convergence"]

    def initialize(self):
        data = os.path.join(self.get_class_path(), "data")
        self.datapath = self.datapath or os.path.join(data, self.data_filename)
        super().initialize()
        _, self.binning_matrix = self.get_binning(self.tracer_comb)
        self.ls = np.arange(0, self.lmax, dtype=np.longlong)

    def get_requirements(self) -> dict:
        return {"Cl": {"pp": self.lmax}}

    def _get_theory(self, **params_values) -> np.ndarray:
        cl = self.provider.get_Cl(ell_factor=False)
        Clkk_theo = (self.ls * (self.ls + 1)) ** 2 * cl["pp"][0 : self.lmax] * 0.25

        Clkk_binned = self.binning_matrix.dot(Clkk_theo)
        return Clkk_binned

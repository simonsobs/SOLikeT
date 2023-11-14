r"""
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
import numpy as np
import sacc
from cobaya.likelihoods.base_classes import InstallableLikelihood
from cobaya.model import get_model
from cobaya.log import LoggedError
# from cobaya.install import NotInstalledError

from ..ps import BinnedPSLikelihood


class LensingLikelihood(BinnedPSLikelihood, InstallableLikelihood):
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
    _url = "https://portal.nersc.gov/project/act/jia_qu/lensing_like/likelihood.tar.gz"
    install_options = {"download_url": _url}
    data_folder = "LensingLikelihood/"
    data_filename = "clkk_reconstruction_sim.fits"

    kind = "pp"
    sim_number = 0
    lmax = 3000
    theory_lmax = 10000

    fiducial_params = {
        "ombh2": 0.02219218,
        "omch2": 0.1203058,
        "H0": 67.02393,
        "tau": 0.6574325e-01,
        "nnu": 3.046,
        "As": 2.15086031154146e-9,
        "ns": 0.9625356e00,
    }

    def initialize(self):
        self.log.info("Initialising.")
        # Set path to data
        if ((not getattr(self, "path", None)) and
                (not getattr(self, "packages_path", None))):
            raise LoggedError(
                self.log,
                "No path given to LensingLikelihood data. "
                "Set the likelihood property "
                "'path' or 'packages_path'"
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

        # Set files where data/covariance are loaded from
        self.datapath = os.path.join(self.data_folder, self.data_filename)
        self.sacc = sacc.Sacc.load_fits(self.datapath)

        # x, y = self._get_data()
        self.cov = self._get_cov()
        self.binning_matrix = self._get_binning_matrix()

        # Initialize fiducial PS
        Cls = self._get_fiducial_Cls()

        # Set the fiducial spectra
        self.ls = np.arange(0, self.lmax)
        self.fcltt = Cls["tt"][0: self.lmax]
        self.fclpp = Cls["pp"][0: self.lmax]
        self.fclee = Cls["ee"][0: self.lmax]
        self.fclte = Cls["te"][0: self.lmax]
        self.fclbb = Cls["bb"][0: self.lmax]
        self.thetaclkk = self.fclpp * (self.ls * (self.ls + 1)) ** 2 * 0.25

        # load the correction terms generate from the script n1so.py

        self.N0cltt = np.loadtxt(os.path.join(self.data_folder, "n0mvdcltt1.txt")).T
        self.N0clte = np.loadtxt(os.path.join(self.data_folder, "n0mvdclte1.txt")).T
        self.N0clee = np.loadtxt(os.path.join(self.data_folder, "n0mvdclee1.txt")).T
        self.N0clbb = np.loadtxt(os.path.join(self.data_folder, "n0mvdclbb1.txt")).T
        self.N1clpp = np.loadtxt(os.path.join(self.data_folder, "n1mvdclkk1.txt")).T
        self.N1cltt = np.loadtxt(os.path.join(self.data_folder, "n1mvdcltte1.txt")).T
        self.N1clte = np.loadtxt(os.path.join(self.data_folder, "n1mvdcltee1.txt")).T
        self.N1clee = np.loadtxt(os.path.join(self.data_folder, "n1mvdcleee1.txt")).T
        self.N1clbb = np.loadtxt(os.path.join(self.data_folder, "n1mvdclbbe1.txt")).T
        self.n0 = np.loadtxt(os.path.join(self.data_folder, "n0mv.txt"))

        super().initialize()

    def _get_fiducial_Cls(self):
        """
        Obtain a set of fiducial ``Cls`` from theory provider (e.g. ``camb``).
        Fiducial ``Cls`` are used to compute correction terms for the theory vector.

        :return: Fiducial ``Cls``
        """
        info_fiducial = {
            "params": self.fiducial_params,
            "likelihood": {"soliket.utils.OneWithCls": {"lmax": self.theory_lmax}},
            "theory": {"camb": {"extra_args": {"kmax": 0.9}}},
            # "modules": modules_path,
        }
        model_fiducial = get_model(info_fiducial)
        model_fiducial.logposterior({})
        Cls = model_fiducial.provider.get_Cl(ell_factor=False)
        return Cls

    def get_requirements(self):
        """
        Set ``lmax`` for theory ``Cls``

        :return: Dictionary ``Cl`` of lmax for each spectrum type.
        """
        return {
            "Cl": {
                "pp": self.theory_lmax,
                "tt": self.theory_lmax,
                "te": self.theory_lmax,
                "ee": self.theory_lmax,
                "bb": self.theory_lmax,
            }
        }

    def _get_data(self):
        bin_centers, bandpowers, cov = \
            self.sacc.get_ell_cl(None, 'ck', 'ck', return_cov=True)
        self.x = bin_centers
        self.y = bandpowers
        return bin_centers, self.y

    def _get_cov(self):
        bin_centers, bandpowers, cov = \
            self.sacc.get_ell_cl(None, 'ck', 'ck', return_cov=True)
        self.cov = cov
        return cov

    def _get_binning_matrix(self):

        bin_centers, bandpowers, cov, ind = \
            self.sacc.get_ell_cl(None, 'ck', 'ck', return_cov=True, return_ind=True)
        bpw = self.sacc.get_bandpower_windows(ind)
        binning_matrix = bpw.weight.T
        self.binning_matrix = binning_matrix
        return binning_matrix

    def _get_theory(self, **params_values):
        """
        Generate binned theory vector of :math:`\kappa \kappa` with correction terms.

        :param params_values: Dictionary of cosmological parameters.

        :return: Array ``Clkk``.
        """
        cl = self.provider.get_Cl(ell_factor=False)

        Cl_theo = cl["pp"][0: self.lmax]
        Cl_tt = cl["tt"][0: self.lmax]
        Cl_ee = cl["ee"][0: self.lmax]
        Cl_te = cl["te"][0: self.lmax]
        Cl_bb = cl["bb"][0: self.lmax]

        ls = self.ls
        Clkk_theo = (ls * (ls + 1)) ** 2 * Cl_theo * 0.25

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


class LensingLiteLikelihood(BinnedPSLikelihood):
    """
    Lite version of Lensing Likelihood for quick tests, which does not make any of the
    bias corrections requiring fiducial spectra calculations or downloads of external
    data. Simply a Gaussian likelihood between a provided binned ``pp`` data vector
    and covariance matrix, and the appropriate theory vector.
    """
    kind: str = "pp"
    lmax: int = 3000

    def initialize(self):
        data = os.path.join(self.get_class_path(), 'data')
        self.datapath = self.datapath or os.path.join(data, 'binnedauto.txt')
        self.covpath = self.covpath or os.path.join(data, 'binnedcov.txt')
        self.binning_matrix_path = self.binning_matrix_path or \
                                   os.path.join(data, 'binningmatrix.txt')
        super().initialize()

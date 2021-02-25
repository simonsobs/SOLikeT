import os
from pkg_resources import resource_filename

import numpy as np

from cobaya.likelihoods._base_classes import _InstallableLikelihood
from cobaya.conventions import _packages_path
from cobaya.model import get_model
from cobaya.log import LoggedError
# from cobaya.install import NotInstalledError

from .ps import BinnedPSLikelihood


class LensingLikelihood(BinnedPSLikelihood, _InstallableLikelihood):
    _url = "https://portal.nersc.gov/project/act/jia_qu/lensing_like/likelihood.tar.gz"
    install_options = {"download_url": _url}
    data_folder = "LensingLikelihood"
    data_filename = "binnedauto.txt"
    cov_filename = "binnedcov.txt"

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
        "num_massive_neutrinos": 1,
        "As": 2.15086031154146e-9,
        "ns": 0.9625356e00,
    }

    def initialize(self):
        self.log.info("Initialising.")
        # Set path to data
        if (not getattr(self, "path", None)) and (not getattr(self, _packages_path, None)):
            raise LoggedError(
                self.log,
                "No path given to LensingLikelihood data. "
                "Set the likelihood property "
                "'path' or the common property '%s'.",
                _packages_path,
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
                    "The 'data_folder' directory does not exist. " "Check the given path [%s].",
                    self.data_folder,
                )

        # Set files where data/covariance are loaded from
        self.datapath = os.path.join(self.data_folder, self.data_filename)
        self.covpath = os.path.join(self.data_folder, self.cov_filename)

        # Initialize fiducial PS
        Cls = self._get_fiducial_Cls()

        # same bin edges to the one used for auto bandpowers
        # TODO: make this information part of the data product!
        self.bin_edges = np.linspace(20, 3000, 10)

        # Set the fiducial spectra
        self.ls = np.arange(0, self.lmax)
        self.fcltt = Cls["tt"][0 : self.lmax]
        self.fclpp = Cls["pp"][0 : self.lmax]
        self.fclee = Cls["ee"][0 : self.lmax]
        self.fclte = Cls["te"][0 : self.lmax]
        self.fclbb = Cls["bb"][0 : self.lmax]
        self.thetaclkk = self.fclpp * (self.ls * (self.ls + 1)) ** 2 * 0.25

        # load the correction terms generate from the script n1so.py

        self.N0cltt = np.loadtxt(os.path.join(self.data_folder, "n0mvdcltt1.txt")).transpose()
        self.N0clte = np.loadtxt(os.path.join(self.data_folder, "n0mvdclte1.txt")).transpose()
        self.N0clee = np.loadtxt(os.path.join(self.data_folder, "n0mvdclee1.txt")).transpose()
        self.N0clbb = np.loadtxt(os.path.join(self.data_folder, "n0mvdclbb1.txt")).transpose()
        self.N1clpp = np.loadtxt(os.path.join(self.data_folder, "n1mvdclkk1.txt")).transpose()
        self.N1cltt = np.loadtxt(os.path.join(self.data_folder, "n1mvdcltte1.txt")).transpose()
        self.N1clte = np.loadtxt(os.path.join(self.data_folder, "n1mvdcltee1.txt")).transpose()
        self.N1clee = np.loadtxt(os.path.join(self.data_folder, "n1mvdcleee1.txt")).transpose()
        self.N1clbb = np.loadtxt(os.path.join(self.data_folder, "n1mvdclbbe1.txt")).transpose()
        self.n0 = np.loadtxt(os.path.join(self.data_folder, "n0mv.txt"))

        super().initialize()

    def _get_fiducial_Cls(self):

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
        bandpowers = np.loadtxt(self.datapath)
        bin_centers = (self.bin_edges[:-1] + self.bin_edges[1:]) / 2

        return bin_centers, bandpowers

    def _get_theory(self, **params_values):
        cl = self.provider.get_Cl(ell_factor=False)

        Cl_theo = cl["pp"][0 : self.lmax]
        Cl_tt = cl["tt"][0 : self.lmax]
        Cl_ee = cl["ee"][0 : self.lmax]
        Cl_te = cl["te"][0 : self.lmax]
        Cl_bb = cl["bb"][0 : self.lmax]

        ls = self.ls
        Clkk_theo = (ls * (ls + 1)) ** 2 * Cl_theo * 0.25

        _, Clkk_binned = self.binner(ls, Clkk_theo, self.bin_edges)
        _, Cltt_binned = self.binner(ls, Cl_tt, self.bin_edges)

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
        _, correction = self.binner(ls, correction, self.bin_edges)

        return Clkk_binned + correction


class LensingLiteLikelihood(BinnedPSLikelihood):
    kind = "pp"
    dataroot = resource_filename(
        "soliket", "data/simulated_clkk_SO_Apr17_mv_nlkk_deproj0_SENS1_fsky_16000_iterOn_20191109"
    )
    cl_file = (
        "simulated_clkk_SO_Apr17_mv_nlkk_deproj0_SENS1_fsky_16000_iterOn_20191109_sim_{:02d}_bandpowers.txt"
    )
    cov_file = "simulated_clkk_SO_Apr17_mv_nlkk_deproj0_SENS1_fsky_16000_iterOn_20191109_binned_covmat.txt"
    sim_number = 0

    def initialize(self):
        self.datapath = os.path.join(self.dataroot, self.cl_file.format(self.sim_number))
        self.covpath = os.path.join(self.dataroot, self.cov_file)
        super().initialize()

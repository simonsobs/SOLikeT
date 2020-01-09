"""
.. module:: mflike

:Synopsis: Definition of simplistic likelihood for Simons Observatory
:Authors: Thibaut Louis, Xavier Garrido, Max Abitbol, Erminia Calabrese, Antony Lewis

"""
# Global
import os
import numpy as np

# Local
from cobaya.log import LoggedError
from cobaya.tools import are_different_params_lists
from cobaya.conventions import _path_install
from cobaya.likelihoods._base_classes import _InstallableLikelihood

from .gaussian import GaussianData
from .ps import PSLikelihood


class MFLike(PSLikelihood, _InstallableLikelihood):
    install_options = {"github_repository": "simonsobs/LAT_MFLike_data", "github_release": "v0.1"}

    def initialize(self):
        self.log.info("Initialising.")
        if not getattr(self, "path", None) and not getattr(self, "path_install", None):
            self.path_install = os.getenv("COBAYA_MODULES", None)
            if self.path_install is None:
                raise LoggedError(
                    self.log,
                    "No path given to MFLike data. Set the likelihood property "
                    "'path' or the common property '%s', or set COBAYA_MODULES env variable.",
                    _path_install,
                )
        # If no path specified, use the modules path
        data_file_path = os.path.normpath(
            getattr(self, "path", None) or os.path.join(self.path_install, "data")
        )

        self.data_folder = os.path.join(data_file_path, self.data_folder)
        if not os.path.exists(self.data_folder):
            raise LoggedError(
                self.log,
                "The 'data_folder' directory does not exist. " "Check the given path [%s].",
                self.data_folder,
            )

        # State requisites to the theory code
        self.requested_cls = ["tt", "te", "ee"]

        self.expected_params = ["a_tSZ", "a_kSZ", "a_p", "beta_p", "a_c", "beta_c", "n_CIBC", "a_s", "T_d"]
        self._lmax = None
        self._prepare_data()  # defines self.ell, self.data_vec, self.cov_mat

        # Same lmax for different cls; lmax is available after data is prepared
        self.l_maxs_cls = [self.lmax for i in self.requested_cls]

    def initialize_with_params(self):
        # Check that the parameters are the right ones
        differences = are_different_params_lists(
            self.input_params, self.expected_params, name_A="given", name_B="expected"
        )
        if differences:
            raise LoggedError(self.log, "Configuration error in parameters: %r.", differences)

    def get_requirements(self):
        return dict(Cl=dict(zip(self.requested_cls, self.l_maxs_cls)))

    def _get_theory(self, **params_values):
        return self._get_power_spectra(self._get_Cl(), **params_values)

    def _prepare_data(self):
        self.Bbl = {}
        self.data_vec = {s: [] for s in self.requested_cls}
        self.spec_list = []

        # Internal function to check for file existence
        def _check_filename(fname):
            if not os.path.exists(fname):
                raise LoggedError(
                    self.log,
                    "The {} file was not found within "
                    "{} directory.".format(os.path.basename(fname), self.data_folder),
                )
            return fname

        # Load cross power spectra
        self.ell = np.array([])
        for exp in self.experiments:
            for exp1, freqs1 in exp.items():
                for id_f1, f1 in enumerate(freqs1):
                    for exp2, freqs2 in exp.items():
                        for id_f2, f2 in enumerate(freqs2):
                            if exp1 == exp2 and id_f1 > id_f2:
                                continue

                            spec = (exp1, f1, exp2, f2)
                            spec_name = "{}_{}x{}_{}".format(*spec)
                            file_name = "{}/Dl_{}".format(self.data_folder, spec_name)
                            file_name += (
                                "_{:05d}.dat".format(self.sim_id) if isinstance(self.sim_id, int) else ".dat"
                            )

                            l, ps = self._read_spectra(_check_filename(file_name))
                            for s in self.requested_cls:
                                self.Bbl[s, spec] = np.loadtxt(
                                    _check_filename(
                                        "{}/Bbl_{}_{}.dat".format(self.data_folder, spec_name, s.upper())
                                    )
                                )
                                if s == "te":
                                    self.data_vec[s] = np.append(self.data_vec[s], (ps["te"] + ps["et"]) / 2)
                                    self.ell = np.concatenate([self.ell, l])
                                else:
                                    self.data_vec[s] = np.append(self.data_vec[s], ps[s])
                                    self.ell = np.concatenate([self.ell, l])
                            self.spec_list += [spec]

        # Read covariance matrix file
        cov_mat = np.loadtxt(_check_filename("{}/covariance.dat".format(self.data_folder)))

        # Set data given selection
        if self.select == "tt-te-ee":
            self.data_vec = np.concatenate([self.data_vec[s] for s in self.requested_cls])
        else:
            self.data_vec = self.data_vec[self.select]
            for count, s in enumerate(self.requested_cls):
                if self.select == s:
                    n_bins = int(cov_mat.shape[0])
                    cov_mat = cov_mat[
                        count * n_bins // 3 : (count + 1) * n_bins // 3,
                        count * n_bins // 3 : (count + 1) * n_bins // 3,
                    ]
        # Store covariance matrix & inverse
        self.cov_mat = cov_mat
        self.logp_const = np.log(2 * np.pi) * (-len(self.data_vec) / 2) + np.linalg.slogdet(cov_mat)[1] * (
            -1 / 2
        )
        self.inv_cov = np.linalg.inv(cov_mat)

        self.data = GaussianData("mflike", self.ell, self.data_vec, self.cov_mat)

    def _read_spectra(self, fname):
        data = np.loadtxt(fname)
        l = data[:, 0]
        spectra = ["tt", "te", "tb", "et", "bt", "ee", "eb", "be", "bb"]
        ps = {f: data[:, c + 1] for c, f in enumerate(spectra)}
        return l, ps

    def _get_power_spectra(self, cl, **params_values):
        # Get Cl's from the theory code
        Dls = {s: cl[s][2:lmax] for s, lmax in zip(self.requested_cls, self.l_maxs_cls)}
        # Get new foreground model given its nuisance parameters
        fg_model = self._get_foreground_model({k: params_values[k] for k in self.expected_params})
        # Compute chi2
        if self.select == "tt-te-ee":
            ps_vec = np.concatenate(
                [
                    np.dot(self.Bbl[s, spec], Dls[s] + fg_model[s, "all", spec[1], spec[3]])
                    for s in self.requested_cls
                    for spec in self.spec_list
                ]
            )
        else:
            ps_vec = np.concatenate(
                [
                    np.dot(
                        self.Bbl[self.select, spec],
                        Dls[self.select] + fg_model[self.select, "all", spec[1], spec[3]],
                    )
                    for spec in self.spec_list
                ]
            )
        return ps_vec

    def _get_foreground_model(self, fg_params):
        # Might change given different lmax
        lmin, lmax = 2, self.lmax
        l = np.arange(lmin, lmax)

        foregrounds = self.foregrounds
        normalisation = foregrounds["normalisation"]
        nu_0 = normalisation["nu_0"]
        ell_0 = normalisation["ell_0"]
        T_CMB = normalisation["T_CMB"]

        all_freqs = np.concatenate([v for exp in self.experiments for k, v in exp.items()])

        from fgspectra import cross as fgc
        from fgspectra import power as fgp
        from fgspectra import frequency as fgf

        cirrus = fgc.FactorizedCrossSpectrum(fgf.PowerLaw(), fgp.PowerLaw())
        ksz = fgc.FactorizedCrossSpectrum(fgf.ConstantSED(), fgp.kSZ_bat())
        cibp = fgc.FactorizedCrossSpectrum(fgf.ModifiedBlackBody(), fgp.PowerLaw())
        radio = fgc.FactorizedCrossSpectrum(fgf.PowerLaw(), fgp.PowerLaw())
        tsz = fgc.FactorizedCrossSpectrum(fgf.ThermalSZ(), fgp.tSZ_150_bat())
        cibc = fgc.FactorizedCrossSpectrum(fgf.CIB(), fgp.PowerLaw())

        model = {}
        model["tt", "kSZ"] = fg_params["a_kSZ"] * ksz({"nu": all_freqs}, {"ell": l, "ell_0": ell_0})
        model["tt", "cibp"] = fg_params["a_p"] * cibp(
            {"nu": all_freqs, "nu_0": nu_0, "temp": fg_params["T_d"], "beta": fg_params["beta_p"]},
            {"ell": l, "ell_0": ell_0, "alpha": 2},
        )
        model["tt", "radio"] = fg_params["a_s"] * radio(
            {"nu": all_freqs, "nu_0": nu_0, "beta": -0.5 - 2}, {"ell": l, "ell_0": ell_0, "alpha": 2}
        )
        model["tt", "tSZ"] = fg_params["a_tSZ"] * tsz(
            {"nu": all_freqs, "nu_0": nu_0}, {"ell": l, "ell_0": ell_0}
        )
        model["tt", "cibc"] = fg_params["a_c"] * cibc(
            {"nu": all_freqs, "nu_0": nu_0, "temp": fg_params["T_d"], "beta": fg_params["beta_c"]},
            {"ell": l, "ell_0": ell_0, "alpha": 2 - fg_params["n_CIBC"]},
        )

        components = foregrounds["components"]
        component_list = {s: components[s] for s in self.requested_cls}
        fg_model = {}
        for c1, f1 in enumerate(all_freqs):
            for c2, f2 in enumerate(all_freqs):
                for s in self.requested_cls:
                    fg_model[s, "all", f1, f2] = np.zeros(len(l))
                    for comp in component_list[s]:
                        fg_model[s, comp, f1, f2] = model[s, comp][c1, c2]
                        fg_model[s, "all", f1, f2] += fg_model[s, comp, f1, f2]

        return fg_model

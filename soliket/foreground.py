import numpy as np
import os
from typing import Optional

from cobaya.theory import Theory
from cobaya.tools import are_different_params_lists
from cobaya.log import LoggedError


class Foreground(Theory):

    spectra: dict
    foregrounds: dict
    freqs: Optional[list]

    # Initializes the foreground model. It sets the SED and reads the templates
    def initialize(self):
        """
        Initializes the foreground models from ``fgspectra``. Sets the SED 
        of kSZ, tSZ, dust, radio, CIB Poisson and clustered,
        tSZxCIB, and reads the templates for CIB and tSZxCIB.
        """
        from fgspectra import cross as fgc
        from fgspectra import frequency as fgf
        from fgspectra import power as fgp

        self.expected_params_fg = ["a_tSZ", "a_kSZ", "a_p", "beta_p",
                   "a_c", "beta_c", "a_s", "a_gtt", "a_gte", "a_gee",
                   "a_psee", "a_pste", "xi", "T_d"]

        self.requested_cls = self.spectra["polarizations"]
        self.lmin = self.spectra["lmin"]
        self.lmax = self.spectra["lmax"]
        self.ell = np.arange(self.lmin, self.lmax + 1)
        self.exp_freqs = self.spectra["exp_freqs"]

        template_path = os.path.join(os.path.dirname(os.path.abspath(fgp.__file__)),
                                     'data')
        cibc_file = os.path.join(template_path, 'cl_cib_Choi2020.dat')

        # set pivot freq and multipole
        self.fg_nu_0 = self.foregrounds["normalisation"]["nu_0"]
        self.fg_ell_0 = self.foregrounds["normalisation"]["ell_0"]

        # We don't seem to be using this
        # cirrus = fgc.FactorizedCrossSpectrum(fgf.PowerLaw(), fgp.PowerLaw())
        self.ksz = fgc.FactorizedCrossSpectrum(fgf.ConstantSED(), fgp.kSZ_bat())
        self.cibp = fgc.FactorizedCrossSpectrum(fgf.ModifiedBlackBody(), fgp.PowerLaw())
        self.radio = fgc.FactorizedCrossSpectrum(fgf.PowerLaw(), fgp.PowerLaw())
        self.tsz = fgc.FactorizedCrossSpectrum(fgf.ThermalSZ(), fgp.tSZ_150_bat())
        self.cibc = fgc.FactorizedCrossSpectrum(fgf.CIB(),
                                                fgp.PowerSpectrumFromFile(cibc_file))
        self.dust = fgc.FactorizedCrossSpectrum(fgf.ModifiedBlackBody(), fgp.PowerLaw())
        self.tSZ_and_CIB = fgc.SZxCIB_Choi2020()

        self.components = self.foregrounds["components"]

    def initialize_with_params(self):
        # Check that the parameters are the right ones
        differences = are_different_params_lists(
            self.input_params, self.expected_params_fg,
            name_A="given", name_B="expected")
        if differences:
            raise LoggedError(
                self.log, "Configuration error in parameters: %r.",
                differences)

    # Gets the actual power spectrum of foregrounds given the passed parameters
    def _get_foreground_model(self,
                              requested_cls=None,
                              ell=None,
                              exp_freqs=None,
                              bandint_freqs=None,
                              **fg_params):
        r"""
        Gets the foreground power spectra for each component computed by ``fgspectra``.
        The computation assumes the bandpass transmissions from the ``BandPass`` class
        and integration in frequency is performed if the passbands are not Dirac delta.

        :param requested_cls: the fields required. If ``None``, 
                              it uses the default ones in the 
                              ``Foreground.yaml``
        :param ell: ell range. If ``None`` the default range 
                    set in ``Foreground.yaml`` is used
        :param exp_freqs: list of strings "exp_freq". If ``None``, 
                      it uses the default ones in the ``Foreground.yaml``
        :param bandint_freqs: the bandpass transmissions. If ``None`` it is built as an 
                              array of freqs trimmed from the strings in ``exp_freqs``, 
                              otherwise the transmissions computed by the ``BandPass`` 
                              class

        :return: the foreground dictionary
        """

        if not requested_cls:
            requested_cls = self.requested_cls
        # if ell = None, it uses ell from yaml, otherwise the ell array provided
        # useful to make tests at different l_max than the data
        if not hasattr(ell, '__len__'):
            ell = self.ell
        ell_0 = self.fg_ell_0
        nu_0 = self.fg_nu_0

        # Normalisation of radio sources
        ell_clp = ell * (ell + 1.)
        ell_0clp = ell_0 * (ell_0 + 1.)

        # Set component spectra
        self.fg_component_list = {s: self.components[s] for s in requested_cls}

        # Set exp_freqs list
        if not hasattr(exp_freqs, '__len__'):
            exp_freqs = self.exp_freqs

        # Set array of freqs to use if bandint_freqs is None
        if not hasattr(bandint_freqs, '__len__'):
            bandint_freqs = np.empty_like(exp_freqs, dtype = float)
            for i, e in enumerate(exp_freqs):
                bandint_freqs[i] = float(e.strip('LAT_'))


        model = {}
        model["tt", "kSZ"] = fg_params["a_kSZ"] * self.ksz({"nu": bandint_freqs},
                                                           {"ell": ell,
                                                            "ell_0": ell_0})

        model["tt", "cibp"] = fg_params["a_p"] * self.cibp({"nu": bandint_freqs,
                                                            "nu_0": nu_0,
                                                            "temp": fg_params["T_d"],
                                                            "beta": fg_params["beta_p"]},
                                                           {"ell": ell_clp,
                                                            "ell_0": ell_0clp,
                                                            "alpha": 1})

        model["tt", "radio"] = fg_params["a_s"] * self.radio({"nu": bandint_freqs,
                                                              "nu_0": nu_0,
                                                              "beta": -0.5 - 2.},
                                                             {"ell": ell_clp,
                                                              "ell_0": ell_0clp,
                                                              "alpha": 1})

        model["tt", "tSZ"] = fg_params["a_tSZ"] * self.tsz({"nu": bandint_freqs,
                                                            "nu_0": nu_0},
                                                           {"ell": ell,
                                                            "ell_0": ell_0})

        model["tt", "cibc"] = fg_params["a_c"] * self.cibc({"nu": bandint_freqs,
                                                            "nu_0": nu_0,
                                                            "temp": fg_params["T_d"],
                                                            "beta": fg_params["beta_c"]},
                                                           {'ell': ell,
                                                            'ell_0': ell_0})

        model["tt", "dust"] = fg_params["a_gtt"] * self.dust({"nu": bandint_freqs,
                                                              "nu_0": nu_0,
                                                              "temp": 19.6,
                                                              "beta": 1.5},
                                                             {"ell": ell,
                                                              "ell_0": 500.,
                                                              "alpha": -0.6})

        model["tt", "tSZ_and_CIB"] = \
            self.tSZ_and_CIB({'kwseq': ({'nu': bandint_freqs, 'nu_0': nu_0},
                                        {'nu': bandint_freqs, 'nu_0': nu_0,
                                         'temp': fg_params['T_d'],
                                         'beta': fg_params["beta_c"]})},
                             {'kwseq': ({'ell': ell, 'ell_0': ell_0,
                                         'amp': fg_params['a_tSZ']},
                                        {'ell': ell, 'ell_0': ell_0,
                                         'amp': fg_params['a_c']},
                                        {'ell': ell, 'ell_0': ell_0,
                                         'amp': - fg_params['xi'] \
                                                    * np.sqrt(fg_params['a_tSZ'] *
                                                              fg_params['a_c'])})})

        model["ee", "radio"] = fg_params["a_psee"] * self.radio({"nu": bandint_freqs,
                                                                 "nu_0": nu_0,
                                                                 "beta": -0.5 - 2.},
                                                                {"ell": ell_clp,
                                                                "ell_0": ell_0clp,
                                                                 "alpha": 1})

        model["ee", "dust"] = fg_params["a_gee"] * self.dust({"nu": bandint_freqs,
                                                              "nu_0": nu_0,
                                                              "temp": 19.6,
                                                              "beta": 1.5},
                                                             {"ell": ell,
                                                              "ell_0": 500.,
                                                              "alpha": -0.4})

        model["te", "radio"] = fg_params["a_pste"] * self.radio({"nu": bandint_freqs,
                                                                 "nu_0": nu_0,
                                                                 "beta": -0.5 - 2.},
                                                                {"ell": ell_clp,
                                                                 "ell_0": ell_0clp,
                                                                 "alpha": 1})

        model["te", "dust"] = fg_params["a_gte"] * self.dust({"nu": bandint_freqs,
                                                              "nu_0": nu_0,
                                                              "temp": 19.6,
                                                              "beta": 1.5},
                                                             {"ell": ell,
                                                              "ell_0": 500.,
                                                              "alpha": -0.4})

        fg_dict = {}
        for c1, f1 in enumerate(exp_freqs):
            for c2, f2 in enumerate(exp_freqs):
                for s in requested_cls:
                    fg_dict[s, "all", f1, f2] = np.zeros(len(ell))
                    for comp in self.fg_component_list[s]:
                        if comp == "tSZ_and_CIB":
                            fg_dict[s, "tSZ", f1, f2] = model[s, "tSZ"][c1, c2]
                            fg_dict[s, "cibc", f1, f2] = model[s, "cibc"][c1, c2]
                            fg_dict[s, "tSZxCIB", f1, f2] = (
                                model[s, comp][c1, c2]
                                - model[s, "tSZ"][c1, c2]
                                - model[s, "cibc"][c1, c2]
                            )
                            fg_dict[s, "all", f1, f2] += model[s, comp][c1, c2]
                        else:
                            fg_dict[s, comp, f1, f2] = model[s, comp][c1, c2]
                            fg_dict[s, "all", f1, f2] += fg_dict[s, comp, f1, f2]
        return fg_dict

    def must_provide(self, **requirements):
        # fg_dict is required by theoryforge
        # and requires some params to be computed
        # Assign those from theoryforge
        # otherwise use default values
        # Foreground requires bandint_freqs from BandPass
        # Bandint_freqs requires some params to be computed
        # Passing those from Foreground
        if "fg_dict" in requirements:
            req = requirements["fg_dict"]
            self.requested_cls = req.get("requested_cls", self.requested_cls)
            self.ell = req.get("ell", self.ell)
            self.exp_freqs = req.get("exp_freqs", self.exp_freqs)
            return {"bandint_freqs": {"freqs": float(self.exp_freqs.strip('LAT_')}}

    def get_bandpasses(self, **params):
        """
        Gets bandpass transmissions from the ``BandPass`` class.
        """
        return self.provider.get_bandint_freqs()

    def calculate(self, state, want_derived=False, **params_values_dict):
        """
        Fills the ``state`` dictionary of the ``Foreground`` Theory class
        with the foreground spectra, computed using the bandpass 
        transmissions from the ``BandPass`` class and the sampled foreground
        parameters.
        """
        self.bandint_freqs = self.get_bandpasses(**params_values_dict)
        fg_params = {k: params_values_dict[k] for k in self.expected_params_fg}
        state["fg_dict"] = self._get_foreground_model(requested_cls=self.requested_cls,
                                                    exp_freqs=self.exp_freqs,
                                                    bandint_freqs=self.bandint_freqs,
                                                    **fg_params)

    def get_fg_dict(self):
        """
        Returns the ``state`` dictionary of fogreground spectra
        """
        return self.current_state["fg_dict"]

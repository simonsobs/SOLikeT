r"""
.. module:: foreground

The ``Foreground`` class initialized the foreground components and computes
the foreground spectra for each component and each channel. The information
on the arrays to use come from the ``TheoryForge`` class by default, through
the dictionary ``bands``. This is a dictionary

.. code-block:: python

   {"experiment_channel": {{"nu": [freqs...],
     "bandpass": [...]}}, ...}

which is filled by ``MFLike`` using the information from the sacc file.
This dictionary is then passed to ``Bandpass`` to compute the bandpass
transmissions, which are then used for the actual foreground spectra computation.


If one wants to use this class as standalone, the ``bands`` dictionary is
filled when initializing ``Foreground``. The name of the channels to use
are read from the ``exp_ch`` list in ``Foreground.yaml``, the effective
frequencies are in the ``eff_freqs`` list. Of course the effective frequencies
have to match the information from ``exp_ch``, i.e.:

.. code-block:: yaml

  exp_ch: ["LAT_93", "LAT_145", "ACT_145"]
  eff_freqs: [93, 145, 145]


The foreground spectra in this case can be computed by calling the
function

.. code-block:: python

   Foreground._get_foreground_model(requested_cls,
                              ell,
                              exp_ch,
                              bandint_freqs=None,
                              eff_freqs,
                              **fg_params):

which will have
``bandint_freqs=None`` (no passbands from ``BandPass``). The spectra will be computed
assuming just a Dirac delta at the effective frequencies ``eff_freqs``.


"""

import numpy as np
import os
from typing import Optional

from cobaya.theory import Theory
from cobaya.tools import are_different_params_lists
from cobaya.log import LoggedError


class Foreground(Theory):
    spectra: dict
    foregrounds: dict
    eff_freqs: Optional[list]
    exp_ch: Optional[list]

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
        #self.ell = np.arange(self.lmin, self.lmax + 1)
        self.exp_ch = self.spectra["exp_ch"]
        self.eff_freqs = self.spectra["eff_freqs"]

        if hasattr(self.eff_freqs, "__len__"):
            if not len(self.exp_ch) == len(self.eff_freqs):
                raise LoggedError(
                    self.log, "list of effective frequency has to have"
                              "same length as list of channels!"
                )

        # self.bands to be filled with passbands read from sacc file
        # if mflike is used
        self.bands = {f"{expc}_s0": {'nu': [self.eff_freqs[iexpc]], 'bandpass': [1.]}
                      for iexpc, expc in enumerate(self.exp_ch)}

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
                              exp_ch=None,
                              bandint_freqs=None,
                              eff_freqs=None,
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
        :param exp_ch: list of strings "experiment_channel" used to indicate the
                      foreground components computed for a particular array
                      of an experiment.
                      If ``None``, it uses the default ones in the ``Foreground.yaml``
        :param bandint_freqs: the bandpass transmissions. If ``None`` it is built as an
                              array of frequencies stored in the ``eff_freqs`` argument,
                              which in this case has to be not ``None``. If
                              ``bandint_freqs`` is not ``None``, it is
                              the transmissions computed by the ``BandPass`` class
        :param eff_freqs: list of the effective frequencies for each channel
                          used to compute the foreground components (assuming a Dirac
                          delta passband at these frequencies) if the
                         ``bandint_freqs`` argument is not provided
        :param *fg_params: parameters of the foreground components

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

        # Set exp_ch list
        if not hasattr(exp_ch, '__len__'):
            exp_ch = self.exp_ch

        # Set array of freqs to use if bandint_freqs is None
        if not hasattr(bandint_freqs, '__len__'):
            if hasattr(eff_freqs, '__len__'):
                bandint_freqs = np.asarray(eff_freqs)
            else:
                raise LoggedError(
                    self.log, "no frequency list provided to compute the passbands"
                )

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
        for c1, f1 in enumerate(exp_ch):
            for c2, f2 in enumerate(exp_ch):
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
            self.ell = req.get("ell", None)
            self.bands = req.get("bands", self.bands)
            self.exp_ch = req.get("exp_ch", self.exp_ch)
            return {"bandint_freqs": {"bands": self.bands}}

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

        :param state: ``state`` dictionary to be filled with computed foreground
                      spectra
        :param *params_values_dict: dictionary of parameters from the sampler
        """

        # compute bandpasses at each step only if bandint_shift params are not null
        # and bandint_freqs has been computed at least once
        if np.all(
                np.array([params_values_dict[k] for k in params_values_dict.keys()
                          if "bandint_shift_" in k]) == 0.0
        ):
            if not hasattr(self, "bandint_freqs"):
                self.log.info("Computing bandpass at first step, no shifts")
                self.bandint_freqs = self.get_bandpasses(**params_values_dict)
        else:
            self.bandint_freqs = self.get_bandpasses(**params_values_dict)

        fg_params = {k: params_values_dict[k] for k in self.expected_params_fg}
        self.log.info('%d', self.ell[-1])
        self.log.info('%s', self.exp_ch)
        state["fg_dict"] = self._get_foreground_model(requested_cls=self.requested_cls,
                                                      ell=self.ell,
                                                      exp_ch=self.exp_ch,
                                                      bandint_freqs=self.bandint_freqs,
                                                      **fg_params)

    def get_fg_dict(self):
        """
        Returns the ``state`` dictionary of fogreground spectra
        """
        return self.current_state["fg_dict"]

import numpy as np
import os
from typing import Optional

from cobaya.theory import Theory
from cobaya.tools import are_different_params_lists
from cobaya.log import LoggedError

from .constants import T_CMB, h_Planck, k_Boltzmann

# Converts from cmb units to brightness.
# Numerical factors not included, it needs proper normalization when used.


def _cmb2bb(nu):
    r"""
    Computes the conversion factor :math:`\frac{\partial B_{\nu}}{\partial T}`
    from CMB thermodynamic units to antenna temperature units.
    There is an additional :math:`\nu^2` factor to convert passbands from
    Rayleigh-Jeans units to antenna temperature units.

    :param nu: frequency array

    :return: the array :math:`\frac{\partial B_{\nu}}{\partial T} \nu^2`
    """
    # NB: numerical factors not included
    x = nu * h_Planck * 1e9 / k_Boltzmann / T_CMB
    return np.exp(x) * (nu * x / np.expm1(x))**2


# Provides the frequency value given the bandpass name. To be modified - it is ACT based!!
def _get_fr(array):
    r"""
    Provides the strings for the ACT frequency array. It will be removed in 
    future versions.
    """

    #a = array.split("_")[0]
    #if a == 'PA1' or a == 'PA2':
    #    fr = 150
    #if a == 'PA3':
    #    fr = array.split("_")[3]
    #return fr
    a = array.split("_")[0]
    if a == "PA1" or a == "PA2":
        fr = 150
    if a == "PA3":
        fr = array.split("_")[3]
    if a == "PA4" or a == "PA5" or a == "PA6":
        fr = array.split("_")[3].replace("f", "")
    return np.int(fr), a


def _get_arrays_weights(arrays, polarized_arrays, freqs):
    """
    Provides the array weights for the ACT frequency array. It could be removed in
    future versions.
    """
    array_weights = {}
    counter = []
    for array in arrays:
        fr, pa = _get_fr(array)
        if (pa in polarized_arrays) and (fr in freqs):
            if fr not in counter:
                counter.append(fr)
                array_weights[fr] = 1
            else:
                array_weights[fr] = array_weights[fr] + 1
    return array_weights


class BandPass(Theory):

    # attributes set from .yaml
    data_folder: Optional[str]
    band_integration: dict

    def initialize(self):

        self.expected_params_bp = ["bandint_shift_93",
                                   "bandint_shift_145",
                                   "bandint_shift_225"]

        self.freqs = None

        # Parameters for band integration
        self.bandint_nsteps = self.band_integration["nsteps"]
        self.bandint_width = self.band_integration["bandwidth"]
        self.bandint_external_bandpass = self.band_integration["external_bandpass"]
        # Bandpass construction for band integration
        #if self.bandint_external_bandpass:
        #    path = os.path.normpath(os.path.join(self.data_folder,
        #                                         '/bp_int/'))
        #    arrays = os.listdir(path)
        #    self._init_external_bandpass_construction(arrays)
        if self.bandint_external_bandpass:
            path = os.path.normpath(os.path.join(self.data_folder,
                                    "external_bandpasses/"))
            arrays = os.listdir(path)
            self._init_external_bandpass_construction(path, arrays)
            self.array_weights = _get_arrays_weights(arrays,
                                                     self.polarized_arrays,
                                                     self.freqs)

    def initialize_with_params(self):
        # Check that the parameters are the right ones
        differences = are_different_params_lists(
            self.input_params, self.expected_params_bp,
            name_A="given", name_B="expected")
        if differences:
            raise LoggedError(
                self.log, "Configuration error in parameters: %r.",
                differences)

    def must_provide(self, **requirements):
        # bandint_freqs is required by Foreground
        # and requires some params to be computed
        # Assign those from Foreground
        if "bandint_freqs" in requirements:
            self.freqs = requirements["bandint_freqs"]["freqs"]

    def calculate(self, state, want_derived=False, **params_values_dict):
        r"""
        Adds the bandpass transmission to the ``state`` dictionary of the
        BandPass Theory class.

        :param *params_values_dict: dictionary of nuisance parameters
        """

        nuis_params = {k: params_values_dict[k] for k in self.expected_params_bp}

        # Bandpass construction for band integration
        if self.bandint_external_bandpass:
            self.bandint_freqs = self._external_bandpass_construction(**nuis_params)
        else:
            self.bandint_freqs = self._bandpass_construction(**nuis_params)

        state["bandint_freqs"] = self.bandint_freqs

    def get_bandint_freqs(self):
        """
        Returns the ``state`` dictionary of bandpass transmissions
        """
        return self.current_state["bandint_freqs"]

    # Takes care of the bandpass construction. It returns a list of nu-transmittance for
    # each frequency or an array with the effective freqs.
    def _bandpass_construction(self, **params):
        r"""
        Builds the bandpass transmission 
        :math:`\frac{\frac{\partial B_{\nu+\Delta \nu}}{\partial T} 
        (\nu+\Delta \nu)^2 \tau(\nu+\Delta \nu)}{\int d\nu 
        \frac{\partial B_{\nu}}{\partial T} (\nu)^2 \tau(\nu)}` 
        (WRONG! DENOMINATOR SHOULD USE BANDPASS SHIFT PARAM, TO BE CORRECTED!) 
        using passbands :math:`\tau(\nu)` (in RJ units, not read from file)
        and bandpass shift :math:`\Delta \nu`. :math:`\tau(\nu)` is built as a top-hat
        with width ``bandint_width`` and number of samples ``nsteps``, 
        read from the ``BandPass.yaml``.
        If ``nstep = 1`` and ``bandint_width = 0``, the passband is a Dirac delta
        centered at :math:`\nu+\Delta \nu`.

        :param *params: dictionary of nuisance parameters 
        :return: the list of [nu, transmission] in the multifrequency case  
                 or just an array of frequencies in the single frequency one
        """

        if not hasattr(self.bandint_width, "__len__"):
            self.bandint_width = np.full_like(self.freqs, self.bandint_width,
                                              dtype=float)
        if np.any(np.array(self.bandint_width) > 0):
            assert self.bandint_nsteps > 1, 'bandint_width and bandint_nsteps not \
                                             coherent'
            assert np.all(np.array(self.bandint_width) > 0), 'one band has width = 0, \
                                                              set a positive width and \
                                                              run again'

            bandint_freqs = []
            for ifr, fr in enumerate(self.freqs):
                bandpar = 'bandint_shift_' + str(fr)
                bandlow = fr * (1 - self.bandint_width[ifr] * .5)
                bandhigh = fr * (1 + self.bandint_width[ifr] * .5)
                nubtrue = np.linspace(bandlow, bandhigh, self.bandint_nsteps, dtype=float)
                nub = np.linspace(bandlow + params[bandpar], bandhigh + params[bandpar],
                                  self.bandint_nsteps, dtype=float)
                tranb = _cmb2bb(nub)
                tranb_norm = np.trapz(_cmb2bb(nubtrue), nubtrue)
                bandint_freqs.append([nub, tranb / tranb_norm])
        else:
            bandint_freqs = np.empty_like(self.freqs, dtype=float)
            for ifr, fr in enumerate(self.freqs):
                bandpar = 'bandint_shift_' + str(fr)
                bandint_freqs[ifr] = fr + params[bandpar]

        return bandint_freqs

    #def _init_external_bandpass_construction(self, arrays):
    #    self.external_bandpass = []
    #    for array in arrays:
    #        fr = _get_fr(array)
    #        nu_ghz, bp = np.loadtxt(array, usecols=(0, 1), unpack=True)
    #        trans_norm = np.trapz(bp * _cmb2bb(nu_ghz), nu_ghz)
    #        self.external_bandpass.append([fr, nu_ghz, bp / trans_norm])
    def _init_external_bandpass_construction(self, path, arrays):
        """
        Initializes the passband reading for ``_external_bandpass_construction``.

        :param path: path of the passband txt file
        :param arrays: list of arrays
        """
        self.external_bandpass = []
        for array in arrays:
            fr, pa = _get_fr(array)
            if pa in self.polarized_arrays and fr in self.freqs:
                nu_ghz, bp = np.loadtxt(path + "/" + array, usecols=(0, 1), unpack=True)
                trans_norm = np.trapz(bp * _cmb2bb(nu_ghz), nu_ghz)
                self.external_bandpass.append([pa, fr, nu_ghz, bp / trans_norm])

    #def _external_bandpass_construction(self, **params):
    #    bandint_freqs = []
    #    for fr, nu_ghz, bp in self.external_bandpass:
    #        bandpar = 'bandint_shift_' + str(fr)
    #        nub = nu_ghz + params[bandpar]
    #        trans = bp * _cmb2bb(nub)
    #        bandint_freqs.append([nub, trans])
    def _external_bandpass_construction(self, **params):
        r"""
        Builds bandpass transmission 
        :math:`\frac{\frac{\partial B_{\nu+\Delta \nu}}{\partial T} 
        (\nu+\Delta \nu)^2 \tau(\nu+\Delta \nu)}{\int d\nu 
        \frac{\partial B_{\nu}}{\partial T} (\nu)^2 \tau(\nu)}` 
        (WRONG! DENOMINATOR SHOULD USE BANDPASS SHIFT PARAM, TO BE CORRECTED!)  
        using passbands :math:`\tau(\nu)` (in RJ units) read from file and 
        possible bandpass shift parameters :math:`\Delta \nu`.

        :param *params: dictionary of nuisance parameters

        :return: the list of [nu, transmission]
        """
        bandint_freqs = []
        order = []
        for pa, fr, nu_ghz, bp in self.external_bandpass:
            bandpar = "bandint_shift_" + str(fr)
            nub = nu_ghz + params[bandpar]
            trans = bp * _cmb2bb(nub)
            bandint_freqs.append([nub, trans])
            order.append(pa + "_" + str(fr))

        return order, bandint_freqs

        return bandint_freqs

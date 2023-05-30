r"""
.. module:: bandpass

This module computes the bandpass transmission based on the inputs from 
the parameter file ``BandPass.yaml``. There are three possibilities:
    * reading the passband :math:`\tau(\nu)` stored in a sacc file 
      (which is the default now, being the mflike default)
    * building the passbands :math:`\tau(\nu)`, either as Dirac delta or as top-hat
    * reading the passbands :math:`\tau(\nu)` from an external file.

Fore the first option, the ``read_from_sacc`` option in ``BandPass.yaml``
has to be set to ``True``:

.. code-block:: yaml
  
  read_from_sacc: True

Otherwise, it has to be left empty. The frequencies and passbands are passed in a 
``bands`` dictionary, which is passed from ``Foregrounds`` through ``TheoryForge``.


For the second option, the ``top_hat_band`` dictionary in ``BandPass.yaml``
has to be filled with two keys:

    * ``nsteps``: setting the number of frequencies used in the band integration
      (either 1 for a Dirac delta or > 1)
    * ``bandwidth``: setting the relative width :math:`\delta` of the band with respect to
      the central frequency, such that the frequency extrems are 
      :math:`\nu_{\rm{low/high}} = \nu_{\rm{center}}(1 \mp \delta/2) + 
      \Delta^{\nu}_{\rm band}` (with :math:`\Delta^{\nu}_{\rm band}` 
       being the possible bandpass shift). ``bandwidth`` has to be 0
       if ``nstep`` = 1, > 0 otherwise. ``bandwidth`` can be a list 
       if you want a different width for each band
       e.g. ``bandwidth: [0.3,0.2,0.3]`` for 3 bands.

The effective frequencies are read from the ``bands`` dictionary as before. If we are not
using passbands from a sacc file, ``bands`` is filled in ``Foreground`` using the default
``eff_freqs`` in ``Foreground.yaml``. In this case it is filled assuming a Dirac delta.

.. code-block:: yaml
  
  top_hat_band:
    nsteps: 1
    bandwidth: 0


For the third option, the ``external_bandpass`` dictionary in ``BandPass.yaml``
has to have the the ``path`` key, representing the path to the folder with all the
passbands. 

.. code-block:: yaml
  
  external_bandpass:
    path: "path_of_passband_folder"


The path has to be relative to the ``data_folder`` in ``BandPass.yaml``. 
This folder has to have files with the names of the experiment or array and the 
nominal frequency of the channel, e.g. ``LAT_93`` or ``dr6_pa4_f150``.

To avoid the options you don't want to select, the corresponding dictionary has to be 
``null``.
If all dictionaries are ``null``, there will be an error message inviting to choose
one of the three options.

The bandpass transmission is built as 

.. math::
  \frac{\frac{\partial B_{\nu+\Delta \nu}}
  {\partial T}(\nu+\Delta \nu)^2 \tau(\nu+\Delta \nu)}{\int d\nu\frac{\partial 
  B_{\nu+\Delta \nu}}{\partial T} (\nu+\Delta \nu)^2 \tau(\nu+\Delta \nu)}

where :math:`\frac{\partial B_{\nu}}{\partial T}` converts from CMB thermodynamic 
units to antenna temperature units, the additional :math:`\nu^2` factor 
converts passbands from Rayleigh-Jeans units to antenna temperature units, 
the passband :math:`\tau(\nu)` has then to be in RJ units and :math:`\Delta \nu` is the 
possible bandpass shift for that channel.

"""

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

    Numerical constants are not included, which is not a problem when using this 
    conversion both at numerator and denominator.

    :param nu: frequency array

    :return: the array :math:`\frac{\partial B_{\nu}}{\partial T} \nu^2`
    """
    # NB: numerical factors not included
    x = nu * h_Planck * 1e9 / k_Boltzmann / T_CMB
    return np.exp(x) * (nu * x / np.expm1(x))**2


class BandPass(Theory):

    # attributes set from .yaml
    data_folder: Optional[str]
    read_from_sacc: dict
    top_hat_band: dict
    external_bandpass: dict

    def initialize(self):

        self.expected_params_bp = ["bandint_shift_LAT_93",
                                   "bandint_shift_LAT_145",
                                   "bandint_shift_LAT_225"]

        self.exp_ch = None
        #self.eff_freqs = None
        self.bands = None

        # To read passbands stored in the sacc files
        # default for mflike
        self.read_from_sacc = bool(self.read_from_sacc)

        # Parameters for band integration
        self.use_top_hat_band = bool(self.top_hat_band)
        if self.use_top_hat_band:
            self.bandint_nsteps = self.top_hat_band["nsteps"]
            self.bandint_width = self.top_hat_band["bandwidth"]

            # checks on the bandpass input params, to be done only at the initialization
            if not hasattr(self.bandint_width, "__len__"):
                self.bandint_width = np.full_like(
                    self.experiments, self.bandint_width, dtype=float
                )
            if self.bandint_nsteps > 1 and np.any(np.array(self.bandint_width) == 0):
                raise LoggedError(
                    self.log, "One band has width = 0, set a positive width and run again"
                )


        self.bandint_external_bandpass = bool(self.external_bandpass)
        if self.bandint_external_bandpass:
            path = os.path.normpath(os.path.join(self.data_folder,
                                    self.external_bandpass["path"]))
            arrays = os.listdir(path)
            self._init_external_bandpass_construction(path, arrays)


        if (not self.read_from_sacc and not self.use_top_hat_band 
                   and not self.bandint_external_bandpass):
            raise LoggedError(
                    self.log, "fill the dictionaries for either reding" \
                    "the passband from sacc file (mflike default) or an"\
                    "external passband or building a top-hat one!"
                )


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
            self.bands = requirements["bandint_freqs"]["bands"]
            self.exp_ch = [k.replace("_s0", "") for k in self.bands.keys()]
            # self.eff_freqs = [np.array(self.bands[k]['nu'])
            #                      for k in self.bands.keys()]

    def calculate(self, state, want_derived=False, **params_values_dict):
        r"""
        Adds the bandpass transmission to the ``state`` dictionary of the
        BandPass Theory class.

        :param *params_values_dict: dictionary of nuisance parameters
        """

        nuis_params = {k: params_values_dict[k] for k in self.expected_params_bp}

        # Bandpass construction
        if self.bandint_external_bandpass:
            self.bandint_freqs = self._external_bandpass_construction(**nuis_params)
        if self.read_from_sacc or self.use_top_hat_band:
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
        \frac{\partial B_{\nu+\Delta \nu}}{\partial T} (\nu+\Delta \nu)^2 
        \tau(\nu+\Delta \nu)}`  
        using passbands :math:`\tau(\nu)` (in RJ units, not read from a txt
        file) and bandpass shift :math:`\Delta \nu`. If ``read_from_sacc = True``
        (the default), :math:`\tau(\nu)` has been read from the sacc file
        and passed through ``Foreground`` from ``TheoryForge``.
        If ``use_top_hat_band``, :math:`\tau(\nu)` is built as a top-hat
        with width ``bandint_width`` and number of samples ``nsteps``, 
        read from the ``BandPass.yaml``.
        If ``nstep = 1`` and ``bandint_width = 0``, the passband is a Dirac delta
        centered at :math:`\nu+\Delta \nu`.

        :param *params: dictionary of nuisance parameters 
        :return: the list of [nu, transmission] in the multifrequency case  
                 or just an array of frequencies in the single frequency one
        """

        data_are_monofreq = False
        bandint_freqs = []
        for ifr, fr in enumerate(self.exp_ch):
            bandpar = 'bandint_shift_' + str(fr)
            bands = self.bands[f"{fr}_s0"]
            nu_ghz, bp = np.asarray(bands["nu"]), np.asarray(bands["bandpass"])
            if self.use_top_hat_band:
                # Compute central frequency given bandpass
                fr = nu_ghz @ bp / bp.sum()
                if self.bandint_nsteps > 1:
                    bandlow = fr * (1 - self.bandint_width[ifr] * .5)
                    bandhigh = fr * (1 + self.bandint_width[ifr] * .5)
                    nub = np.linspace(bandlow + params[bandpar], 
                            bandhigh + params[bandpar],
                                  self.bandint_nsteps, dtype=float)
                    tranb = _cmb2bb(nub)
                    tranb_norm = np.trapz(_cmb2bb(nub), nub)
                    bandint_freqs.append([nub, tranb / tranb_norm])
                if self.bandint_nsteps == 1:
                    nub = fr + params[bandpar]
                    data_are_monofreq = True
                    bandint_freqs.append(nub)
            if self.read_from_sacc:
                nub = nu_ghz + params[bandpar]
                if len(bp) == 1:
                    # Monofrequency channel
                    data_are_monofreq = True
                    bandint_freqs.append(nub[0])
                else:
                    trans_norm = np.trapz(bp * _cmb2bb(nub), nub)
                    trans = bp / trans_norm * _cmb2bb(nub)
                    bandint_freqs.append([nub, trans])

        if data_are_monofreq:
            bandint_freqs = np.asarray(bandint_freqs)
            self.log.info("bandpass is delta function, no band integration performed")

        return bandint_freqs

    def _init_external_bandpass_construction(self, exp_ch, path):
        """
        Initializes the passband reading for ``_external_bandpass_construction``.

        :param exp_ch: list of the frequency channels
        :param path: path of the passband txt file
        """
        self.external_bandpass = []
        for expc in exp_ch:
            if expc in self.exp_ch:
                nu_ghz, bp = np.loadtxt(path + "/" + expc, usecols=(0, 1), unpack=True)
                self.external_bandpass.append([expc, nu_ghz, bp])

    def _external_bandpass_construction(self, **params):
        r"""
        Builds bandpass transmission 
        :math:`\frac{\frac{\partial B_{\nu+\Delta \nu}}{\partial T} 
        (\nu+\Delta \nu)^2 \tau(\nu+\Delta \nu)}{\int d\nu 
        \frac{\partial B_{\nu+\Delta \nu}}{\partial T} (\nu+\Delta \nu)^2 
        \tau(\nu+\Delta \nu)}`   
        using passbands :math:`\tau(\nu)` (in RJ units) read from 
        an external txt file and 
        possible bandpass shift parameters :math:`\Delta \nu`.

        :param *params: dictionary of nuisance parameters

        :return: the list of [nu, transmission] or array of effective freqs
                 if the passbands read are monofrequency.
        """
        bandint_freqs = []
        for expc, nu_ghz, bp in self.external_bandpass:
            bandpar = "bandint_shift_" + expc
            nub = nu_ghz + params[bandpar]
            if not hasattr(bp, "__len__"):
                bandint_freqs.append(nub)
                bandint_freqs = np.asarray(bandint_freqs)
                self.log.info("bandpass is delta function, no band integration performed")
            else:
                trans_norm = np.trapz(bp * _cmb2bb(nub), nub)
                trans = bp / trans_norm * _cmb2bb(nub)
                bandint_freqs.append([nub, trans])
        
        return bandint_freqs

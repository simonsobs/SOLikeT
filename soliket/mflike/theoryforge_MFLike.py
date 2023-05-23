"""
**This class will be heavily modified with the updated version of mflike**
"""

import numpy as np
import os
from typing import Optional

from cobaya.theory import Theory
from cobaya.tools import are_different_params_lists
from cobaya.log import LoggedError


class TheoryForge_MFLike(Theory):

    # attributes set from .yaml
    data_folder: Optional[str]
    freqs: str
    spectra: dict
    band_integration: dict
    systematics_template: dict

    def initialize(self):

        self.lmin = self.spectra["lmin"]
        self.lmax = self.spectra["lmax"]
        self.ell = np.arange(self.lmin, self.lmax + 1)


        # State requisites to the theory code
        # Which lmax for theory CMB
        # Note this must be greater than lmax above to avoid approx errors
        self.lmax_boltzmann = 9000

        # Which lmax for theory FG
        # This can be larger than lmax boltzmann
        self.lmax_fg = 9000

        # Which spectra to consider
        self.requested_cls = self.spectra["polarizations"]

        # Set lmax for theory CMB requirements
        self.lcuts = {k: self.lmax_boltzmann for k in self.requested_cls}

        self.expected_params_nuis = ["calT_93", "calE_93",
                                     "calT_145", "calE_145",
                                     "calT_225", "calE_225",
                                     "calG_all",
                                     "alpha_93", "alpha_145", "alpha_225",
                                     ]

        # Initialize template for marginalization, if needed
        if self.systematics_template["has_file"]:
            self._init_template_from_file()


    def initialize_with_params(self):
        # Check that the parameters are the right ones
        differences = are_different_params_lists(
            self.input_params, self.expected_params_nuis,
            name_A="given", name_B="expected")
        if differences:
            raise LoggedError(
                self.log, "Configuration error in parameters: %r.",
                differences)

    def must_provide(self, **requirements):
        # cmbfg_dict is required by mflike
        # and requires some params to be computed
        # Assign required params from mflike
        # otherwise use default values
        if "cmbfg_dict" in requirements:
            req = requirements["cmbfg_dict"]
            self.ell = req.get("ell", self.ell)
            self.requested_cls = req.get("requested_cls", self.requested_cls)
            self.lcuts = req.get("lcuts", self.lcuts)
            self.freqs = req.get("freqs", self.freqs)

        # theoryforge requires Cl from boltzmann solver
        # and fg_dict from Foreground theory component
        # Both requirements require some params to be computed
        # Passing those from theoryforge
        reqs = dict()
        # Be sure that CMB is computed at lmax > lmax_data (lcuts from mflike here)
        reqs["Cl"] = {k: max(c, self.lmax_boltzmann + 1) for k, c in self.lcuts.items()}
        reqs["fg_dict"] = {"requested_cls": self.requested_cls,
                           "ell": np.arange(max(self.ell[-1], self.lmax_fg + 1)),
                           "freqs": self.freqs}
        return reqs

    def get_cmb_theory(self, **params):
        return self.provider.get_Cl(ell_factor=True)

    def get_foreground_theory(self, **params):
        return self.provider.get_fg_dict()

    def calculate(self, state, want_derived=False, **params_values_dict):
        Dls = self.get_cmb_theory(**params_values_dict)
        params_values_nocosmo = {k: params_values_dict[k] for k in (
            self.expected_params_nuis)}
        fg_dict = self.get_foreground_theory(**params_values_nocosmo)
        state["cmbfg_dict"] = self.get_modified_theory(Dls,
            fg_dict, **params_values_nocosmo)

    def get_cmbfg_dict(self):
        return self.current_state["cmbfg_dict"]

    def get_modified_theory(self, Dls, fg_dict, **params):
        """
        Takes the theory :math:`D_{\ell}`, sums it to the total 
        foreground power spectrum (possibly computed with bandpass 
        shift and bandpass integration) and applies calibration, 
        polarization angles rotation and systematic templates.

        :param Dls: CMB theory spectra
        :param fg_dict: total foreground spectra, provided by 
                        ``soliket.Foreground``
        :param *params: dictionary of nuisance and foregrounds parameters

        :return: the CMB+foregrounds :math:`D_{\ell}` dictionary, 
                 modulated by systematics
        """
        self.Dls = Dls

        nuis_params = {k: params[k] for k in self.expected_params_nuis}

        cmbfg_dict = {}
        # Sum CMB and FGs
        for f1 in self.freqs:
            for f2 in self.freqs:
                for s in self.requested_cls:
                    cmbfg_dict[s, f1, f2] = (self.Dls[s][self.ell] +
                        fg_dict[s, 'all', f1, f2][self.ell])

        # Apply alm based calibration factors
        cmbfg_dict = self._get_calibrated_spectra(cmbfg_dict, **nuis_params)

        # Introduce spectra rotations
        cmbfg_dict = self._get_rotated_spectra(cmbfg_dict, **nuis_params)

        # Introduce templates of systematics from file, if needed
        if self.systematics_template['has_file']:
            cmbfg_dict = self._get_template_from_file(cmbfg_dict, **nuis_params)


        return cmbfg_dict


    def _get_calibrated_spectra(self, dls_dict, **nuis_params):
        r"""
        Calibrates the spectra through calibration factors at 
        the map level:

        .. math::
        
           D^{{\rm cal}, TT, \nu_1 \nu_2}_{\ell} &= {\rm cal}_{G}\, 
           {\rm cal}^{\nu_1}_{\rm T}\,
           {\rm cal}^{\nu_2}_{\rm T}\, D^{TT, \nu_1 \nu_2}_{\ell} 

           D^{{\rm cal}, TE, \nu_1 \nu_2}_{\ell} &= {\rm cal}_{G}\,
           {\rm cal}^{\nu_1}_{\rm T}\,
           {\rm cal}^{\nu_2}_{\rm E}\, D^{TT, \nu_1 \nu_2}_{\ell} 
        
           D^{{\rm cal}, EE, \nu_1 \nu_2}_{\ell} &= {\rm cal}_{G}\,
           {\rm cal}^{\nu_1}_{\rm E}\,
           {\rm cal}^{\nu_2}_{\rm E}\, D^{EE, \nu_1 \nu_2}_{\ell}

        It uses the ``syslibrary.syslib_mflike.Calibration_alm`` function.

        :param dls_dict: the CMB+foregrounds :math:`D_{\ell}` dictionary
        :param *nuis_params: dictionary of nuisance parameters

        :return: dictionary of calibrated CMB+foregrounds :math:`D_{\ell}`
        """
        from syslibrary import syslib_mflike as syl

        cal_pars = {}
        if "tt" in self.requested_cls or "te" in self.requested_cls:
            cal_pars["tt"] = (nuis_params["calG_all"] *
                              np.array([nuis_params['calT_' + str(fr)] for
                                        fr in self.freqs]))

        if "ee" in self.requested_cls or "te" in self.requested_cls:
            cal_pars["ee"] = (nuis_params["calG_all"] *
                              np.array([nuis_params['calE_' + str(fr)] for
                                        fr in self.freqs]))

        calib = syl.Calibration_alm(ell=self.ell, spectra=dls_dict)

        return calib(cal1=cal_pars, cal2=cal_pars, nu=self.freqs)

###########################################################################
# This part deals with rotation of spectra
# Each freq {freq1,freq2,...,freqn} gets a rotation angle alpha_93, alpha_145, etc..
###########################################################################

    def _get_rotated_spectra(self, dls_dict, **nuis_params):
        r"""
        Rotates the polarization spectra through polarization angles:
        
        .. math::
        
           D^{{\rm rot}, TE, \nu_1 \nu_2}_{\ell} &= \cos(\alpha^{\nu_2})
           D^{TE, \nu_1 \nu_2}_{\ell} 

           D^{{\rm rot}, EE, \nu_1 \nu_2}_{\ell} &= \cos(\alpha^{\nu_1})
           \cos(\alpha^{\nu_2}) D^{EE, \nu_1 \nu_2}_{\ell}

        It uses the ``syslibrary.syslib_mflike.Rotation_alm`` function.

        :param dls_dict: the CMB+foregrounds :math:`D_{\ell}` dictionary
        :param *nuis_params: dictionary of nuisance parameters

        :return: dictionary of rotated CMB+foregrounds :math:`D_{\ell}`
        """
        from syslibrary import syslib_mflike as syl

        rot_pars = [nuis_params['alpha_' + str(fr)] for fr in self.freqs]

        rot = syl.Rotation_alm(ell=self.ell, spectra=dls_dict, cls=self.requested_cls)

        return rot(rot_pars, nu=self.freqs)

###########################################################################
# This part deals with template marginalization
# A dictionary of template dls is read from yaml (likely to be not efficient)
# then rescaled and added to theory dls
###########################################################################

    # Initializes the systematics templates
    # This is slow, but should be done only once
    def _init_template_from_file(self):
        """
        Reads the systematics template from file, using
        the ``syslibrary.syslib_mflike.ReadTemplateFromFile``
        function.
        """
        from syslibrary import syslib_mflike as syl

        # decide where to store systematics template.
        # Currently stored inside syslibrary package
        templ_from_file = \
                syl.ReadTemplateFromFile(rootname=self.systematics_template["rootname"])
        self.dltempl_from_file = templ_from_file(ell=self.ell)

    def _get_template_from_file(self, dls_dict, **nuis_params):
        """
        Adds the systematics template, modulated by ``nuis_params['templ_freq']``
        parameters, to the :math:`D_{\ell}`.

        :param dls_dict: the CMB+foregrounds :math:`D_{\ell}` dictionary
        :param *nuis_params: dictionary of nuisance parameters

        :return: dictionary of CMB+foregrounds :math:`D_{\ell}`
                 with systematics templates
        """
        # templ_pars=[nuis_params['templ_'+str(fr)] for fr in self.freqs]
        # templ_pars currently hard-coded
        # but ideally should be passed as input nuisance
        templ_pars = {cls: np.zeros((len(self.freqs), len(self.freqs)))
                      for cls in self.requested_cls}

        for cls in self.requested_cls:
            for i1, f1 in enumerate(self.freqs):
                for i2, f2 in enumerate(self.freqs):
                    dls_dict[cls, f1, f2] += (templ_pars[cls][i1][i2] *
                                              self.dltempl_from_file[cls, f1, f2])

        return dls_dict

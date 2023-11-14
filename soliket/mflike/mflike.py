r"""
.. module:: mflike

:Synopsis: Multi frequency likelihood for TTTEEE CMB power spectra for Simons Observatory
:Authors: Thibaut Louis, Xavier Garrido, Max Abitbol,
          Erminia Calabrese, Antony Lewis, David Alonso.

MFLike is a multi frequency likelihood code interfaced with the Cobaya 
sampler and a theory Boltzmann code such as CAMB, CLASS or Cosmopower.
The ``MFLike`` likelihood class reads the data file (in ``sacc`` format) 
and all the settings 
for the MCMC run (such as file paths, :math:`\ell` ranges, experiments 
and frequencies to be used, parameters priors...)
from the ``MFLike.yaml`` file. 

The theory :math:`C_{\ell}` are then summed to the (possibly frequency 
integrated) foreground power spectra and modified by systematic effects 
in the ``TheoryForge_MFLike`` class. The foreground power spectra are 
computed by the ``soliket.Foreground`` class, while the bandpasses from 
the ``soliket.BandPass`` one; the ``Foreground`` class is required by 
``TheoryForge_MFLike``, while ``BandPass`` is requires by ``Foreground``.
This is a scheme of how ``MFLike`` and ``TheoryForge_MFLike`` are interfaced:

.. image:: images/mflike_scheme.png
   :width: 400
"""
import os
from typing import Optional

import numpy as np
from cobaya.conventions import data_path, packages_path_input
from cobaya.likelihoods.base_classes import InstallableLikelihood
from cobaya.log import LoggedError
from cobaya.tools import are_different_params_lists
from cobaya.typing import InfoDict

from ..gaussian import GaussianData, GaussianLikelihood


class MFLike(GaussianLikelihood, InstallableLikelihood):
    _url = "https://portal.nersc.gov/cfs/sobs/users/MFLike_data"
    _release = "v0.8" 
    install_options = {"download_url": "{}/{}.tar.gz".format(_url, _release)}

    # attributes set from .yaml
    input_file: Optional[str]
    cov_Bbl_file: Optional[str]
    lmax_theory: Optional[int]
    data: InfoDict
    defaults: InfoDict
    data_folder: str

    def initialize(self):
        # Set default values to data member not initialized via yaml file
        self.l_bpws = None
        self.spec_meta = []


        # Set path to data
        if ((not getattr(self, "path", None)) and
                (not getattr(self, "packages_path", None))):
            raise LoggedError(self.log,
                              "No path given to MFLike data. "
                              "Set the likelihood property "
                              "'path' or 'packages_path'"
                              )
        # If no path specified, use the modules path
        data_file_path = os.path.normpath(getattr(self, "path", None) or
                                          os.path.join(self.packages_path,
                                                       "data"))

        self.data_folder = os.path.join(data_file_path, self.data_folder)
        if not os.path.exists(self.data_folder):
            if not getattr(self, "path", None):
                self.install(path=self.packages_path)
            else:
                raise LoggedError(
                    self.log,
                    "The 'data_folder' directory does not exist. "\
                    "Check the given path [%s].",
                    self.data_folder,
                )

        self.requested_cls = [p.lower() for p in self.defaults["polarizations"]]
        for x in ["et", "eb", "bt"]:
            if x in self.requested_cls:
                self.requested_cls.remove(x)

        # Read data
        self.prepare_data()
        self.lmax_theory = self.lmax_theory or 9000
        self.log.debug(f"Maximum multipole value: {self.lmax_theory}")
        
        self.log.info("Initialized!")


    def get_requirements(self):
        r"""
        Passes the fields ``ell``, ``requested_cls``, ``lcuts``, 
        ``exp_ch`` (list of array names) and ``bands`` 
        (dictionary of ``exp_ch`` and the corresponding frequency
        and passbands) inside the dictionary ``requirements["cmbfg_dict"]``.

        :return: the dictionary ``requirements["cmbfg_dict"]``
        """
        # mflike requires cmbfg_dict from theoryforge
        # cmbfg_dict requires some params to be computed
        reqs = dict()
        reqs["cmbfg_dict"] = {"ell": self.l_bpws,
                              "requested_cls": self.requested_cls,
                              "lcuts": self.lcuts,
                              "exp_ch": self.experiments, 
                              "bands": self.bands}
        return reqs

    def _get_theory(self, **params_values):
        cmbfg_dict = self.provider.get_cmbfg_dict()
        return self._get_power_spectra(cmbfg_dict)

    def logp(self, **params_values):
        cmbfg_dict = self.provider.get_cmbfg_dict()
        return self.loglike(cmbfg_dict)

    def loglike(self, cmbfg_dict):
        """
        Computes the gaussian log-likelihood

        :param cmbfg_dict: the dictionary of theory + foregrounds
                           :math:`D_{\ell}`

        :return: the exact loglikelihood :math:`\ln \mathcal{L}` 
        """
        ps_vec = self._get_power_spectra(cmbfg_dict)
        delta = self.data_vec - ps_vec
        logp = -0.5 * (delta @ self.inv_cov @ delta)
        logp += self.logp_const
        self.log.debug(
            "Log-likelihood value computed "
            "= {} (Χ² = {})".format(logp, -2 * (logp - self.logp_const)))
        return logp

    def prepare_data(self, verbose=False):
        """
        Reads the sacc data, extracts the data tracers,
        trims the spectra and covariance according to the ell scales
        set in the input file. It stores the ell vector, the deta vector
        and the covariance in a GaussianData object.
        If ``verbose=True``, it plots the tracer names, the spectrum name,
        the shape of the indices array, lmin, lmax.
        """
        import sacc
        data = self.data
        # Read data
        input_fname = os.path.join(self.data_folder, self.input_file)
        s = sacc.Sacc.load_fits(input_fname)

        # Read extra file containing covariance and windows if needed.
        cbbl_extra = False
        s_b = s
        if self.cov_Bbl_file:
            if self.cov_Bbl_file != self.input_file:
                cov_Bbl_fname = os.path.join(self.data_folder,
                                             self.cov_Bbl_file)
                s_b = sacc.Sacc.load_fits(cov_Bbl_fname)
                cbbl_extra = True

        try:
            default_cuts = self.defaults
        except AttributeError:
            raise KeyError("You must provide a list of default cuts")

        # Translation betwen TEB and sacc C_ell types
        pol_dict = {"T": "0",
                    "E": "e",
                    "B": "b"}
        ppol_dict = {"TT": "tt",
                     "EE": "ee",
                     "TE": "te",
                     "ET": "te",
                     "BB": "bb",
                     "EB": "eb",
                     "BE": "eb",
                     "TB": "tb",
                     "BT": "tb",
                     "BB": "bb"}

        def get_cl_meta(spec):
            """
            Lower-level function of `prepare_data`.
            For each of the entries of the `spectra` section of the
            yaml file, extracts the relevant information: channel,
            polarization combinations, scale cuts and
            whether TE should be symmetrized.

            :param spec: the dictionary ``data["spectra"]`` 
            """
            # Experiments/frequencies
            exp_1, exp_2 = spec["experiments"]
            # Read off polarization channel combinations
            pols = spec.get("polarizations",
                            default_cuts["polarizations"]).copy()
            # Read off scale cuts
            scls = spec.get("scales",
                            default_cuts["scales"]).copy()

            # For the same two channels, do not include ET and TE, only TE
            if (exp_1 == exp_2):
                if "ET" in pols:
                    pols.remove("ET")
                    if "TE" not in pols:
                        pols.append("TE")
                        scls["TE"] = scls["ET"]
                symm = False
            else:
                # Symmetrization
                if ("TE" in pols) and ("ET" in pols):
                    symm = spec.get("symmetrize",
                                    default_cuts["symmetrize"])
                else:
                    symm = False

            return exp_1, exp_2, pols, scls, symm

        def get_sacc_names(pol, exp_1, exp_2):
            """
            Lower-level function of `prepare_data`.
            Translates the polarization combination and channel 
            name of a given entry in the `spectra`
            part of the input yaml file into the names expected
            in the SACC files.

            :param pol: temperature or polarization fields, i.e. 'TT', 'TE'
            :param exp_1: experiment of map 1
            :param exp_2: experiment of map 2

            :return: tracer name 1, tracer name 2, string for :math:`C_{\ell}`
                     type
            """
            tname_1 = exp_1
            tname_2 = exp_2
            p1, p2 = pol
            if p1 in ["E", "B"]:
                tname_1 += "_s2"
            else:
                tname_1 += "_s0"
            if p2 in ["E", "B"]:
                tname_2 += "_s2"
            else:
                tname_2 += "_s0"

            if p2 == "T":
                dtype = "cl_" + pol_dict[p2] + pol_dict[p1]
            else:
                dtype = "cl_" + pol_dict[p1] + pol_dict[p2]
            return tname_1, tname_2, dtype

        # First we trim the SACC file so it only contains
        # the parts of the data we care about.
        # Indices to be kept
        indices = []
        indices_b = []
        # Length of the final data vector
        len_compressed = 0
        for spectrum in data["spectra"]:
            (exp_1, exp_2, pols, scls, symm) = get_cl_meta(spectrum)
            for pol in pols:
                tname_1, tname_2, dtype = get_sacc_names(pol, exp_1, exp_2)
                lmin, lmax = scls[pol]
                ind = s.indices(dtype,  # Power spectrum type
                                (tname_1, tname_2),  # Channel combinations
                                ell__gt=lmin, ell__lt=lmax)  # Scale cuts
                indices += list(ind)

                # Note that data in the cov_Bbl file may be in different order.
                if cbbl_extra:
                    ind_b = s_b.indices(dtype,
                                        (tname_1, tname_2),
                                        ell__gt=lmin, ell__lt=lmax)
                    indices_b += list(ind_b)

                if symm and pol == "ET":
                    pass
                else:
                    len_compressed += ind.size

                
                self.log.debug(f"{tname_1} {tname_2} {dtype} {ind.shape} {lmin} {lmax}")
            

        # Get rid of all the unselected power spectra.
        # Sacc takes care of performing the same cuts in the
        # covariance matrix, window functions etc.
        s.keep_indices(np.array(indices))
        if cbbl_extra:
            s_b.keep_indices(np.array(indices_b))

        # Now create metadata for each spectrum
        len_full = s.mean.size
        # These are the matrices we'll use to compress the data if
        # `symmetrize` is true.
        # Note that a lot of the complication in this function is caused by the
        # symmetrization option, for which SACC doesn't have native support.
        mat_compress = np.zeros([len_compressed, len_full])
        mat_compress_b = np.zeros([len_compressed, len_full])
        
        self.lcuts = {k: c[1] for k, c in default_cuts["scales"].items()}
        index_sofar = 0

        for spectrum in data["spectra"]:
            (exp_1, exp_2, pols, scls, symm) = get_cl_meta(spectrum)
            for k in scls.keys():
                self.lcuts[k] = max(self.lcuts[k], scls[k][1])
            for pol in pols:
                tname_1, tname_2, dtype = get_sacc_names(pol, exp_1, exp_2)
                # The only reason why we need indices is the symmetrization.
                # Otherwise all of this could have been done in the previous
                # loop over data["spectra"].
                ls, cls, ind = s.get_ell_cl(dtype, tname_1, tname_2, return_ind=True)
                if cbbl_extra:
                    ind_b = s_b.indices(dtype,
                                        (tname_1, tname_2))
                    ws = s_b.get_bandpower_windows(ind_b)
                else:
                    ws = s.get_bandpower_windows(ind)

                if self.l_bpws is None:
                    # The assumption here is that bandpower windows
                    # will all be sampled at the same ells.
                    self.l_bpws = ws.values

                # Symmetrize if needed.
                if (pol in ["TE", "ET"]) and symm:
                    pol2 = pol[::-1]
                    pols.remove(pol2)
                    tname_1, tname_2, dtype = get_sacc_names(pol2,
                                                             exp_1, exp_2)
                    ind2 = s.indices(dtype,
                                     (tname_1, tname_2))
                    cls2 = s.get_ell_cl(dtype, tname_1, tname_2)[1]
                    cls = 0.5 * (cls + cls2)

                    for i, (j1, j2) in enumerate(zip(ind, ind2)):
                        mat_compress[index_sofar + i, j1] = 0.5
                        mat_compress[index_sofar + i, j2] = 0.5
                    if cbbl_extra:
                        ind2_b = s_b.indices(dtype,
                                             (tname_1, tname_2))
                        for i, (j1, j2) in enumerate(zip(ind_b, ind2_b)):
                            mat_compress_b[index_sofar + i, j1] = 0.5
                            mat_compress_b[index_sofar + i, j2] = 0.5
                else:
                    for i, j1 in enumerate(ind):
                        mat_compress[index_sofar + i, j1] = 1
                    if cbbl_extra:
                        for i, j1 in enumerate(ind_b):
                            mat_compress_b[index_sofar + i, j1] = 1
                # The fields marked with # below aren't really used, but
                # we store them just in case.
                self.spec_meta.append({"ids": (index_sofar +
                                               np.arange(cls.size,
                                                         dtype=int)),
                                       "pol": ppol_dict[pol],
                                       "hasYX_xsp": pol in ["ET", "BE", "BT"],  # For symm
                                       "t1": exp_1,
                                       "t2": exp_2,
                                       "leff": ls,
                                       "cl_data": cls,
                                       "bpw": ws})
                index_sofar += cls.size
        if not cbbl_extra:
            mat_compress_b = mat_compress
        # Put data and covariance in the right order.
        self.data_vec = np.dot(mat_compress, s.mean)
        self.cov = np.dot(mat_compress_b,
                          s_b.covariance.covmat.dot(mat_compress_b.T))
        self.inv_cov = np.linalg.inv(self.cov)
        self.logp_const = np.log(2 * np.pi) * (-len(self.data_vec) / 2)
        self.logp_const -= 0.5 * np.linalg.slogdet(self.cov)[1]

        self.experiments = data["experiments"]
        self.bands = {
            name: {"nu": tracer.nu, "bandpass": tracer.bandpass}
            for name, tracer in s.tracers.items()
        }

        # Put lcuts in a format that is recognisable by CAMB.
        self.lcuts = {k.lower(): c for k, c in self.lcuts.items()}
        if "et" in self.lcuts:
            del self.lcuts["et"]

        ell_vec = np.zeros_like(self.data_vec)
        for m in self.spec_meta:
            i = m["ids"]
            ell_vec[i] = m["leff"]
        self.ell_vec = ell_vec

        self.data = GaussianData("mflike", self.ell_vec, self.data_vec, self.cov)

    def _get_power_spectra(self, cmbfg):
        """
        Get :math:`D_{\ell}` from the theory component
        already modified by ``theoryforge_MFLike``

        :param cmbfg: the dictionary of theory+foreground :math:`D_{\ell}`

        :return: the binned data vector
        """
        ps_vec = np.zeros_like(self.data_vec)
        DlsObs = dict()
        # Note we rescale l_bpws because cmbfg spectra start from l=2
        ell = self.l_bpws - 2

        for m in self.spec_meta:
            p = m["pol"]
            i = m["ids"]
            w = m["bpw"].weight.T

            if p in ['tt', 'ee', 'bb']:
                DlsObs[p,  m['t1'], m['t2']] = cmbfg[p, m['t1'], m['t2']][ell]
            else:  # ['te','tb','eb']
                if m['hasYX_xsp']:  # not symmetrizing
                    DlsObs[p,  m['t1'], m['t2']] = cmbfg[p, m['t2'], m['t1']][ell]
                else:
                    DlsObs[p,  m['t1'], m['t2']] = cmbfg[p, m['t1'], m['t2']][ell]
#
                if self.defaults['symmetrize']:  # we average TE and ET (as for data)
                    DlsObs[p,  m['t1'], m['t2']] += cmbfg[p, m['t2'], m['t1']][ell]
                    DlsObs[p,  m['t1'], m['t2']] *= 0.5

            clt = w @ DlsObs[p, m["t1"], m["t2"]]
            ps_vec[i] = clt

        return ps_vec


class TestMFLike(MFLike):

    _url = "https://portal.nersc.gov/cfs/sobs/users/MFLike_data"
    filename = "v0.1_test"
    install_options = {"download_url": f"{_url}/{filename}.tar.gz"}

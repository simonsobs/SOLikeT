"""
.. module:: mflike

:Synopsis: Definition of simplistic likelihood for Simons Observatory
:Authors: Thibaut Louis, Xavier Garrido, Max Abitbol,
          Erminia Calabrese, Antony Lewis, David Alonso.

"""
import os
from typing import Optional
import numpy as np
from cobaya.conventions import _packages_path
from cobaya.likelihoods._base_classes import _InstallableLikelihood
from cobaya.log import LoggedError
from cobaya.tools import are_different_params_lists
from cobaya.theory import HelperTheory

from .gaussian import GaussianData, GaussianLikelihood
from .ps import PSLikelihood


class MFLike(GaussianLikelihood, _InstallableLikelihood):
    _url = "https://portal.nersc.gov/cfs/sobs/users/MFLike_data"
    _release = "v0.6"
    install_options = {"download_url": "{}/{}.tar.gz".format(_url, _release)}

    # attributes set from .yaml
    input_file: Optional[str]
    cov_Bbl_file: Optional[str]
    data: dict
    defaults: dict
    foregrounds: dict

    def initialize(self):
        self.log.info("Initialising.")
        # Set path to data
        if (not getattr(self, "path", None)) and (not getattr(self, _packages_path, None)):
            raise LoggedError(
                self.log,
                "No path given to MFLike data. "
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

        # Read data
        self.prepare_data()

        # State requisites to the theory code
        self.requested_cls = ["tt", "te", "ee"]

        self.expected_params = ["a_tSZ", "a_kSZ", "a_p", "beta_p",
                                "a_c", "beta_c", "n_CIBC", "a_s", "T_d"]

    # def get_helper_theories(self):
    #     """
    #     Foreground model is a helper theory
    #     """
    #     self._foreground = Foreground(
    #         self, "foreground", dict(stop_at_error=self.stop_at_error), timing=self.timer
    #     )

    #     # setattr(self._camb_transfers, _requires, self._transfer_requires)
    #     return {"foreground": self._foreground}

    def initialize_with_params(self):
        # Check that the parameters are the right ones
        differences = are_different_params_lists(
            self.input_params, self.expected_params,
            name_A="given", name_B="expected")
        if differences:
            raise LoggedError(
                self.log, "Configuration error in parameters: %r.",
                differences)

    def get_requirements(self):
        reqs = dict(Cl={k: max(c, 9000) for k, c in self.lcuts.items()})
        reqs['bandint_freqs'] = {}
        # reqs["foreground_model"] = {}
        return reqs

    def _get_theory(self, **params_values):
        cl = self.provider.get_Cl(ell_factor=True)
        return self._get_power_spectra(cl, **params_values)

    def prepare_data(self, verbose=False):
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
            raise KeyError('You must provide a list of default cuts')

        # Translation betwen TEB and sacc C_ell types
        pol_dict = {'T': '0',
                    'E': 'e',
                    'B': 'b'}
        ppol_dict = {'TT': 'tt',
                     'EE': 'ee',
                     'TE': 'te',
                     'ET': 'te',
                     'BB': 'bb',
                     'EB': 'eb',
                     'BE': 'eb',
                     'TB': 'tb',
                     'BT': 'tb',
                     'BB': 'bb'}

        def xp_nu(xp, nu):
            return xp + '_' + str(nu)

        def get_cl_meta(spec):
            # For each of the entries of the `spectra` section of the
            # yaml file, extract the relevant information: experiments,
            # frequencies, polarization combinations, scale cuts and
            # whether TE should be symmetrized.
            # Experiments/frequencies
            exp_1, exp_2 = spec['experiments']
            freq_1, freq_2 = spec['frequencies']
            # Read off polarization channel combinations
            pols = spectrum.get('polarizations',
                                default_cuts['polarizations']).copy()
            # Read off scale cuts
            scls = spectrum.get('scales',
                                default_cuts['scales']).copy()

            # For the same two channels, do not include ET and TE, only TE
            if (exp_1 == exp_2) and (freq_1 == freq_2):
                if 'ET' in pols:
                    pols.remove('ET')
                    if 'TE' not in pols:
                        pols.append('TE')
                        scls['TE'] = scls['ET']
                symm = False
            else:
                # Symmetrization
                if ('TE' in pols) and ('ET' in pols):
                    symm = spectrum.get('symmetrize',
                                        default_cuts['symmetrize'])
                else:
                    symm = False

            return exp_1, exp_2, freq_1, freq_2, pols, scls, symm

        def get_sacc_names(pol, exp_1, exp_2, freq_1, freq_2):
            # Translate the polarization combination, experiment
            # and frequency names of a given entry in the `spectra`
            # part of the input yaml file into the names expected
            # in the SACC files.
            p1, p2 = pol
            tname_1 = xp_nu(exp_1, freq_1)
            tname_2 = xp_nu(exp_2, freq_2)
            if p1 in ['E', 'B']:
                tname_1 += '_s2'
            else:
                tname_1 += '_s0'
            if p2 in ['E', 'B']:
                tname_2 += '_s2'
            else:
                tname_2 += '_s0'

            if p2 == 'T':
                dtype = 'cl_' + pol_dict[p2] + pol_dict[p1]
            else:
                dtype = 'cl_' + pol_dict[p1] + pol_dict[p2]
            return tname_1, tname_2, dtype

        # First we trim the SACC file so it only contains
        # the parts of the data we care about.
        # Indices to be kept
        indices = []
        indices_b = []
        # Length of the final data vector
        len_compressed = 0
        for spectrum in data['spectra']:
            (exp_1, exp_2, freq_1, freq_2,
             pols, scls, symm) = get_cl_meta(spectrum)
            for pol in pols:
                tname_1, tname_2, dtype = get_sacc_names(pol, exp_1, exp_2,
                                                         freq_1, freq_2)
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

                if symm and pol == 'ET':
                    pass
                else:
                    len_compressed += ind.size

                if verbose:
                    print(tname_1, tname_2, dtype, ind.shape, lmin, lmax)

        # Get rid of all the unselected power spectra.
        # Sacc takes care of performing the same cuts in the
        # covariance matrix, window functions etc.
        s.keep_indices(np.array(indices))
        if cbbl_extra:
            s_b.keep_indices(np.array(indices_b))

        # Now create metadata for each spectrum
        self.spec_meta = []
        len_full = s.mean.size
        # These are the matrices we'll use to compress the data if
        # `symmetrize` is true.
        # Note that a lot of the complication in this function is caused by the
        # symmetrization option, for which SACC doesn't have native support.
        mat_compress = np.zeros([len_compressed, len_full])
        mat_compress_b = np.zeros([len_compressed, len_full])
        bands = {}
        self.lcuts = {k: c[1] for k, c in default_cuts['scales'].items()}
        index_sofar = 0

        self.l_bpws = None
        for spectrum in data['spectra']:
            (exp_1, exp_2, freq_1, freq_2,
             pols, scls, symm) = get_cl_meta(spectrum)
            bands[xp_nu(exp_1, freq_1)] = freq_1
            bands[xp_nu(exp_2, freq_2)] = freq_2
            for k in scls.keys():
                self.lcuts[k] = max(self.lcuts[k], scls[k][1])
            for pol in pols:
                tname_1, tname_2, dtype = get_sacc_names(pol, exp_1, exp_2,
                                                         freq_1, freq_2)
                # The only reason why we need indices is the symmetrization.
                # Otherwise all of this could have been done in the previous
                # loop over data['spectra'].
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
                if (pol in ['TE', 'ET']) and symm:
                    pol2 = pol[::-1]
                    pols.remove(pol2)
                    tname_1, tname_2, dtype = get_sacc_names(pol2,
                                                             exp_1, exp_2,
                                                             freq_1, freq_2)
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
                self.spec_meta.append({'ids': (index_sofar +
                                               np.arange(cls.size,
                                                         dtype=int)),
                                       'pol': ppol_dict[pol],
                                       't1': xp_nu(exp_1, freq_1),  #
                                       't2': xp_nu(exp_2, freq_2),  #
                                       'nu1': freq_1,
                                       'nu2': freq_2,
                                       'leff': ls,  #
                                       'cl_data': cls,  #
                                       'bpw': ws})
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

        # TODO: we should actually be using bandpass integration
        self.bands = sorted(bands)
        self.freqs = np.array([bands[b] for b in self.bands])

        # Put lcuts in a format that is recognisable by CAMB.
        self.lcuts = {k.lower(): c for k, c in self.lcuts.items()}
        if 'et' in self.lcuts:
            del self.lcuts['et']

        ell_vec = np.zeros_like(self.data_vec)
        for m in self.spec_meta:
            i = m["ids"]
            ell_vec[i] = m["leff"]
        self.ell_vec = ell_vec

        self.data = GaussianData("mflike", self.ell_vec, self.data_vec, self.cov)

    def loglike(self, cl, **params_values):
        ps_vec = self._get_power_spectra(cl, **params_values)
        delta = self.data_vec - ps_vec
        logp = -0.5 * (delta @ self.inv_cov @ delta)
        logp += self.logp_const
        self.log.debug(
            "Log-likelihood value computed "
            "= {} (Χ² = {})".format(logp, -2 * (logp - self.logp_const)))
        return logp

    def _get_power_spectra(self, cl, **params_values):
        # Get Cl's from the theory code
        Dls = {s: cl[s][self.l_bpws] for s, _ in self.lcuts.items()}

        # Get new foreground model given its nuisance parameters
        fg_model = self._get_foreground_model(
            {k: params_values[k] for k in self.expected_params})

        ps_vec = np.zeros_like(self.data_vec)
        for m in self.spec_meta:
            p = m['pol']
            i = m['ids']
            w = m['bpw'].weight.T
            clt = np.dot(w, Dls[p] + fg_model[p, 'all', m['nu1'], m['nu2']])
            ps_vec[i] = clt

        return ps_vec

    def _get_foreground_model(self, fg_params):
        bandint_freqs = self.provider.get_result('bandint_freqs')

        return get_foreground_model(fg_params=fg_params,
                                    fg_model=self.foregrounds,
                                    frequencies=self.freqs,
                                    bandint_freqs=bandint_freqs, 
                                    ell=self.l_bpws,
                                    requested_cls=self.requested_cls)
        

# Standalone function to return the foregroung model
# given the nuisance parameters
def get_foreground_model(fg_params, fg_model,
                         frequencies, bandint_freqs, ell,
                         requested_cls=["tt", "te", "ee"]):
    normalisation = fg_model["normalisation"]
    nu_0 = normalisation["nu_0"]
    ell_0 = normalisation["ell_0"]

    from fgspectra import cross as fgc
    from fgspectra import frequency as fgf
    from fgspectra import power as fgp

    # We don't seem to be using this
    # cirrus = fgc.FactorizedCrossSpectrum(fgf.PowerLaw(), fgp.PowerLaw())
    ksz = fgc.FactorizedCrossSpectrum(fgf.ConstantSED(), fgp.kSZ_bat())
    cibp = fgc.FactorizedCrossSpectrum(fgf.ModifiedBlackBody(), fgp.PowerLaw())
    radio = fgc.FactorizedCrossSpectrum(fgf.PowerLaw(), fgp.PowerLaw())
    tsz = fgc.FactorizedCrossSpectrum(fgf.ThermalSZ(), fgp.tSZ_150_bat())
    cibc = fgc.FactorizedCrossSpectrum(fgf.CIB(), fgp.PowerLaw())

    # Make sure to pass a numpy array to fgspectra
    if not isinstance(frequencies, np.ndarray):
        frequencies = np.array(frequencies)

    model = {}
    model["tt", "kSZ"] = fg_params["a_kSZ"] * ksz(
        {"nu": bandint_freqs},
        {"ell": ell, "ell_0": ell_0})
    model["tt", "cibp"] = fg_params["a_p"] * cibp(
        {"nu": bandint_freqs, "nu_0": nu_0,
         "temp": fg_params["T_d"], "beta": fg_params["beta_p"]},
        {"ell": ell, "ell_0": ell_0, "alpha": 2})
    model["tt", "radio"] = fg_params["a_s"] * radio(
        {"nu": bandint_freqs, "nu_0": nu_0, "beta": -0.5 - 2},
        {"ell": ell, "ell_0": ell_0, "alpha": 2})
    model["tt", "tSZ"] = fg_params["a_tSZ"] * tsz(
        {"nu": bandint_freqs, "nu_0": nu_0},
        {"ell": ell, "ell_0": ell_0})
    model["tt", "cibc"] = fg_params["a_c"] * cibc(
        {"nu": bandint_freqs, "nu_0": nu_0,
         "temp": fg_params["T_d"], "beta": fg_params["beta_c"]},
        {"ell": ell, "ell_0": ell_0, "alpha": 2 - fg_params["n_CIBC"]})

    components = fg_model["components"]
    component_list = {s: components[s] for s in requested_cls}
    fg_dict = {}
    for c1, f1 in enumerate(frequencies):
        for c2, f2 in enumerate(frequencies):
            for s in requested_cls:
                fg_dict[s, "all", f1, f2] = np.zeros(len(ell))
                for comp in component_list[s]:
                    fg_dict[s, comp, f1, f2] = model[s, comp][c1, c2]
                    fg_dict[s, "all", f1, f2] += fg_dict[s, comp, f1, f2]

    return fg_dict

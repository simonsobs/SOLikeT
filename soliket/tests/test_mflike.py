"""
Make sure that this returns the same result as original mflike.MFLike from LAT_MFlike repo
"""
import os
import tempfile
import unittest
from distutils.version import LooseVersion

import camb
import mflike  # noqa
import soliket  # noqa

packages_path = os.environ.get("COBAYA_PACKAGES_PATH") or os.path.join(
    tempfile.gettempdir(), "LAT_packages"
)

cosmo_params = {
    "cosmomc_theta": 0.0104085,
    "As": 2.0989031673191437e-09,
    "ombh2": 0.02237,
    "omch2": 0.1200,
    "ns": 0.9649,
    "Alens": 1.0,
    "tau": 0.0544,
}

nuisance_params = {
    "a_tSZ": 3.3044404448917724,
    "a_kSZ": 1.6646620740058649,
    "a_p": 6.912474322461401,
    "beta_p": 2.077474196171309,
    "a_c": 4.88617700670901,
    "beta_c": 2.2030316332596014,
    "a_s": 3.099214100532393,
    "T_d": 9.60,
    "a_gtt": 0,
    "a_gte": 0,
    "a_gee": 0,
    "a_psee": 0,
    "a_pste": 0,
    "xi": 0,
    "bandint_shift_93": 0,
    "bandint_shift_145": 0,
    "bandint_shift_225": 0,
    "calT_93": 1,
    "calE_93": 1,
    "calT_145": 1,
    "calE_145": 1,
    "calT_225": 1,
    "calE_225": 1,
    "calG_all": 1,
    "alpha_93": 0,
    "alpha_145": 0,
    "alpha_225": 0,
}


if LooseVersion(camb.__version__) < LooseVersion('1.3'):
    chi2s = {"tt": 1384.5669,
             "te": 1400.2760,
             "ee": 1428.7597,
             "tt-te-et-ee": 2412.9275}
else:
    chi2s = {"tt": 737.8571537677649,
             "te-et": 998.2730263280033,
             "ee": 716.4015196388742,
             "tt-te-et-ee": 2459.7250}

pre = "data_sacc_"


class MFLikeTest(unittest.TestCase):
    orig = False

    def setUp(self):
        from cobaya.install import install

        install({"likelihood": {"mflike.MFLike": None}}, path=packages_path, skip_global=True)

    def get_mflike_type(self, as_string=False):
        if self.orig:
            t = "mflike.MFLike"
        else:
            t = "soliket.MFLike"

        if as_string:
            return t
        else:
            return eval(t)

    def test_mflike(self):
        # As of now, there is not a mechanism
        # in soliket to ensure there is .loglike that can be called like this
        # w/out cobaya

        camb_cosmo = cosmo_params.copy()
        camb_cosmo.update({"lmax": 9000, "lens_potential_accuracy": 1})
        pars = camb.set_params(**camb_cosmo)
        results = camb.get_results(pars)
        powers = results.get_cmb_power_spectra(pars, CMB_unit="muK")
        cl_dict = {k: powers["total"][:, v] for k, v in {"tt": 0, "ee": 1, "te": 3}.items()}
        for select, chi2 in chi2s.items():
            MFLike = self.get_mflike_type()

            my_mflike = MFLike(
                {
                    "packages_path": packages_path,
                    "input_file": pre + "00000.fits",
                    "cov_Bbl_file": pre + "w_covar_and_Bbl.fits",
                    "defaults": {
                        "polarizations": select.upper().split("-"),
                        "scales": {
                            "TT": [2, 5000],
                            "TE": [2, 5000],
                            "ET": [2, 5000],
                            "EE": [2, 5000],
                        },
                        "symmetrize": False,
                    },
                }
            )

            loglike = my_mflike.loglike(cl_dict, **nuisance_params)

            self.assertAlmostEqual(-2 * (loglike - my_mflike.logp_const), chi2, 2)

    def test_cobaya(self):
        mflike_type = self.get_mflike_type(as_string=True)

        # params = dict(cosmo_params)
        # params['a_tSZ'] = 3.3

        info = {
            "likelihood": {
                mflike_type: {
                    "input_file": pre + "00000.fits",
                    "cov_Bbl_file": pre + "w_covar_and_Bbl.fits",
                }
            },
            "theory": {"camb": {"extra_args": {"lens_potential_accuracy": 1},
                                "stop_at_error": True}},
            "params": cosmo_params,
            "modules": packages_path,
            "debug": True,
        }
        from cobaya.model import get_model

        model = get_model(info)
        my_mflike = model.likelihood[mflike_type]
        chi2 = -2 * (model.loglikes(nuisance_params)[0] - my_mflike.logp_const)
        self.assertAlmostEqual(chi2[0], chi2s["tt-te-et-ee"], 2)


class MFLikeTestOrig(MFLikeTest):
    orig = True

"""
Make sure that this returns the same result as original mflike.MFLike from LAT_MFlike repo
"""
import os

import camb
import numpy as np
import pytest
from cobaya.tools import resolve_packages_path
from packaging.version import Version

import soliket
from soliket.mflike import TestMFLike

packages_path = resolve_packages_path()

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
    "bandint_shift_LAT_93": 0,
    "bandint_shift_LAT_145": 0,
    "bandint_shift_LAT_225": 0,
    "cal_LAT_93": 1,
    "cal_LAT_145": 1,
    "cal_LAT_225": 1,
    "calT_LAT_93": 1,
    "calE_LAT_93": 1,
    "calT_LAT_145": 1,
    "calE_LAT_145": 1,
    "calT_LAT_225": 1,
    "calE_LAT_225": 1,
    "calG_all": 1,
    "alpha_LAT_93": 0,
    "alpha_LAT_145": 0,
    "alpha_LAT_225": 0,
}


if Version(camb.__version__) >= Version('1.4'):
    chi2s = {"tt": 544.9017,
             "te": 136.6051,
             "ee": 166.1897,
             "tt-te-et-ee": 787.9529}
else:
    chi2s = {"tt": 544.8797,
             "te-et": 151.8197,
             "ee": 166.2835,
             "tt-te-et-ee": 787.9843}

pre = "test_data_sacc_"


class Test_mflike:

    @classmethod
    def setup_class(cls):
        from cobaya.install import install

        install(
            {"likelihood": {"soliket.mflike.TestMFLike": None}},
            path=packages_path,
            skip_global=False,
            force=True,
            debug=True,
            no_set_global=True,
        )

    @pytest.mark.usefixtures("test_cosmology_params")
    def test_mflike(self, test_cosmology_params):

        # As of now, there is not a mechanism
        # in soliket to ensure there is .loglike that can be called like this
        # w/out cobaya

        lmax = 9000
        test_cosmology_params.update({"lmax": lmax, "lens_potential_accuracy": 1})
        pars = camb.set_params(**test_cosmology_params)
        results = camb.get_results(pars)
        powers = results.get_cmb_power_spectra(pars, CMB_unit="muK")
        cl_dict = {k: powers["total"][:, v] for
                   k, v in {"tt": 0, "ee": 1, "te": 3}.items()}


        BP = soliket.BandPass()
        FG = soliket.Foreground()
        TF = soliket.TheoryForge_MFLike()

        bands = TF.bands
        exp_ch = TF.exp_ch

        requested_cls = TF.requested_cls
        BP.bands = bands
        BP.exp_ch = [k.replace("_s0", "") for k in bands.keys()
                          if "_s0" in k]

        bandpass = BP._bandpass_construction(**nuisance_params)

        for select, chi2 in chi2s.items():

            my_mflike = TestMFLike(
                {
                    "external": TestMFLike,
                    "packages_path": packages_path,
                    "data_folder": "TestMFLike",
                    "input_file": pre + "00000.fits",
                    "defaults": {
                        "polarizations": select.upper().split("-"),
                        "scales": {
                            "TT": [2, 179],
                            "TE": [2, 179],
                            "ET": [2, 179],
                            "EE": [2, 179],
                        },
                        "symmetrize": False,
                    },
                }
            )

            ell_cut = my_mflike.l_bpws
            dls_cut = {s: cl_dict[s][ell_cut] for s, _ in my_mflike.lcuts.items()}
            fg_dict = FG._get_foreground_model(requested_cls=requested_cls,
                                                    ell=ell_cut,
                                                    exp_ch=exp_ch,
                                                    bandint_freqs=bandpass,
                                                    **nuisance_params)
            dlobs_dict = TF.get_modified_theory(dls_cut, fg_dict, **nuisance_params)

            loglike = my_mflike.loglike(dlobs_dict)

            assert np.isclose(
                -2 * (loglike - my_mflike.logp_const), chi2, atol=1e-2, rtol=0.0
            )

    @pytest.mark.usefixtures("test_cosmology_params")
    def test_cobaya(self, test_cosmology_params):

        info = {
            "likelihood": {
                "soliket.mflike.TestMFLike": {
                    "datapath": os.path.join(packages_path, "data/TestMFLike"),
                    "data_folder": "TestMFLike",
                    "input_file": pre + "00000.fits",
                    "defaults": {
                        "polarizations": ["TT", "TE", "ET", "EE"],
                        "scales": {
                            "TT": [2, 179],
                            "TE": [2, 179],
                            "ET": [2, 179],
                            "EE": [2, 179],
                        },
                        "symmetrize": False,
                    },
                },
            },
            "theory": {"camb": {"extra_args": {"lens_potential_accuracy": 1},
                                "stop_at_error": True}},
            "params": test_cosmology_params,
            "modules": packages_path,
            "debug": True,
        }

        info["theory"]["soliket.TheoryForge_MFLike"] = {'stop_at_error': True}
        info["theory"]["soliket.Foreground"] = {'stop_at_error': True}
        info["theory"]["soliket.BandPass"] = {'stop_at_error': True}
        from cobaya.model import get_model

        model = get_model(info)
        my_mflike = model.likelihood["soliket.mflike.TestMFLike"]
        chi2 = -2 * (model.loglikes(nuisance_params)[0] - my_mflike.logp_const)

        assert np.isclose(chi2[0], chi2s["tt-te-et-ee"], atol=1e-2, rtol=0.0)

import numpy as np
from cobaya.model import get_model
from cobaya.tools import resolve_packages_path

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


def test_lensing_and_mflike_installations(check_skip_mflike):
    import mflike

    from soliket import LensingLikelihood

    is_installed = LensingLikelihood.is_installed(
        path=packages_path,
    )
    assert is_installed is True, (
        "LensingLikelihood is not installed! Please install it using "
        "'cobaya-install soliket.LensingLikelihood'"
    )

    is_installed = mflike.TTTEEE.is_installed(
        path=packages_path,
    )
    assert is_installed is True, (
        "mflike.TTTEEE is not installed! Please install it using "
        "'cobaya-install mflike.TTTEEE'"
    )


def test_multi(test_cosmology_params, check_skip_mflike):
    lensing_options = {"theory_lmax": 5000}

    mflike_options = {
        "input_file": "LAT_simu_sacc_00044.fits",
        "cov_Bbl_file": "data_sacc_w_covar_and_Bbl.fits",
        "stop_at_error": True,
    }

    camb_options = {"extra_args": {"lens_potential_accuracy": 1}}

    fg_params = {
        "a_tSZ": {"prior": {"min": 3.0, "max": 3.6}},
        "a_kSZ": {"prior": {"min": 1.4, "max": 1.8}},
    }
    mflike_params = test_cosmology_params | nuisance_params | fg_params

    lensing_params = test_cosmology_params

    info = {
        "likelihood": {
            "soliket.gaussian.MultiGaussianLikelihood": {
                "components": ["mflike.TTTEEE", "soliket.LensingLikelihood"],
                "options": [mflike_options, lensing_options],
                "stop_at_error": True,
            }
        },
        "theory": {
            "camb": camb_options,
            "mflike.BandpowerForeground": {"stop_at_error": True},
        },
        "params": mflike_params,
    }

    info1 = {
        "likelihood": {"mflike.TTTEEE": mflike_options},
        "theory": {
            "camb": camb_options,
            "mflike.BandpowerForeground": {"stop_at_error": True},
        },
        "params": mflike_params,
    }

    info2 = {
        "likelihood": {"soliket.LensingLikelihood": lensing_options},
        "theory": {"camb": camb_options},
        "params": lensing_params,
    }

    model = get_model(info)
    model1 = get_model(info1)
    model2 = get_model(info2)

    # To test here, the absolute values of the logps are not identical
    # to the sum of components when combined (probably due to numerical issues of
    # computing inv_cov); so here we test to make sure
    # that the change in logp between two different sets of params is identical

    fg_values_a = {"a_tSZ": nuisance_params["a_tSZ"], "a_kSZ": nuisance_params["a_kSZ"]}
    fg_values_b = {k: v * 1.1 for k, v in fg_values_a.items()}

    logp_a = model.loglikes(fg_values_a, cached=False)[0].sum()
    logp_b = model.loglikes(fg_values_b, cached=False)[0].sum()
    d_logp = logp_b - logp_a
    assert np.isclose(d_logp, -503.395, rtol=1e-4)

    model1_logp_a = model1.loglikes(fg_values_a, cached=False)[0].sum()
    model2_logp_a = model2.loglikes({}, cached=False)[0].sum()

    model1_logp_b = model1.loglikes(fg_values_b, cached=False)[0].sum()
    model2_logp_b = model2.loglikes({}, cached=False)[0].sum()

    d_logp1 = model1_logp_b - model1_logp_a
    d_logp2 = model2_logp_b - model2_logp_a
    d_logp_sum = d_logp1 + d_logp2

    assert np.isclose(d_logp, d_logp_sum, rtol=1e-5)

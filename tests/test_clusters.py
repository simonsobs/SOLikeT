import os

import numpy as np
from cobaya.model import get_model

import soliket.clusters.survey as survey
from soliket.clusters import tinker


def _get_cosmocnc_common_config():
    """Common config shared across all cosmocnc cluster tests."""
    import cosmocnc

    base = cosmocnc.path_to_cosmocnc
    survey_sr = os.path.join(base, "surveys", "survey_sr_so_sim.py")
    survey_cat = os.path.join(base, "surveys", "survey_cat_so_sim.py")

    cnc_common = {
        "stop_at_error": True,
        "survey_sr": survey_sr,
        "survey_cat": survey_cat,
        "cosmology_tool": "cobaya",
        "power_spectrum_type": "cobaya",
        "obs_select": "q_so_sim",
        "cluster_catalogue": "SO_sim_0",
        "n_points": 2048,
        "n_z": 50,
        "z_min": 0.01,
        "z_max": 3.0,
        "M_min": 5e13,
        "M_max": 5e15,
        "obs_select_min": 5.0,
        "obs_select_max": 200.0,
        "hmf_calc": "cnc",
        "hmf_type": "Tinker08",
        "mass_definition": "500c",
        "hmf_type_deriv": "numerical",
        "interp_tinker": "linear",
        "cosmo_amplitude_parameter": "sigma_8",
        "cosmo_param_density": "physical",
        "cosmocnc_verbose": "none",
        "number_cores_hmf": 1,
        "number_cores_abundance": 1,
        "number_cores_data": 1,
        "number_cores_stacked": 1,
    }
    theory = {
        "camb": {
            "extra_args": {
                "num_massive_neutrinos": 1,
                "accurate_massive_neutrino_transfers": True,
                "nonlinear": False,
                "kmax": 50.0,
            }
        },
    }
    params = {
        "ombh2": 0.0224,
        "omch2": 0.12,
        "H0": 67.0,
        "logA": 3.05,
        "As": {"value": "lambda logA: 1e-10*np.exp(logA)"},
        "ns": 0.965,
        "tau": 0.054,
        "nnu": 3.046,
        "mnu": 0.06,
        "A_szifi": -4.3054,
        "alpha_szifi": 1.12,
        "sigma_lnq_szifi": 0.173,
        "bias_sz": 0.8,
        "dof": 0.0,
    }
    return cnc_common, theory, params


def _get_cosmocnc_clusters_info(mode="unbinned"):
    """CNCLike-based cluster likelihood setup for various modes."""
    cnc_common, theory, params = _get_cosmocnc_common_config()

    if mode == "unbinned":
        cnc_like = dict(
            **cnc_common,
            observables=[["q_so_sim"]],
            data_lik_from_abundance=True,
            likelihood_type="unbinned",
            stacked_likelihood=False,
        )
    elif mode == "binned":
        cnc_like = dict(
            **cnc_common,
            observables=[["q_so_sim"]],
            data_lik_from_abundance=True,
            likelihood_type="binned",
            stacked_likelihood=False,
        )
    elif mode == "unbinned_backward":
        cnc_like = dict(
            **cnc_common,
            observables=[["q_so_sim"]],
            data_lik_from_abundance=False,
            likelihood_type="unbinned",
            stacked_likelihood=False,
        )
    elif mode == "multi_obs":
        cnc_like = dict(
            **cnc_common,
            observables=[["q_so_sim", "p_so_sim"]],
            data_lik_from_abundance=True,
            likelihood_type="unbinned",
            stacked_likelihood=False,
        )
        params.update(
            {
                "bias_cmblens": 0.8,
                "a_lens": 1.0,
                "sigma_lnp": 0.2,
                "corr_lnq_lnp": 0.0,
            }
        )
    elif mode == "stacked_lensing":
        cnc_like = dict(
            **cnc_common,
            observables=[["q_so_sim"]],
            data_lik_from_abundance=False,
            likelihood_type="unbinned",
            stacked_likelihood=True,
            stacked_data=["p_so_sim_stacked"],
            compute_stacked_cov=True,
        )
        params.update(
            {
                "bias_cmblens": 0.8,
                "a_lens": 1.0,
                "sigma_lnp": 0.2,
                "corr_lnq_lnp": 0.0,
            }
        )

    return {
        "likelihood": {"cosmocnc.CNCLike": cnc_like},
        "theory": theory,
        "params": params,
        "sampler": {"evaluate": None},
    }


clusters_like_and_theory = {
    "likelihood": {"soliket.ClusterLikelihood": {"stop_at_error": True}},
    "theory": {
        "camb": {
            "extra_args": {
                "accurate_massive_neutrino_transfers": True,
                "num_massive_neutrinos": 1,
                "redshifts": np.linspace(0, 2, 41),
                "nonlinear": False,
                "kmax": 10.0,
                "dark_energy_model": "ppf",
                "bbn_predictor": "PArthENoPE_880.2_standard.dat",
            }
        },
    },
}


def test_clusters_model(check_skip_pyccl, evaluate_one_info, test_cosmology_params):
    evaluate_one_info["params"] = test_cosmology_params
    evaluate_one_info.update(clusters_like_and_theory)

    _ = get_model(evaluate_one_info)


def test_clusters_loglike(
    check_skip_pyccl, evaluate_one_info, test_cosmology_params, likelihood_refs
):
    ref = likelihood_refs["clusters"]

    evaluate_one_info["params"] = test_cosmology_params
    evaluate_one_info.update(clusters_like_and_theory)

    model_fiducial = get_model(evaluate_one_info)

    lnl = model_fiducial.loglikes({})[0]
    assert np.isclose(lnl, ref["value"], rtol=ref["rtol"], atol=ref["atol"])


def test_clusters_n_expected(
    check_skip_pyccl, evaluate_one_info, test_cosmology_params, likelihood_refs
):
    ref = likelihood_refs["clusters"]

    evaluate_one_info["params"] = test_cosmology_params
    evaluate_one_info.update(clusters_like_and_theory)

    model_fiducial = get_model(evaluate_one_info)

    lnl = model_fiducial.loglikes({})[0]

    like = model_fiducial.likelihood["soliket.ClusterLikelihood"]

    assert np.isclose(lnl, ref["value"], rtol=ref["rtol"], atol=ref["atol"])
    assert like._get_n_expected() > 40


class FakeTable:
    def __init__(self, data):
        # data is dict of arrays
        self._data = data

    def keys(self):
        return list(self._data.keys())

    def __getitem__(self, key):
        return self._data[key]


def test_loadQ_file_exists(monkeypatch, tmp_path):
    # create a fake combined table file path
    fname = str(tmp_path / "QFit.fits")

    # monkeypatch os.path.exists to True for this path
    monkeypatch.setattr(os.path, "exists", lambda p: p == fname)

    # create a fake table with theta500Arcmin and one tile column
    theta = np.array([1.0, 2.0, 3.0])
    tilevals = np.array([0.1, 0.2, 0.3])
    fake = FakeTable({"theta500Arcmin": theta, "TILE1": tilevals})

    # monkeypatch atpy.Table.read to return our fake table
    monkeypatch.setattr(survey.atpy.Table, "read", lambda self, path=None: fake)
    # monkeypatch the spline builder to avoid scipy's m>k check for small arrays
    monkeypatch.setattr(survey.interpolate, "splrep", lambda x, y: (x, y))

    out = survey.loadQ(fname)
    # tck dict should have TILE1
    assert "TILE1" in out
    # the value should be a spline representation (tuple)
    assert isinstance(out["TILE1"], tuple)


def test_loadQ_file_missing_with_tiles(monkeypatch):
    fname = "/nonexistent/QFit.fits"
    # os.path.exists False
    monkeypatch.setattr(os.path, "exists", lambda p: False)

    # create fake per-tile files by having atpy.Table.read return a table for replacements
    def fake_read(self, path=None):
        # produce a table with theta500Arcmin and Q
        theta = np.array([1.0, 2.0, 3.0])
        q = np.array([0.2, 0.3, 0.4])
        return FakeTable({"theta500Arcmin": theta, "Q": q})

    monkeypatch.setattr(survey.atpy.Table, "read", fake_read)
    monkeypatch.setattr(survey.interpolate, "splrep", lambda x, y: (x, y))

    out = survey.loadQ(fname, tileNames=["tileA", "tileB"])
    assert set(out.keys()) == {"tileA", "tileB"}


def test_SurveyData_Q_property():
    sd = survey.SurveyData.__new__(survey.SurveyData)
    # when tiles True, Q should return tckQFit['Q']
    sd.tiles = True
    sd.tckQFit = {"Q": (1, 2, 3), "PRIMARY": (9, 9, 9)}
    assert sd.Q == sd.tckQFit["Q"]

    sd.tiles = False
    assert sd.Q == sd.tckQFit["PRIMARY"]


def test_tinker_params_and_radius():
    A0, a0, b0, c0 = tinker.tinker_params(200.0)
    assert np.isfinite(A0) and np.isfinite(a0)

    R = tinker.radius_from_mass(1e14, 1e10)
    assert R > 0


def test_top_hat_and_tinker_f():
    val = tinker.top_hatf(0.1)
    assert np.isfinite(val)
    params = (0.2, 1.0, 1.0, 0.1)
    out = tinker.tinker_f(0.5, params)
    assert np.isfinite(out)


def test_dn_dlogM_small_grid():
    # small synthetic grid
    M = np.array([1e13, 2e13])
    z = np.array([0.1])
    rho = np.array([1e10])
    delta = 200.0
    k = np.array([0.1, 0.2, 0.3])
    # P shape should be (nz, nk)
    P = np.tile(np.array([1.0, 2.0, 3.0]), (len(z), 1))

    out = tinker.dn_dlogM(M, z, rho, delta, k, P, comoving=False)
    # Expect output shape (nM, nz)
    assert out.shape[0] == M.shape[0]
    assert out.shape[1] == z.shape[0]


# --- cosmocnc-based cluster tests (no pyccl required) ---


def test_cosmocnc_clusters_model(check_skip_cosmocnc):
    info = _get_cosmocnc_clusters_info()
    _ = get_model(info)


def test_cosmocnc_clusters_loglike(check_skip_cosmocnc, likelihood_refs):
    ref = likelihood_refs["cosmocnc_clusters"]
    info = _get_cosmocnc_clusters_info()
    model = get_model(info)
    lnl = model.loglikes({})[0]
    assert np.isclose(lnl, ref["value"], rtol=ref["rtol"], atol=ref["atol"])


def test_cosmocnc_clusters_binned(check_skip_cosmocnc, likelihood_refs):
    ref = likelihood_refs["cosmocnc_clusters_binned"]
    info = _get_cosmocnc_clusters_info(mode="binned")
    model = get_model(info)
    lnl = model.loglikes({})[0]
    assert np.isclose(lnl, ref["value"], rtol=ref["rtol"], atol=ref["atol"])


def test_cosmocnc_clusters_unbinned_backward(check_skip_cosmocnc, likelihood_refs):
    ref = likelihood_refs["cosmocnc_clusters_unbinned_backward"]
    info = _get_cosmocnc_clusters_info(mode="unbinned_backward")
    model = get_model(info)
    lnl = model.loglikes({})[0]
    assert np.isclose(lnl, ref["value"], rtol=ref["rtol"], atol=ref["atol"])


def test_cosmocnc_clusters_multi_obs(check_skip_cosmocnc, likelihood_refs):
    ref = likelihood_refs["cosmocnc_clusters_multi_obs"]
    info = _get_cosmocnc_clusters_info(mode="multi_obs")
    model = get_model(info)
    lnl = model.loglikes({})[0]
    assert np.isclose(lnl, ref["value"], rtol=ref["rtol"], atol=ref["atol"])


def test_cosmocnc_clusters_stacked_lensing(check_skip_cosmocnc, likelihood_refs):
    ref = likelihood_refs["cosmocnc_clusters_stacked_lensing"]
    info = _get_cosmocnc_clusters_info(mode="stacked_lensing")
    model = get_model(info)
    lnl = model.loglikes({})[0]
    assert np.isclose(lnl, ref["value"], rtol=ref["rtol"], atol=ref["atol"])

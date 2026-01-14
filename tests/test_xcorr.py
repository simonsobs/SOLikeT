# pytest -k xcorr -v --pdb .

import copy

import numpy as np
import pytest
from cobaya.model import get_model

from soliket.xcorr.limber import do_limber, mag_bias_kernel

fiducial_params = {
    "b1": 1.0,
    "s1": 0.4,
    "tau": 0.05,
    "mnu": 0.0,
    "nnu": 3.046,
}
info = {}
info["params"] = fiducial_params
info["likelihood"] = {
    "soliket.xcorr.XcorrLikelihood": {
        "stop_at_error": True,
    }
}
info["debug"] = True


def get_demo_xcorr_info(theory):
    if theory == "camb":
        info["theory"] = {"camb": {"extra_args": {"lens_potential_accuracy": 1}}}
    elif theory == "classy":
        info["theory"] = {
            "classy": {"extra_args": {"output": "lCl, tCl"}, "path": "global"}
        }
    return info


def test_wrong_types():
    from soliket.xcorr import XcorrLikelihood

    base_case = {
        "auto_file": "auto",
        "cross_file": "cross",
        "dndz_file": "dndz",
        "datapath": "path",
        "k_tracer_name": "k_tracer",
        "gc_tracer_name": "gc_tracer",
        "high_ell": 1000,
        "nz": 10,
        "Nchi": 100,
        "Nchi_mag": 100,
        "Pk_interp_kmax": 1.0,
        "b1": 1.0,
        "s1": 1.0,
    }

    wrong_type_cases = {
        "auto_file": 123,
        "cross_file": 123,
        "dndz_file": 123,
        "datapath": 123,
        "k_tracer_name": 123,
        "gc_tracer_name": 123,
        "high_ell": "not_an_int",
        "nz": "not_an_int",
    }

    for key, wrong_value in wrong_type_cases.items():
        case = copy.deepcopy(base_case)
        case[key] = wrong_value
        with pytest.raises(TypeError):
            XcorrLikelihood(**case)


@pytest.mark.parametrize("theory", ["camb"])  # , "classy"])
def test_xcorr_model(theory):
    info = get_demo_xcorr_info(theory)
    _ = get_model(info)


def test_xcorr_get_theory(tmp_path):
    from soliket.xcorr import XcorrLikelihood

    # Build a minimal provider that supplies the methods used by _get_theory
    class PkObj:
        def __init__(self, P):
            self.P = P

    class LocalProvider:
        def __init__(self):
            self._params = {"H0": 70.0, "omegam": 0.3, "zstar": 1.0}

        def get_param(self, name):
            return self._params[name]

        def get_Hubble(self, z, units=None):
            return np.ones_like(z) * 70.0

        def get_comoving_radial_distance(self, z):
            return np.array(z) * 10.0

        def get_Pk_interpolator(self, *args, **kwargs):
            # Return an object with attribute P that matches the simple pk used
            def pk(z, k_arr):
                # ensure shape (1, len(k_arr)) as expected downstream
                return np.ones((1, k_arr.size))

            return PkObj(pk)

    # create a minimal XcorrLikelihood object and attach required attrs
    xl = XcorrLikelihood.__new__(XcorrLikelihood)
    xl.provider = LocalProvider()

    # simple identical dndz arrays used by do_limber
    xl.dndz = np.array([[0.0, 1.0], [1.0, 0.0]])
    # use a slightly larger grid so scipy spline (k=3) has enough points
    xl.nz = 5
    xl.Nchi = 5
    xl.Nchi_mag = 5
    xl.high_ell = 100
    xl.ell_range = np.linspace(1, xl.high_ell, int(xl.high_ell + 1))

    # small data object with x of even length so binning works
    class SimpleData:
        x = np.array([1.0, 2.0, 3.0, 4.0])

    xl.data = SimpleData()

    # call _get_theory with required params
    # _setup_chi expects a zarray attribute, normally set in initialize
    xl.zarray = np.linspace(xl.dndz[:, 0].min(), xl.dndz[:, 0].max(), xl.nz)

    # Monkeypatch the heavy do_limber call to a simple deterministic stub
    import soliket.xcorr.xcorr as xcorr_mod

    def simple_do_limber(
        ell, provider, dndz1, dndz2, s1, s2, Pk, b1, b2, aa, ac, chi_grids, **kwargs
    ):
        # return two Cl arrays matching ell length
        return np.ones_like(ell), 0.5 * np.ones_like(ell)

    xcorr_mod.do_limber = simple_do_limber

    xl.binning_matrix = np.eye(xl.high_ell + 1)[: xl.data.x.shape[0]]
    res = xl._get_theory(s1=0.4, b1=1.0, alpha_auto=1.0, alpha_cross=1.0)

    assert isinstance(res, np.ndarray)
    assert res.shape[0] == xl.data.x.shape[0] * 2
    assert np.all(np.isfinite(res))


@pytest.mark.skip(reason="Under development")
@pytest.mark.parametrize("theory", ["camb"])  # , "classy"])
def test_xcorr_like(theory):
    params = {"b1": 1.0, "s1": 0.4}

    info = get_demo_xcorr_info(theory)
    model = get_model(info)

    lnl = model.loglike(params)[0]
    assert np.isfinite(lnl)

    xcorr_lhood = model.likelihood["soliket.XcorrLikelihood"]

    setup_chi_out = xcorr_lhood._setup_chi()

    Pk_interpolator = xcorr_lhood.provider.get_Pk_interpolator(
        ("delta_nonu", "delta_nonu"), extrap_kmax=1.0e8, nonlinear=False
    ).P

    from soliket.xcorr.limber import do_limber

    cl_gg, cl_kappag = do_limber(
        xcorr_lhood.ell_range,
        xcorr_lhood.provider,
        xcorr_lhood.dndz,
        xcorr_lhood.dndz,
        params["s1"],
        params["s1"],
        Pk_interpolator,
        params["b1"],
        params["b1"],
        params["alpha_auto"],
        params["alpha_cross"],
        setup_chi_out,
        Nchi=xcorr_lhood.Nchi,
        dndz1_mag=xcorr_lhood.dndz,
        dndz2_mag=xcorr_lhood.dndz,
    )

    ell_load = xcorr_lhood.data.x
    cl_load = xcorr_lhood.data.y
    # cov_load = xcorr_lhood.data.cov
    # cl_err_load = np.sqrt(np.diag(cov_load))
    n_ell = len(ell_load) // 2

    ell_obs_gg = ell_load[n_ell:]
    ell_obs_kappag = ell_load[:n_ell]

    cl_obs_gg = cl_load[:n_ell]
    cl_obs_kappag = cl_load[n_ell:]

    # Nell_unwise_g = np.ones_like(cl_gg) \
    #                         / (xcorr_lhood.ngal * (60 * 180 / np.pi)**2)
    Nell_obs_unwise_g = np.ones_like(cl_obs_gg) / (
        xcorr_lhood.ngal * (60 * 180 / np.pi) ** 2
    )

    import pyccl as ccl

    h2 = (xcorr_lhood.provider.get_param("H0") / 100) ** 2

    cosmo = ccl.Cosmology(
        Omega_c=xcorr_lhood.provider.get_param("omch2") / h2,
        Omega_b=xcorr_lhood.provider.get_param("ombh2") / h2,
        h=xcorr_lhood.provider.get_param("H0") / 100,
        n_s=xcorr_lhood.provider.get_param("ns"),
        A_s=xcorr_lhood.provider.get_param("As"),
        Omega_k=xcorr_lhood.provider.get_param("omk"),
        Neff=xcorr_lhood.provider.get_param("nnu"),
        matter_power_spectrum="linear",
    )

    g_bias_zbz = (
        xcorr_lhood.dndz[:, 0],
        params["b1"] * np.ones(len(xcorr_lhood.dndz[:, 0])),
    )
    mag_bias_zbz = (
        xcorr_lhood.dndz[:, 0],
        params["s1"] * np.ones(len(xcorr_lhood.dndz[:, 0])),
    )

    tracer_g = ccl.NumberCountsTracer(
        cosmo,
        has_rsd=False,
        dndz=xcorr_lhood.dndz.T,
        bias=g_bias_zbz,
        mag_bias=mag_bias_zbz,
    )

    tracer_k = ccl.CMBLensingTracer(cosmo, z_source=1100)

    cl_gg_ccl = ccl.cells.angular_cl(cosmo, tracer_g, tracer_g, xcorr_lhood.ell_range)
    cl_kappag_ccl = ccl.cells.angular_cl(cosmo, tracer_k, tracer_g, xcorr_lhood.ell_range)

    assert np.allclose(cl_gg_ccl, cl_gg)
    assert np.allclose(cl_kappag_ccl, cl_kappag)

    cl_obs_gg_ccl = ccl.cells.angular_cl(cosmo, tracer_g, tracer_g, ell_obs_gg)
    cl_obs_kappag_ccl = ccl.cells.angular_cl(cosmo, tracer_k, tracer_g, ell_obs_kappag)

    assert np.allclose(cl_obs_gg_ccl + Nell_obs_unwise_g, cl_obs_gg)
    assert np.allclose(cl_obs_kappag_ccl, cl_obs_kappag)


class DummyProvider:
    def __init__(self):
        self._params = {"H0": 70.0, "omegam": 0.3, "zstar": 1.0}

    def get_param(self, name):
        return self._params[name]

    def get_Hubble(self, z, units=None):
        # simple constant H(z)
        return np.ones_like(z) * 70.0

    def get_comoving_radial_distance(self, z):
        # simple linear mapping
        return np.array(z) * 10.0

    def get_Pk_interpolator(self, *args, **kwargs):
        # Provide a very small Pk interpolator compatible with do_limber tests
        class PkObj:
            def __init__(self, P):
                self.P = P

        def pk(z, k_arr):
            return np.ones((1, k_arr.size))

        return PkObj(pk)


def test_mag_bias_kernel_basic():
    prov = DummyProvider()
    # dndz as two-column array (z, nz)
    dndz = np.array([[0.0, 0.1], [1.0, 0.9]])
    s1 = 0.4
    zatchi = lambda chi: chi * 0.1
    chi_arr = np.linspace(1.0, 3.0, 3)
    # make chiprime array longer so trapezoid along axis=0 sees multiple intervals
    chiprime_arr = np.linspace(1.5, 4.5, 4).reshape(-1, 1)
    zprime_arr = zatchi(chiprime_arr[:, 0]).reshape(-1, 1)

    W = mag_bias_kernel(prov, dndz, s1, zatchi, chi_arr, chiprime_arr, zprime_arr)
    assert W.shape == chi_arr.shape
    # values should be finite
    assert np.all(np.isfinite(W))


def test_do_limber_basic():
    prov = DummyProvider()
    ell = np.array([10.0, 20.0])

    # simple identical dndz arrays
    dndz1 = np.array([[0.0, 1.0], [1.0, 0.0]])
    dndz2 = dndz1.copy()

    # trivial power spectrum: return ones of same shape as k_arr
    def pk(z, k_arr):
        # return shape (1, len(k_arr)) so downstream code broadcasting works
        return np.ones((1, k_arr.size))

    chi_arr = np.linspace(1.0, 3.0, 3)
    chivalp = np.linspace(1.5, 4.5, 4).reshape(-1, 1)
    zvalp = (chivalp * 0.1).reshape(-1, 1)
    chi_grids = {
        "zatchi": (lambda chi: chi * 0.1),
        "chival": chi_arr,
        "zval": chi_arr * 0.1,
        "chivalp": chivalp,
        "zvalp": zvalp,
    }

    clgg, clkappa = do_limber(
        ell,
        prov,
        dndz1,
        dndz2,
        0.4,
        0.4,
        pk,
        1.0,
        1.0,
        1.0,
        1.0,
        chi_grids,
        Nchi=3,
    )

    assert clgg.shape[0] == ell.shape[0]
    assert clkappa.shape[0] == ell.shape[0]
    assert np.all(np.isfinite(clgg))
    assert np.all(np.isfinite(clkappa))

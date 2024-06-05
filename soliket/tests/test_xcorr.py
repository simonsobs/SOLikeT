# pytest -k xcorr -v --pdb .

import numpy as np
import pytest
from cobaya.model import get_model
from cobaya.yaml import yaml_load


def get_demo_xcorr_model(theory):
    if theory == "camb":
        info_yaml = r"""
        likelihood:
            soliket.XcorrLikelihood:
                stop_at_error: True
                datapath: soliket/tests/data/unwise_g-so_kappa.sim.sacc.fits
                k_tracer_name: ck_so
                gc_tracer_name: gc_unwise

        theory:
            camb:
                extra_args:
                  lens_potential_accuracy: 1

        params:
            tau: 0.05
            mnu: 0.0
            nnu: 3.046
            b1:
                prior:
                    min: 0.
                    max: 10.
                ref:
                    min: 1.
                    max: 4.
                proposal: 0.1
            s1:
                prior:
                    min: 0.1
                    max: 1.0
                proposal: 0.1
        """
    elif theory == "classy":
        info_yaml = r"""
        likelihood:
            soliket.XcorrLikelihood:
                stop_at_error: True
                datapath: soliket/tests/data/unwise_g-so_kappa.sim.sacc.fits
                k_tracer_name: ck_so
                gc_tracer_name: gc_unwise

        theory:
            classy:
                extra_args:
                    output: lCl, tCl
                path: global

        params:
            b1:
                prior:
                    min: 0.
                    max: 10.
                ref:
                    min: 1.
                    max: 4.
                proposal: 0.1
            s1:
                prior:
                    min: 0.1
                    max: 1.0
                proposal: 0.1

        """

    info = yaml_load(info_yaml)
    model = get_model(info)
    return model


@pytest.mark.skip(reason="Under development")
@pytest.mark.parametrize("theory", ["camb"])# , "classy"])
def test_xcorr(theory):

    params = {'b1': 1.0, 's1': 0.4}

    model = get_demo_xcorr_model(theory)

    lnl = model.loglike(params)[0]
    assert np.isfinite(lnl)

    xcorr_lhood = model.likelihood['soliket.XcorrLikelihood']

    setup_chi_out = xcorr_lhood._setup_chi()

    Pk_interpolator = xcorr_lhood.provider.get_Pk_interpolator(("delta_nonu",
                                                                "delta_nonu"),
                                                             extrap_kmax=1.e8,
                                                             nonlinear=False).P

    from soliket.xcorr.limber import do_limber

    cl_gg, cl_kappag = do_limber(xcorr_lhood.ell_range,
                                   xcorr_lhood.provider,
                                   xcorr_lhood.dndz,
                                   xcorr_lhood.dndz,
                                   params['s1'],
                                   params['s1'],
                                   Pk_interpolator,
                                   params['b1'],
                                   params['b1'],
                                   xcorr_lhood.alpha_auto,
                                   xcorr_lhood.alpha_cross,
                                   setup_chi_out,
                                   Nchi=xcorr_lhood.Nchi,
                                   dndz1_mag=xcorr_lhood.dndz,
                                   dndz2_mag=xcorr_lhood.dndz)

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
    Nell_obs_unwise_g = np.ones_like(cl_obs_gg) \
                            / (xcorr_lhood.ngal * (60 * 180 / np.pi)**2)

    import pyccl as ccl
    h2 = (xcorr_lhood.provider.get_param('H0') / 100)**2

    cosmo = ccl.Cosmology(Omega_c=xcorr_lhood.provider.get_param('omch2') / h2,
                          Omega_b=xcorr_lhood.provider.get_param('ombh2') / h2,
                          h=xcorr_lhood.provider.get_param('H0') / 100,
                          n_s=xcorr_lhood.provider.get_param('ns'),
                          A_s=xcorr_lhood.provider.get_param('As'),
                          Omega_k=xcorr_lhood.provider.get_param('omk'),
                          Neff=xcorr_lhood.provider.get_param('nnu'),
                          matter_power_spectrum='linear')

    g_bias_zbz = (xcorr_lhood.dndz[:, 0],
                  params['b1'] * np.ones(len(xcorr_lhood.dndz[:, 0])))
    mag_bias_zbz = (xcorr_lhood.dndz[:, 0],
                    params['s1'] * np.ones(len(xcorr_lhood.dndz[:, 0])))

    tracer_g = ccl.NumberCountsTracer(cosmo,
                                      has_rsd=False,
                                      dndz=xcorr_lhood.dndz.T,
                                      bias=g_bias_zbz,
                                      mag_bias=mag_bias_zbz)

    tracer_k = ccl.CMBLensingTracer(cosmo, z_source=1100)

    cl_gg_ccl = ccl.cells.angular_cl(cosmo, tracer_g, tracer_g, xcorr_lhood.ell_range)
    cl_kappag_ccl = ccl.cells.angular_cl(cosmo, tracer_k, tracer_g, xcorr_lhood.ell_range)

    assert np.allclose(cl_gg_ccl, cl_gg)
    assert np.allclose(cl_kappag_ccl, cl_kappag)

    cl_obs_gg_ccl = ccl.cells.angular_cl(cosmo, tracer_g, tracer_g, ell_obs_gg)
    cl_obs_kappag_ccl = ccl.cells.angular_cl(cosmo, tracer_k, tracer_g, ell_obs_kappag)

    assert np.allclose(cl_obs_gg_ccl + Nell_obs_unwise_g, cl_obs_gg)
    assert np.allclose(cl_obs_kappag_ccl, cl_obs_kappag)

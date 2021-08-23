# pytest -k xcorr -v --pdb .

import pytest
import numpy as np

from cobaya.yaml import yaml_load
from cobaya.model import get_model

import os
import pdb

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
            s1: 
                prior:
                    min: 0.1
                    max: 1.0
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
            s1: 
                prior:
                    min: 0.1
                    max: 1.0

        """

    info = yaml_load(info_yaml)
    model = get_model(info)
    return model


@pytest.mark.parametrize("theory", ["camb"])#, "classy"])
def test_xcorr(theory):

    params = {'b1': 1.0, 's1': 0.4}

    model = get_demo_xcorr_model(theory)
    
    lnl = model.loglike(params)[0]
    assert np.isfinite(lnl)

    xcorr_lhood = model.likelihood['soliket.XcorrLikelihood']
    
    setup_chi_out = xcorr_lhood._setup_chi()

    Pk_interpolator = xcorr_lhood.theory.get_Pk_interpolator(("delta_nonu", "delta_nonu"), extrap_kmax=1.e8, nonlinear=False).P

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
                                   Nchi=xcorr_lhood.Nchi,
                                   autoCMB=False,
                                   use_zeff=False,
                                   dndz1_mag=xcorr_lhood.dndz,
                                   dndz2_mag=xcorr_lhood.dndz,
                                   setup_chi_flag=True,
                                   setup_chi_out=setup_chi_out)

    ell_load = xcorr_lhood.data.x
    cl_load = xcorr_lhood.data.y
    cov_load = xcorr_lhood.data.cov
    cl_err_load = np.sqrt(np.diag(cov_load))
    n_ell = len(ell_load) // 2

    ell_obs_gg = ell_load[n_ell:]
    ell_obs_kappag = ell_load[:n_ell]

    cl_obs_gg = cl_load[:n_ell]
    cl_obs_kappag = cl_load[n_ell:]

    Nell_unwise_g = np.ones_like(cl_gg) / (xcorr_lhood.ngal * (60 * 180 / np.pi)**2)
    Nell_obs_unwise_g = np.ones_like(cl_obs_gg) / (xcorr_lhood.ngal * (60 * 180 / np.pi)**2)

    import pyccl as ccl
    cosmo = ccl.Cosmology(Omega_c=xcorr_lhood.provider.get_param('omch2') / (xcorr_lhood.provider.get_param('H0') / 100 * xcorr_lhood.provider.get_param('H0') / 100),
                          Omega_b=xcorr_lhood.provider.get_param('ombh2') / (xcorr_lhood.provider.get_param('H0') / 100 * xcorr_lhood.provider.get_param('H0') / 100),
                          h=xcorr_lhood.provider.get_param('H0') / 100,
                          n_s=xcorr_lhood.provider.get_param('ns'),
                          A_s=xcorr_lhood.provider.get_param('As'),
                          Omega_k=xcorr_lhood.provider.get_param('omk'),
                          Neff=xcorr_lhood.provider.get_param('nnu'),
                          matter_power_spectrum='linear')

    tracer_g = ccl.NumberCountsTracer(cosmo,
                                      has_rsd=False,
                                      dndz = xcorr_lhood.dndz.T,
                                      bias =(xcorr_lhood.dndz[:,0], params['b1']*np.ones(len(xcorr_lhood.dndz[:,0]))), 
                                      mag_bias = (xcorr_lhood.dndz[:,0], params['s1']*np.ones(len(xcorr_lhood.dndz[:,0])))
                                      )

    tracer_k = ccl.CMBLensingTracer(cosmo, z_source=1100)

    cl_gg_ccl = ccl.cls.angular_cl(cosmo, tracer_g, tracer_g, xcorr_lhood.ell_range)
    cl_kappag_ccl = ccl.cls.angular_cl(cosmo, tracer_k, tracer_g, xcorr_lhood.ell_range)

    assert np.allclose(cl_gg_ccl, cl_gg)
    assert np.allclose(cl_kappag_ccl, cl_kappag)

    cl_obs_gg_ccl = ccl.cls.angular_cl(cosmo, tracer_g, tracer_g, ell_obs_gg)
    cl_obs_kappag_ccl = ccl.cls.angular_cl(cosmo, tracer_k, tracer_g, ell_obs_kappag)

    assert np.allclose(cl_obs_gg_ccl + Nell_obs_unwise_g, cl_obs_gg)
    assert np.allclose(cl_obs_kappag_ccl, cl_obs_kappag)

    # from matplotlib import pyplot as plt
    # test_dir = os.path.dirname(os.path.abspath(__file__))

    # plt.close('all')
    # plt.figure(1, figsize=(2*4.5, 3.75))
    # plt.subplot(121)
    # plt.plot(xcorr_lhood.ell_range, 1.e5*(cl_gg_ccl + Nell_unwise_g), '+', color='C2', label='CCL (soliket.cross_correlation)')
    # plt.plot(xcorr_lhood.ell_range, 1.e5*(cl_gg + Nell_unwise_g), '-', color='C1', label='soliket.xcorr')
    # plt.plot(ell_load[:n_ell], 1.e5*cl_load[:n_ell], 'o', ms=3, color='C0', label='sacc file')
    # plt.errorbar(ell_load[:n_ell], 1.e5*cl_load[:n_ell], yerr=1.e5*cl_err_load[:n_ell], fmt='none', color='C0')
    # plt.xlabel('$\\ell$')
    # plt.ylabel('$C_{\\ell}$')
    # plt.xlim([0,600])
    # plt.title('$gg$')
    # plt.legend(loc='upper right', fontsize='small')
    # plt.subplot(122)
    # plt.plot(xcorr_lhood.ell_range, xcorr_lhood.ell_range*1.e5*cl_kappag_ccl, '+', color='C2')
    # plt.plot(xcorr_lhood.ell_range, xcorr_lhood.ell_range*1.e5*cl_kappag, '-', color='C1')
    # plt.plot(ell_load[n_ell:], 1.e5*ell_load[n_ell:]*cl_load[n_ell:], 'o', ms=3, color='C0')
    # plt.errorbar(ell_load[n_ell:], 1.e5*ell_load[n_ell:]*cl_load[n_ell:], yerr=1.e5*ell_load[n_ell:]*cl_err_load[n_ell:], fmt='none', color='C0')
    # plt.ylabel('$\\ell C_{\\ell}$')
    # plt.xlabel('$\\ell$')
    # plt.xlim([0,600])
    # plt.title('$\\kappa g$')
    # # plt.subplot(133)
    # # plt.plot(xcorr_lhood.ell_range, xcorr_lhood.ell_range*1.e5*cl_kappakappa, '--', color='C1', label='soliket.xcorr')
    # # #plt.plot(xcorr_lhood.ell_range[::10], xcorr_lhood.ell_range[::10]*1.e5*Clkk_theo[::10], 'x', color='C0', label='camb')
    # # plt.ylabel('$\\ell C_{\\ell}$')
    # # plt.xlabel('$\\ell$')
    # # plt.xlim([0,600])
    # # plt.legend(loc='upper right', fontsize='small')
    # # plt.title('$\\kappa\\kappa$')
    # plt.savefig(os.path.join(test_dir, 'xcorr_dv.png'), dpi=300, bbox_inches='tight')

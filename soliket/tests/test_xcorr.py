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

    params = {'b1': 1.0, 's1': 0.5}

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


    from matplotlib import pyplot as plt
    test_dir = os.path.dirname(os.path.abspath(__file__))

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

    plt.close('all')
    plt.figure(1, figsize=(2*4.5, 3.75))
    plt.subplot(121)
    plt.plot(xcorr_lhood.ell_range, 1.e5*cl_gg_ccl, '+', color='C2', label='CCL (soliket.cross_correlation)')
    plt.plot(xcorr_lhood.ell_range, 1.e5*cl_gg, '-', color='C1', label='soliket.xcorr')
    plt.xlabel('$\\ell$')
    plt.ylabel('$C_{\\ell}$')
    plt.xlim([0,600])
    plt.title('$gg$')
    plt.legend(loc='upper right', fontsize='small')
    plt.subplot(122)
    plt.plot(xcorr_lhood.ell_range, xcorr_lhood.ell_range*1.e5*cl_kappag_ccl, '+', color='C2')
    plt.plot(xcorr_lhood.ell_range, xcorr_lhood.ell_range*1.e5*cl_kappag, '-', color='C1')
    plt.ylabel('$\\ell C_{\\ell}$')
    plt.xlabel('$\\ell$')
    plt.xlim([0,600])
    plt.title('$\\kappa g$')
    # plt.subplot(133)
    # plt.plot(xcorr_lhood.ell_range, xcorr_lhood.ell_range*1.e5*cl_kappakappa, '--', color='C1', label='soliket.xcorr')
    # #plt.plot(xcorr_lhood.ell_range[::10], xcorr_lhood.ell_range[::10]*1.e5*Clkk_theo[::10], 'x', color='C0', label='camb')
    # plt.ylabel('$\\ell C_{\\ell}$')
    # plt.xlabel('$\\ell$')
    # plt.xlim([0,600])
    # plt.legend(loc='upper right', fontsize='small')
    # plt.title('$\\kappa\\kappa$')
    plt.savefig(os.path.join(test_dir, 'xcorr_dv.png'), dpi=300, bbox_inches='tight')

# pytest -k xcorr -v --pdb .

import pytest
import numpy as np

from cobaya.yaml import yaml_load
from cobaya.model import get_model

import os

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
            b1: 
                prior:
                    min: 0.
                    max: 10.
            s1: 
                value: 0.4
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
                value: 0.4

        """

    info = yaml_load(info_yaml)
    model = get_model(info)
    return model


@pytest.mark.parametrize("theory", ["camb"])#, "classy"])
def test_xcorr(theory):

    params = {"b1": 1.0, "s1": 0.4}

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

    # c_ells = xcorr_lhood.theory.get_Cl(ell_factor=False)
    # Cl_theo = c_ells["pp"][0 : xcorr_lhood.high_ell+1]
    # Clkk_theo = (xcorr_lhood.ell_range * (xcorr_lhood.ell_range + 1)) ** 2 * Cl_theo * 0.25

    N_ell_auto = xcorr_lhood.ell_auto.shape[0]
    N_ell_cross = xcorr_lhood.ell_cross.shape[0]

    plt.close('all')
    plt.figure(1, figsize=(2*4.5, 3.75))
    plt.subplot(121)
    plt.plot(xcorr_lhood.ell_range, 1.e5*cl_gg, '--', color='C1', label='soliket.xcorr')
    plt.plot(xcorr_lhood.data.x[:N_ell_auto], 1.e5*xcorr_lhood.data.y[:N_ell_auto], 'o', color='C2', label='simonsobs/xcorr')
    plt.xlabel('$\\ell$')
    plt.ylabel('$C_{\\ell}$')
    plt.xlim([0,600])
    plt.title('$gg$')
    plt.legend(loc='upper right', fontsize='small')
    plt.subplot(122)
    plt.plot(xcorr_lhood.ell_range, xcorr_lhood.ell_range*1.e5*cl_kappag, '--', color='C1')
    plt.plot(xcorr_lhood.data.x[:-N_ell_cross], xcorr_lhood.ell_cross*1.e5*xcorr_lhood.data.y[:-N_ell_cross], 'o', color='C2')
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

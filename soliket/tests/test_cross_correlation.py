import numpy as np
from soliket.ccl import CCL
from soliket.cross_correlation import GalaxyKappaLikelihood, ShearKappaLikelihood
from cobaya.model import get_model


def test_galaxykappa():
    cosmo_params = {
        "Omega_c": 0.25,
        "Omega_b": 0.05,
        "h": 0.67,
        "n_s": 0.96
    }

    info = {"params": {"omch2": cosmo_params['Omega_c'] * cosmo_params['h'] ** 2.,
                       "ombh2": cosmo_params['Omega_b'] * cosmo_params['h'] ** 2.,
                       "H0": cosmo_params['h'] * 100,
                       "ns": cosmo_params['n_s'],
                       "As": 2.2e-9,
                       "tau": 0,
                       "b1": 1,
                       "s1": 0.4},
            "likelihood": {"GalaxyKappaLikelihood": GalaxyKappaLikelihood},
            "theory": {
                "camb": None,
                "ccl": {"external": CCL, "nonlinear": False}
            },
            "debug": False, "stop_at_error": True}

    model = get_model(info)
    loglikes, derived = model.loglikes()
    assert np.isclose(loglikes[0], 88.2, atol=.2, rtol=0.)


def test_shearkappa():
    cosmo_params = {
        "h": 0.673,
        "n_s": 0.964
    }

    info = {"params": {#"omch2": cosmo_params['Omega_c'] * cosmo_params['h'] ** 2.,
                       "omch2": 0.1200,
                       "ombh2": 0.0223,
                       "H0": cosmo_params['h'] * 100,
                       "ns": cosmo_params['n_s'],
                       "As": 2.1e-9,
                       "tau": 0.054,
                       "A_IA": 0.0},
            "likelihood": {"ShearKappaLikelihood":
                            {"external": ShearKappaLikelihood,
                             'datapath': '/Users/ianharrison/Dropbox/code_cdf/act-x-des/desgamma-x-actkappa/data/des_s-act_kappa.FLASK-sim_mockdata_and_cov.fits'
                             # "dndz_file": "soliket/tests/data/dndz_hsc.txt"  # noqa E501
                             }
                          },
            "theory": {
                "camb": None,
                "ccl": {"external": CCL, "nonlinear": False}
            },
            "debug": False, "stop_at_error": True}

    model = get_model(info)
    loglikes, derived = model.loglikes()

    assert np.isfinite(loglikes)

    # lhood = model.likelihood['ShearKappaLikelihood']

    # ell_paper, ellcl_paper = np.loadtxt('soliket/tests/data/lcl_hsc.txt', unpack=True)
    # ells = ell_paper

    # ell_load = np.concatenate([ells, ells])
    # n_ell = len(ell_load) // 2

    # lhood.ell_auto = ell_load[n_ell:]
    # lhood.ell_cross = ell_load[:n_ell]

    # ell_obs_kappagamma = ell_load[n_ell:]
    # # ell_obs_gammagamma = ell_load[:n_ell]

    # cl_theory = lhood._get_theory(**info["params"])
    # cl_kappagamma = cl_theory[n_ell:]
    # cl_gammagamma = cl_theory[:n_ell]

    # from matplotlib import pyplot as plt
    # plt.figure(1, figsize=(4.5, 3.75))
    # plt.plot(ell_obs_kappagamma, ell_obs_kappagamma * cl_kappagamma * 1.e6,
    #          label='Calculated')
    # plt.plot(ell_paper, ellcl_paper, '--',
    #          label='Marques et al, Planck 2018 $A=1$')
    # plt.title(r'$\kappa \gamma$ HSC')
    # plt.ylabel(r'$\ell C_{\ell} / 10^6$')
    # plt.xlabel(r'$\ell$')
    # # plt.xscale('log')
    # plt.legend(loc='upper right')
    # plt.ylim([-0.5, 1.6])
    # plt.xlim([60, 2000])
    # plt.axhline(0, color='k', linestyle='dashed', alpha=0.4)
    # plt.savefig('./plots/kappagamma-hsc.png', dpi=300, bbox_inches='tight')

    # assert np.allclose(ell_obs_kappagamma * cl_kappagamma * 1.e6,
    #                    ellcl_paper,
    #                    atol=.2, rtol=0.)

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
                             'datapath': '/Users/ianharrison/Dropbox/code_cdf/act-x-des/desgamma-x-actkappa/data/des_s-act_kappa.FLASK-sim_mockdata_and_cov_exact_win.fits'
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

    lhood = model.likelihood['ShearKappaLikelihood']

    cl_binned = lhood._get_theory(**info["params"])

    # ell_unbinned = unbinned[0]
    # cl_unbinned = unbinned[1]

    # ell_unbinned = ell_unbinned.reshape(4, 1951)
    # cl_unbinned = cl_unbinned.reshape(4, 1951)
    cl_binned = cl_binned.reshape(4, 13)

    import sacc

    s = sacc.Sacc.load_fits('/Users/ianharrison/Dropbox/code_cdf/act-x-des/desgamma-x-actkappa/data/des_s-act_kappa.FLASK-sim_mockdata_and_cov_exact_win.fits')

    from matplotlib import pyplot as plt

    # import pdb
    # pdb.set_trace()

    plt.close(1)
    plt.figure(1, figsize=(4.*4.5, 3.75))

    plt.suptitle(r'ACT $\times$ DES (FLASK sim)')

    nbins_des = 4
    ell_max = 1950

    for ibin in np.arange(1,nbins_des+1):
        plt.subplot(1,4,ibin)
        plt.title(r'$\kappa \gamma_{}$'.format(ibin))
        ell, cl, cov = s.get_ell_cl('cl_20', 'gs_des_bin{}'.format(ibin), 'ck_act', return_cov=True)
        cl_err = np.sqrt(np.diag(cov))
        # plt.plot(ell_theory, 1.e5*ell_theory*cl_gk_theory , '--', c='C1')
        plt.plot(ell, cl, 'o', ms=3, c='C1', label='Data')
        plt.errorbar(ell, cl, yerr=cl_err, fmt='none', c='C1')

        # plt.plot(ell_unbinned, cl_unbinned[ibin-1], '-', ms=3, c='C2', label='SOLikeT ShearKappaLikelihood')
        plt.plot(ell, cl_binned[ibin-1], '.-', ms=3, c='C2', label='SOLikeT ShearKappaLikelihood')

        plt.xlim([1,ell_max])
        plt.ylim([1.e-11, 1.e-7])
        plt.xlabel(r'$\ell$')
        plt.yscale('log')
        plt.axhline(0, color='k', linestyle='dashed', alpha=0.4)
        if ibin == 1:
            plt.ylabel(r'$C_\ell$')
            plt.legend()

    plt.savefig('./des-act-flask.png', dpi=300, bbox_inches='tight')

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

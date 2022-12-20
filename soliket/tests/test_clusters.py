import numpy as np
import pytest

from cobaya.model import get_model


params = {
    'h': 0.68,
    'n_s': 0.965,
    'Omega_b': 0.049,      
    'Omega_c': 0.261, 
    'sigma8': 0.81,
    'm_nu': 0.,    
    'tenToA0': 1.9e-05,
    'B0': 0.08,
    'scatter_sz': 0.,
    'bias_sz': 1.,
    'C0': 2.
}

path = './clusters/data/advact/DR5CosmoSims/sim-kit_NemoCCL_A10tSZ_DR5White_ACT-DR5_tenToA0Tuned/NemoCCL_A10tSZ_DR5White_ACT-DR5_tenToA0Tuned/'

lkl_common = { 
    'verbose': True,
    'stop_at_error': True,
    'data': {
        'data_path': path,
        'cat_file': 'NemoCCL_A10tSZ_DR5White_ACT-DR5_tenToA0Tuned_mass.fits',
        'Q_file': 'selFn/QFit.fits',
        'tile_file': 'selFn/tileAreas.txt',
        'rms_file': 'selFn/RMSTab.fits'
    },
    'theorypred': {
        'choose_theory': 'CCL',
        'massfunc_mode': 'ccl',
        'compl_mode': 'erf_diff',
        'md_hmf': '200c',
        'md_ym': '200c'      
    },
    'YM': {
        'Mpivot': 4.25e14
    },
    'selfunc': {
        'SNRcut': 5.,
        'method': 'SNRbased',
        'whichQ': 'injection',
        'resolution': 'downsample',
        'dwnsmpl_bins': 2,
        'save_dwsmpld': False,
    },
    'binning': {
        'z': {
            'zmin': 0.,
            'zmax': 2.,
            'dz': 0.1
        },
        'q': {
            'log10qmin': 0.6,
            'log10qmax': 2.0,
            'dlog10q': 0.25
        },
        'M': {
            'Mmin': 5e13,
            'Mmax': 1e16,
            'dlogM': 0.01
        },
        'exclude_zbin': 2,
    }
}

ccl_baseline = {
    'transfer_function': 'boltzmann_camb',
    'matter_pk': 'halofit',
    'baryons_pk': 'nobaryons',
    'md_hmf': '200c'
}        




info_binned = {
    'params': params,
    'likelihood': {'soliket.BinnedClusterLikelihood': lkl_common},
    'theory': {'soliket.clusters.CCL': ccl_baseline}
}

info_unbinned = {
    'params': params,
    'likelihood': {'soliket.UnbinnedClusterLikelihood': lkl_common},
    'theory': {'soliket.clusters.CCL': ccl_baseline}
}


def test_clusters_import():

    from soliket.clusters import BinnedClusterLikelihood
    from soliket.clusters import UnbinnedClusterLikelihood


def test_clusters_model():

    binned_model = get_model(info_binned)
    unbinned_model = get_model(info_unbinned)


def test_clusters_loglike():

    binned_model = get_model(info_binned)
    unbinned_model = get_model(info_unbinned)

    binned_lnl = binned_model.loglikes({})[0]
    unbinned_lnl = unbinned_model.loglikes({})[0]

    assert np.isfinite(binned_lnl)
    assert np.isfinite(unbinned_lnl)


def test_clusters_prediction():

    binned_model = get_model(info_binned)
    unbinned_model = get_model(info_unbinned)

    binned_model.loglikes({})[0]
    unbinned_model.loglikes({})[0]

    binned_like = binned_model.likelihood['soliket.BinnedClusterLikelihood']
    unbinned_like = unbinned_model.likelihood['soliket.UnbinnedClusterLikelihood']

    binned_pk_intp = binned_like.theory.get_Pk_interpolator() 
    unbinned_pk_intp = unbinned_like.theory.get_Pk_interpolator()
    SZparams = {
        'tenToA0': 1.9e-05,
        'B0': 0.08,
        'C0': 2.,
        'scatter_sz': 0.,
        'bias_sz': 1.  
    }

    Nzq = binned_like._get_theory(binned_pk_intp, **SZparams)
    Ntot = unbinned_like._get_n_expected(unbinned_pk_intp, **SZparams)
    
    assert np.isclose(Nzq.sum(), Ntot)


# test_clusters_import()
# test_clusters_model()
# test_clusters_loglike()
# test_clusters_prediction()  

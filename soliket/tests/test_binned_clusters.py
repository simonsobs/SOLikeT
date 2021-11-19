import numpy as np
from soliket import BinnedClusterLikelihood
from cobaya.model import get_model
import camb
import pytest

@pytest.mark.skip(reason="Under development")
def test_binned_clusters():
    params = {
        'cosmomc_theta': 0.0104135,
        'ns': 0.965,
        'ombh2': 0.0226576,
        'omch2': 0.1206864,
        'As': 2.022662e-9,
        'tenToA0': 4.35e-5,
        'B0': 0.08,
        'scatter_sz': 0.,
        'bias_sz': 1.,
        'tau': 0.055,
        'mnu': 0.0,
        'nnu': 3.046,
        'omnuh2': 0.,
        'w': -1
    }

    info = {
        'params': params,
        'likelihood': {'soliket.BinnedClusterLikelihood': {
            'single_tile_test': "no",
            'choose_dim': "2D",
            'Q_optimise': "yes",
            'stop_at_error': True,
            'data_path': "/Users/eunseonglee/SOLikeT/soliket/binned_clusters/data/so/",
            'cat_file': "MFMF_WebSkyHalos_A10tSZ_3freq_tiles/MFMF_WebSkyHalos_A10tSZ_3freq_tiles_mass.fits",
            'Q_file': "MFMF_WebSkyHalos_A10tSZ_3freq_tiles/selFn/quick_theta_Q.npz",
            'tile_file': "MFMF_WebSkyHalos_A10tSZ_3freq_tiles/selFn/tileAreas.txt",
            'rms_file': "MFMF_WebSkyHalos_A10tSZ_3freq_tiles/selFn/downsampled.txt"}},
        'theory': {'camb': {'extra_args': {'num_massive_neutrinos': 0}}}
    }

    # initialisation
    model = get_model(info)

    lnl = model.loglikes({})[0]
    print('like:', lnl)

    assert np.isfinite(lnl)

    like = model.likelihood['soliket.BinnedClusterLikelihood']
    print('like:', like)

test_binned_clusters()

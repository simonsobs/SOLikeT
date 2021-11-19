import numpy as np
from soliket import BinnedClusterLikelihood
from cobaya.model import get_model
from cobaya.yaml import yaml_load
import camb
import pytest
import os
import tempfile
import pytest

def get_demo_binned_cluster_model():
    info_yaml = r"""
    likelihood:
      solike.BinnedClusterLikelihood:
        single_tile_test: "no"
        choose_dim: "2D"
        Q_optimise: "yes"
        stop_at_error: true
        data_path: /Users/eunseonglee/SOLikeT/soliket/binned_clusters/data/so/
        cat_file: MFMF_WebSkyHalos_A10tSZ_3freq_tiles/MFMF_WebSkyHalos_A10tSZ_3freq_tiles_mass.fits
        Q_file: MFMF_WebSkyHalos_A10tSZ_3freq_tiles/selFn/quick_theta_Q.npz
        tile_file: MFMF_WebSkyHalos_A10tSZ_3freq_tiles/selFn/tileAreas.txt
        rms_file: MFMF_WebSkyHalos_A10tSZ_3freq_tiles/selFn/downsampled.txt
    theory:
      camb:
        stop_at_error: true
        extra_args:
          num_massive_neutrinos: 0
    params:
      logA:
        prior:
          min: 2.
          max: 4.
        ref: 3.007
        latex: \log(10^{10} A_\mathrm{s})
        drop: true
      As:
        value: 'lambda logA: 1e-10*np.exp(logA)'
        latex: A_\mathrm{s}

      theta_MC_100:
        prior:
          min: 0.5
          max: 10
        ref: 1.04135
        latex: 100\theta_\mathrm{MC}
        drop: true
        renames: theta
      cosmomc_theta:
        value: 'lambda theta_MC_100: 1.e-2*theta_MC_100'
        derived: false

      tenToA0: 4.35e-5
      B0: 0.08
      scatter_sz: 0.
      bias_sz: 1.

      ombh2: 0.0226576
      omch2: 0.1206864
      ns: 0.965
      tau: 0.055
      mnu: 0.0
      nnu: 3.046
      omnuh2: 0.
      w: -1
    """

    info = yaml_load(info_yaml)

    test_point = {}

    for par, pdict in info["params"].items():
        if not isinstance(pdict, dict):
            continue

        if "ref" in pdict:
            try:
                value = float(pdict["ref"])
            except TypeError:
                value = (pdict["ref"]["min"] + pdict["ref"]["max"]) / 2
            test_point[par] = value
        elif "prior" in pdict:
            value = (pdict["prior"]["min"] + pdict["prior"]["max"]) / 2
            test_point[par] = value

    model = get_model(info)
    return model, test_point

def test_binned_clusters():
    model, test_point = get_demo_binned_cluster_model()
    lnl = model.loglike(test_point)[0]
    print('lnl binned clusters :', lnl)
    assert np.isfinite(lnl)

test_binned_clusters()

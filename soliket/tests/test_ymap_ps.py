import os
import tempfile

import pytest
import numpy as np

from cobaya.yaml import yaml_load
from cobaya.model import get_model

# packages_path = os.environ.get("COBAYA_PACKAGES_PATH") or os.path.join(
#     tempfile.gettempdir(), "ymap_packages"
# )

# print(packages_path)
# exit(0)

def get_demo_ymap_ps_model():
    # if theory == "camb":
    #     print('The tsz power spectrum likelihood currently requires class_sz')
    #     # info_yaml = r"""
    #     # likelihood:
    #     #     soliket.LensingLikelihood:
    #     #         stop_at_error: True
    #     #
    #     # theory:
    #     #     camb:
    #     #         extra_args:
    #     #             lens_potential_accuracy: 1
    #     #
    #     # params:
    #     #     ns:
    #     #         prior:
    #     #           min: 0.8
    #     #           max: 1.2
    #     #     H0:
    #     #         prior:
    #     #           min: 40
    #     #           max: 100
    #     # """
    # elif theory == "classy":
    info_yaml = r"""

    likelihood:
      soliket.ymap.ymap_ps.SZLikelihood:


    theory:
      soliket.ymap.ymap_ps.SZForegroundTheory:
        speed: 2



      soliket.ymap.classy_sz:
         extra_args:
              non linear: 'halofit'
              output : 'tSZ_1h'
              units for tSZ spectrum : 'dimensionless'
              component of tSZ power spectrum : 'total'
              path_to_class : '/Users/boris/Work/CLASS-SZ/SO-SZ/class_sz'
              write sz results to files : 'no'
              mass function : 'M500'
              pressure profile : 'A10'
              multipoles_sz : 'P15'
              nlSZ : 18

              M1SZ : 1e11
              M2SZ : 1e16

              z1SZ : 1e-5
              z2SZ : 4.
              z_max_pk : 4.

              N_ur : 0.00641
              N_ncdm : 1
              deg_ncdm : 3
              m_ncdm : 0.02
              T_ncdm : 0.71611

              HMF_prescription_NCDM: 'CDM'

              input_verbose : 0
              background_verbose: 0
              perturbations_verbose: 0
              sz_verbose: 0

              create reference trispectrum for likelihood code: 'NO'
              append_name_trispectrum_ref: 'total-planck-collab-15_step_1'
              path to reference trispectrum for likelihood code: '/Users/boris/Work/CLASS-SZ/SO-SZ/Likelihoods_sz/solike/ymap/chains/sz_ps_completeness_analysis/'



    params:
      A_CIB:
          prior:
              min: 0
              max: 5
          ref: 0.66
          latex: A_\mathrm{CIB}

      A_RS:
          prior:
              min: 0
              max: 5
          ref: 0.004
          proposal: 0.34
          latex: A_\mathrm{RS}

      A_IR:
          prior:
              min: 0
              max: 5
          ref: 2.04
          proposal: 0.18
          latex: A_\mathrm{IR}

      B:
          prior:
              min: 1.0
              max: 2.0
          ref: 1.4
          proposal: 0.2
          latex: B

      omega_b:
        prior:
          min: 0.02
          max: 0.025
        ref: 0.0224
        proposal: 0.00015
        latex: \Omega_\mathrm{b} h^2

      omega_cdm:
        prior:
          min: 0.11
          max: 0.13
        ref: 0.1202
        proposal: 0.0014
        latex: \Omega_\mathrm{c} h^2

      A_s:
        prior:
          min: 0.1e-9
          max: 10e-9
        ref: 2.1e-9
        proposal: 0.032e-9
        latex: A_\mathrm{s}

      n_s:
        prior:
          min: 0.94
          max: 1.
        ref: 0.965
        proposal: 0.0044
        latex: n_\mathrm{s}

      H0:
        prior:
          min: 55.
          max: 90.
        ref: 67.27
        proposal: 0.6
        latex: H_0
    """

    info = yaml_load(info_yaml)

    # from cobaya.install import install
    # install(info, path=packages_path, skip_global=True)

    test_point = {}

    for par, pdict in info["params"].items():
        # print(pdict)
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


# @pytest.mark.parametrize("theory", ["camb", "classy"])
def test_ymap_ps():
    model, test_point = get_demo_ymap_ps_model()
    lnl = model.loglike(test_point)[0]
    print('lnl ymap_ps :', lnl)
    assert np.isfinite(lnl)

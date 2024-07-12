# sample grid parameter file

import numpy as np
from cobaya import InputDict
from cobaya.grid_tools.batchjob import DataSet
from cobaya.yaml import yaml_load_file
import copy

default: InputDict = {
    'sampler': {'mcmc': {'Rminus1_stop': 0.01,
                         'Rminus1_cl_stop': 0.2,
                         'drag': False,
                         'max_tries': 10000,
                         'oversample_power': 0.4,
                         'proposal_scale': 1.9}},
}

# dict or list of default settings to combine; each item can be dict or yaml file name
defaults = [default]

importance_defaults = []
minimize_defaults = []
getdist_options = {'ignore_rows': 0.3}

MFLike: InputDict = {'likelihood': yaml_load_file('like_mflike.yaml'),
                     'params': yaml_load_file('params_cosmo_smooth.yaml') | \
                               yaml_load_file('params_mflikefg_smooth.yaml') | \
                               yaml_load_file('params_mflikesyst_smooth.yaml'),
                     'theory': yaml_load_file('theory_camb.yaml') | \
                               yaml_load_file('theory_bandpass.yaml') | \
                               yaml_load_file('theory_foregrounds.yaml') | \
                               yaml_load_file('theory_theoryforge.yaml')
                    }

LensingLike: InputDict = {'likelihood':  yaml_load_file('like_LensingLikelihood.yaml'),
                          'params': yaml_load_file('params_cosmo_smooth.yaml'),
                          'theory': yaml_load_file('theory_camb.yaml') | \
                                    yaml_load_file('theory_ccl.yaml')
                        }

ShearKappaLike: InputDict = {'likelihood':  yaml_load_file('like_shearkappa.yaml'),
                          'params': yaml_load_file('params_cosmo_smooth.yaml') | \
                                    yaml_load_file('params_shearkappanuisance_smooth.yaml'),
                          'theory': yaml_load_file('theory_camb.yaml') | \
                                    yaml_load_file('theory_ccl.yaml')
                        }

joint1 = DataSet(['MFLike', 'LensingLike'], [MFLike, LensingLike])
joint2 = DataSet(['MFLike', 'ShearKappaLike'], [MFLike, ShearKappaLike])
joint3 = DataSet(['LensingLike', 'ShearKappaLike'], [LensingLike, ShearKappaLike])
joint4 = DataSet(['MFLike', 'LensingLike', 'ShearKappaLike'], [MFLike, LensingLike, ShearKappaLike])


groups = {
    'main': {
        'models': ['LCDM', 'Neff'],
        'datasets': [('MFLike', MFLike),
                     ('LensingLike', LensingLike),
                     ('ShearKappaLike', ShearKappaLike),
                     joint1,
                     joint2,
                     joint3,
                     joint4],
        "defaults": {},  # options specific to this group
    }
}

models = {}
models['LCDM'] = {'params': {'nnu': {'value': 3.044}}}
models['Neff'] = {'params': {'nnu': {'prior': {'min': 0, 'max': 10}, 'ref': 3.044}}}

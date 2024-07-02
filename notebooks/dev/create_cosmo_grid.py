# sample grid parameter file
# build grid with "cobaya-grid-create grid_dir simple_grid.py"

import numpy as np
from cobaya import InputDict
from cobaya.grid_tools.batchjob import DataSet
from cobaya.yaml import yaml_load_file

default: InputDict = {
    'sampler': {'mcmc': {'max_samples': 10, 'burn_in': 2, 'covmat': 'auto'}},
}

# dict or list of default settings to combine; each item can be dict or yaml file name
defaults = [default]

importance_defaults = []
minimize_defaults = []
getdist_options = {'ignore_rows': 0.3}

MFLike = yaml_load_file('run_mflike.yaml')
LensingLike = yaml_load_file('run_lensing.yaml')

# DataSet is a combination of likelihoods, list of name tags to identify data components
joint = DataSet(['MFLike', 'LensingLike'], [MFLike, LensingLike])

# Dictionary of groups of data/parameter combination to run
# datasets is a list of DataSet objects, or tuples of data name tag combinations and
# corresponding list of input dictionaries or yaml files.

groups = {
    'main': {
        'models': ['LCDM', 'Neff'],
        'datasets': [('MFLike', MFLike), ('LensingLike', LensingLike), joint],
        "defaults": {},  # options specific to this group
    }
}

models = {}
models['LCDM'] = {'params': {'nnu': {'value': 3.044}}}
models['Neff'] = {'params': {'nnu': {'prior': {'min': 0, 'max': 10}, 'ref': 3.044}}}

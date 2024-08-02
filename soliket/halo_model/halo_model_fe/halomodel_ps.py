#!/usr/bin/env python
# coding: utf-8

import numpy as np
import yaml
from yaml import SafeLoader
from utils import *
from HODS import *
from power_spectrum import *
from lin_matterPS import * 
from cosmology import *

with open("paramfile_halomod.yaml") as f:
    settings = yaml.load(f, Loader=SafeLoader)

#-----------------------------------------------general settings------------------------------------------------
read_matterPS  = settings['options']['read_matterPS']
#redshift_path  = settings['options']['redshift']
gal_mod        = settings['options']['two_populations']
ps_computation = settings['options']['power_spectra']
redshift = np.linspace(settings['options']['zmin'], settings['options']['zmax'], settings['options']['nz'])
print(redshift)
#redshift       = np.loadtxt(redshift_path)

if gal_mod == True:
    print('halo model assuming two galaxy populations')
else:
    print('halo model assuming a single galaxy population')

nl_ps = []

for i in ps_computation:
    if ps_computation[i] != None:
        nl_ps.append(ps_computation[i])
nl_ps = np.array(nl_ps)

#------------------------------------------------paramters setting------------------------------------------------
param               = settings['parameters']
cosmological_param  = param['cosmology']
clust_param         = param['clustering']

#compute cosmological parameters, matter power spectrum
default_lin_matter_PS = './tabulated/matterPS_Planck18.txt'

cosmo_param = cosmo_param(redshift, cosmological_param, cosmo, default_lin_matter_PS)

h, dV_dz = cosmo_param.compute_params()

if read_matterPS == True:
    k_array, Pk_array = cosmo_param.read_matter_PS()
else:
    compute_PS = matter_PS(redshift, h, cosmo_param, cosmological_param)
    k_array = compute_PS.lin_matter_PS()[0]
    Pk_array = compute_PS.lin_matter_PS()[2]


#----------------------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------Other settings-------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------
# set mass range
logmass = np.linspace(settings['options']['Mmin'], settings['options']['Mmax'], settings['options']['nm'])
mh      = 10 ** logmass / (h ** -1)

# set the mass overdensity
delta_200 = 200

#----------------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------Utils computation-------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------

#compute utils
instance_200 = u_p_nfw_hmf_bias(k_array, Pk_array, mh, redshift, delta_200)
instance_HOD = hod_ngal(mh, redshift, clust_param, instance_200)

#----------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------Power spectra computation---------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------

spectra = mm_gg_mg_spectra(
        k_array,
        Pk_array,
        mh,
        redshift,
        instance_HOD,
        instance_200,
        gal_mod
    )

for ps in nl_ps:
    if ps=='mm':
        print('computing non-linear matter-matter ps')
        mm_tot, mm_1h, mm_2h = spectra.halo_terms_matter()
    if ps=='gg':
        if gal_mod==True:
            print('computing non-linear galaxy-galaxy ps with two galaxy populations')
            Pgal, Pk_1h_EP, Pk_1h_LP, Pk_1h_mix, Pk_2h_EP, Pk_2h_LP, Pk_2h_mix = spectra.halo_terms_galaxy()
        else:
            print('computing non-linear galaxy-galaxy ps with a single galaxy population')
            Pgal, Pk_1h, Pk_2h = spectra.halo_terms_galaxy()
    if ps=='mg':
        print('computing non-linear matter-galaxy ps')


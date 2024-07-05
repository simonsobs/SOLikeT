import numpy as np
from numpy import trapz


class mm_gg_mg_spectra:
    def __init__(
        self,
        k_array,
        Pk_array,
        mh,
        redshift,
        instance_HOD,
        instance_200,
        gal_mod,
    ):

        self.k_array           = k_array
        self.Pk_array          = Pk_array
        self.mass              = mh
        self.redshift          = redshift
        self.instance_HOD      = instance_HOD
        self.instance_200      = instance_200
        self.gal_mod           = gal_mod

    def halo_terms_matter(self):
        rho_mean = self.instance_200.mean_density()

        intmass_1h = np.zeros([len(self.k_array), len(self.redshift), len(self.mass)])
        intmass_2h = np.zeros([len(self.k_array), len(self.redshift), len(self.mass)])

        Pk_1h      = np.zeros([len(self.k_array), len(self.redshift)])
        Pk_2h      = np.zeros([len(self.k_array), len(self.redshift)])

        for k in range(len(self.k_array)):
            intmass_1h[k,:,:] = self.instance_200.dndM * (self.mass[np.newaxis, :] * self.instance_200.u_k[:,:,k]) ** 2
            intmass_2h[k,:,:] = self.instance_200.bias_cib * self.instance_200.dndM * self.mass[np.newaxis, :] * self.instance_200.u_k[:,:,k]

            B = 1. - trapz(self.instance_200.bias_cib * self.instance_200.dndM * self.mass[np.newaxis, :]  / rho_mean, self.mass, axis=-1) 
            
            Pk_1h[k,:] = trapz(intmass_1h[k,:,:], self.mass, axis=-1) / rho_mean ** 2
            Pk_2h[k,:] = self.Pk_array[:,k] * (trapz(intmass_2h[k,:,:], self.mass, axis=-1)/ rho_mean + B) ** 2 

        Pk_mm = Pk_1h + Pk_2h

        return Pk_mm, Pk_1h, Pk_2h

    def halo_terms_galaxy(self):
        if self.gal_mod == True:
            intmass_2h_EP  = np.zeros([len(self.k_array), len(self.redshift), len(self.mass)])
            intmass_1h_EP  = np.zeros([len(self.k_array), len(self.redshift), len(self.mass)])
            intmass_2h_LP  = np.zeros([len(self.k_array), len(self.redshift), len(self.mass)])
            intmass_1h_LP  = np.zeros([len(self.k_array), len(self.redshift), len(self.mass)])
            intmass_1h_mix = np.zeros([len(self.k_array), len(self.redshift), len(self.mass)])

            Pk_1h_EP = np.zeros([len(self.k_array), len(self.redshift)])
            Pk_1h_LP = np.zeros([len(self.k_array), len(self.redshift)])
            Pk_1h_mix = np.zeros([len(self.k_array), len(self.redshift)])
            Pk_2h_EP = np.zeros([len(self.k_array), len(self.redshift)])
            Pk_2h_LP = np.zeros([len(self.k_array), len(self.redshift)])
            Pk_2h_mix = np.zeros([len(self.k_array), len(self.redshift)])

            for k in range(len(self.k_array)):
                intmass_1h_EP[k,:,:] = self.instance_200.dndM * (2 * self.instance_HOD.Ncent_EP[np.newaxis, :]
                                     * self.instance_HOD.Nsat_EP[np.newaxis, :] 
                                     * self.instance_200.u_k[:,:,k] 
                                     + self.instance_HOD.Nsat_EP[np.newaxis, :] **2
                                     * self.instance_200.u_k[:,:,k] **2)

                intmass_1h_LP[k,:,:] = self.instance_200.dndM * (2 * self.instance_HOD.Ncent_LP[np.newaxis, :]
                                     * self.instance_HOD.Nsat_LP[np.newaxis, :] 
                                     * self.instance_200.u_k[:,:,k] 
                                     + self.instance_HOD.Nsat_LP[np.newaxis, :] **2
                                     * self.instance_200.u_k[:,:,k] **2) 

                intmass_1h_mix[k,:,:] = self.instance_200.dndM * (
                                       (self.instance_HOD.Ncent_EP[np.newaxis, :] * self.instance_HOD.Nsat_LP[np.newaxis, :] 
                                      + self.instance_HOD.Nsat_EP[np.newaxis, :]  * self.instance_HOD.Ncent_LP[np.newaxis, :])
                                      * self.instance_200.u_k[:,:,k] 
                                      + self.instance_HOD.Nsat_EP[np.newaxis, :] * self.instance_HOD.Nsat_LP[np.newaxis, :] 
                                      * self.instance_200.u_k[:,:,k] ** 2) 

                intmass_2h_EP[k,:,:] = self.instance_200.dndM * self.instance_200.bias_cib * self.instance_HOD.Nbra_EP[np.newaxis, :] * self.instance_200.u_k[:,:,k]

                intmass_2h_LP[k,:,:] = self.instance_200.dndM * self.instance_200.bias_cib * self.instance_HOD.Nbra_LP[np.newaxis, :] * self.instance_200.u_k[:,:,k]

                Pk_1h_EP[k,:]  = trapz(intmass_1h_EP[k,:,:], self.mass, axis=-1) / self.instance_HOD.ngal_EP_200c **2
                Pk_1h_LP[k,:]  = trapz(intmass_1h_LP[k,:,:], self.mass, axis=-1) / self.instance_HOD.ngal_LP_200c **2
                Pk_1h_mix[k,:] = trapz(intmass_1h_mix[k,:,:], self.mass, axis=-1) / (self.instance_HOD.ngal_EP_200c * self.instance_HOD.ngal_LP_200c)

                Pk_2h_EP[k,:] = self.Pk_array[:,k] * (trapz(intmass_2h_EP[k,:,:], self.mass, axis=-1)) ** 2 / self.instance_HOD.ngal_EP_200c ** 2
                Pk_2h_LP[k,:] = self.Pk_array[:,k] * (trapz(intmass_2h_LP[k,:,:], self.mass, axis=-1)) ** 2 / self.instance_HOD.ngal_LP_200c ** 2
                Pk_2h_mix[k,:] = self.Pk_array[:,k] * trapz(intmass_2h_EP[k,:,:], self.mass, axis=-1) * trapz(intmass_2h_LP[k,:,:], self.mass, axis=-1) / (self.instance_HOD.ngal_EP_200c * self.instance_HOD.ngal_LP_200c)

            Pgal = Pk_1h_EP + Pk_1h_LP + Pk_1h_mix + Pk_2h_EP + Pk_2h_LP + Pk_2h_mix

            return Pgal, Pk_1h_EP, Pk_1h_LP, Pk_1h_mix, Pk_2h_EP, Pk_2h_LP, Pk_2h_mix

        else:
            intmass_1h = np.zeros([len(self.k_array), len(self.redshift), len(self.mass)])
            intmass_2h = np.zeros([len(self.k_array), len(self.redshift), len(self.mass)])

            Pk_1h      = np.zeros([len(self.k_array), len(self.redshift)])
            Pk_2h      = np.zeros([len(self.k_array), len(self.redshift)])

            for k in range(len(self.k_array)):
                intmass_1h[k,:,:] = self.instance_200.dndM * (2 * self.instance_HOD.Ncent_EP[np.newaxis, :]
                                  * self.instance_HOD.Nsat_EP[np.newaxis, :] 
                                  * self.instance_200.u_k[:,:,k] 
                                  + self.instance_HOD.Nsat_EP[np.newaxis, :] **2
                                  * self.instance_200.u_k[:,:,k] **2)

                intmass_2h[k,:,:] = self.instance_200.dndM * self.instance_200.bias_cib * self.instance_HOD.Nbra_EP[np.newaxis,:] * self.instance_200.u_k[:,:,k]

                Pk_1h[k,:] = trapz(intmass_1h[k,:,:], self.mass, axis=-1)  / self.instance_HOD.ngal_EP_200c ** 2
                Pk_2h[k,:] = self.Pk_array[:,k] * (trapz(intmass_2h[k,:,:], self.mass, axis=-1)) ** 2 / self.instance_HOD.ngal_EP_200c ** 2

            Pgal = Pk_1h + Pk_2h

            return Pgal, Pk_1h, Pk_2h
        
    def halo_terms_matter_galaxy(self):
        rho_mean  = self.instance_200.mean_density()
        ngal_mean = self.instance_HOD.ngal_EP_200c

        intmass_1h   = np.zeros([len(self.k_array), len(self.redshift), len(self.mass)])
        intmass_2h_m = np.zeros([len(self.k_array), len(self.redshift), len(self.mass)])
        intmass_2h_g = np.zeros([len(self.k_array), len(self.redshift), len(self.mass)])

        Pk_1h      = np.zeros([len(self.k_array), len(self.redshift)])
        Pk_2h      = np.zeros([len(self.k_array), len(self.redshift)])

        for k in range(len(self.k_array)):
            intmass_1h[k,:,:]   = self.instance_200.dndM * self.instance_HOD.Nbra_EP[np.newaxis,:] * self.mass[np.newaxis,:] * self.instance_200.u_k[:,:,k] ** 2
            intmass_2h_m[k,:,:] = self.instance_200.dndM * self.instance_200.bias_cib * self.mass[np.newaxis,:] * self.instance_200.u_k[:,:,k]
            intmass_2h_g[k,:,:] = self.instance_200.dndM * self.instance_200.bias_cib * self.instance_HOD.Nbra_EP[np.newaxis,:] * self.instance_200.u_k[:,:,k]

            B = 1. - trapz(self.instance_200.bias_cib * self.instance_200.dndM * self.mass[np.newaxis, :]  / rho_mean, self.mass, axis=-1)

            Pk_1h[k,:] = trapz(intmass_1h[k,:,:], self.mass, axis=-1) / (ngal_mean * rho_mean)
            Pk_2h[k,:] = self.Pk_array[:,k] * (trapz(intmass_2h_m[k,:,:], self.mass, axis=-1) / rho_mean + B) * trapz(intmass_2h_g[k,:,:], self.mass, axis=-1) / ngal_mean

        Pmg = Pk_1h + Pk_2h

        return Pmg, Pk_1h, Pk_2h

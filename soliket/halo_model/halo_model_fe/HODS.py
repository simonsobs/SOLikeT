import numpy as np
from scipy import special
from numpy import trapz


class hod_ngal:
    def __init__(self, mh, redshift, clust_param, instance_200):
        self.mh           = mh
        self.redshift     = redshift
        self.clust_param  = clust_param
        self.instance_200 = instance_200

        self.HODS_EP()
        self.mean_gal_EP()
        self.HODS_LP()
        self.mean_gal_LP()

    def HODS_EP(self):
        Ncent = np.zeros([len(self.mh)])
        Nsat = np.zeros([len(self.mh)])
        Nbra = np.zeros([len(self.mh)])

        Mmin = 10 ** self.clust_param['LogMmin_EP']
        Msat = self.clust_param['scale_EP'] * Mmin
        Ncent = 0.5 * (1 + special.erf((np.log10(self.mh) - np.log10(Mmin)) / self.clust_param['sigma_EP']))
        Nsat = (
            0.5
            * (1 + special.erf((np.log10(self.mh) - np.log10(2 * Mmin)) / self.clust_param['sigma_EP']))
            * ((self.mh) / Msat) ** self.clust_param['alpha_EP']
        )
        Nbra = Ncent + Nsat

        self.Nbra_EP  = Nbra
        self.Ncent_EP = Ncent
        self.Nsat_EP  = Nsat

        return Ncent, Nsat, Nbra

    def mean_gal_EP(self):

        Nbra = self.HODS_EP()[2]
        ngal_200c = trapz(self.instance_200.dndM[:, :] * Nbra[np.newaxis, :], self.mh[:])

        self.ngal_EP_200c = ngal_200c

        return

    def HODS_LP(self):
        Ncent = np.zeros([len(self.mh)])
        Nsat = np.zeros([len(self.mh)])
        Nbra = np.zeros([len(self.mh)])

        Mmin = 10 ** self.clust_param['LogMmin_LP']
        Msat = self.clust_param['scale_LP'] * Mmin
        Ncent = 0.5 * (1 + special.erf((np.log10(self.mh) - np.log10(Mmin)) / self.clust_param['sigma_LP']))
        Nsat = (
            0.5
            * (1 + special.erf((np.log10(self.mh) - np.log10(2 * Mmin)) / self.clust_param['sigma_LP']))
            * ((self.mh) / Msat) ** self.clust_param['alpha_LP']
        )
        Nbra = Ncent + Nsat

        self.Nbra_LP  = Nbra
        self.Ncent_LP = Ncent
        self.Nsat_LP  = Nsat

        return Ncent, Nsat, Nbra

    def mean_gal_LP(self):

        Nbra = self.HODS_LP()[2]

        ngal_200c = trapz(self.instance_200.dndM[:, :] * Nbra[np.newaxis, :], self.mh[:])

        self.ngal_LP_200c = ngal_200c

        return

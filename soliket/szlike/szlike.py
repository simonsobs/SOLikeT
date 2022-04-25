'''
Likelihood for SZ model
'''
import numpy as np
from ..gaussian import GaussianData, GaussianLikelihood
from .projection_functions import project_ksz, project_tsz

class SZLikelihood(GaussianLikelihood):
    def initialize(self):

        self.beam_txt = self.beam_file
        self.z = self.redshift
        self.nu = self.frequency_GHz
        self.M = self.mass_halo_mean_Msol 
        self.input_model = self.input_model

        x,y,dy = self._get_data()
        cov = np.diag(dy**2) #come back to this #sr2sqarcmin
        self.data = GaussianData("SZModel",x,y,cov)

    def logp(self,**params_values):
        theory = self._get_theory(**params_values)
        return self.data.loglike(theory)

class KSZLikelihood(SZLikelihood):

    def _get_data(self,**params_values):
        thta_arc,ksz_data = np.loadtxt(self.sz_data_file,usecols=(0,1),unpack=True)
        cov_ksz = np.loadtxt(self.cov_ksz_file) #units muK*sr

        self.thta_arc = thta_arc
        self.ksz_data = ksz_data
        self.dy_ksz = np.sqrt(np.diag(cov_ksz)) * 3282.8 * 60.**2 #units to muK*sqarcmin
        return self.thta_arc,self.ksz_data,self.dy_ksz

    def _get_theory(self,**params_values):
        model_params = {
            "gnfw":[np.log10(params_values['gnfw_rho0']),params_values['gnfw_al_ksz']
                ,params_values['gnfw_bt_ksz'],params_values['gnfw_A2h_ksz']],
            "obb":["similar_array"]
        }
        model_params = model_params.get(self.input_model)

        rho = np.zeros(len(self.thta_arc))
        for ii in range(len(self.thta_arc)):
            rho[ii] = project_ksz(self.thta_arc[ii], self.M, self.z, self.beam_txt, self.input_model, model_params, self.provider)
        return rho

class TSZLikelihood(SZLikelihood):

    def _get_data(self,**params_values):
        thta_arc,tsz_data = np.loadtxt(self.sz_data_file,usecols=(0,2),unpack=True) #do we need two separate get data functions?
        cov_tsz = np.loadtxt(self.cov_tsz_file) #units muK*sr

        self.thta_arc = thta_arc
        self.tsz_data = tsz_data
        self.dy_tsz = np.sqrt(np.diag(cov_tsz)) * 3282.8 * 60.**2 #units to muK*sqarcmin
        return self.thta_arc,self.tsz_data,self.dy_tsz

    def _get_theory(self,**params_values):
        model_params = {
            "gnfw":[params_values['gnfw_P0'],params_values['gnfw_xc_tsz']
            ,params_values['gnfw_bt_tsz'],params_values['gnfw_A2h_tsz']],
            "obb":["similar_array"]
        }
        model_params = model_params.get(self.input_model)

        pth = np.zeros(len(self.thta_arc))
        for ii in range(len(self.thta_arc)):
            pth[ii] = project_tsz(self.thta_arc[ii], self.M, self.z, self.nu, self.beam_txt, self.input_model, model_params, self.provider)
        return pth

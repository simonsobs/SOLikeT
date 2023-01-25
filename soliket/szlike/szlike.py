'''
Likelihood for SZ model
'''
import numpy as np
from ..gaussian import GaussianData, GaussianLikelihood
from .projection_functions import project_ksz, project_tsz, project_obb

class SZLikelihood(GaussianLikelihood):
    def initialize(self):

        self.beam_txt = self.beam_file
        self.z = self.redshift
        self.nu = self.frequency_GHz
        self.M = self.mass_halo_mean_Msol 
        #self.input_model = self.input_model
        self.beam_response = self.beam_response

        x,y,cov = self._get_data()
        self.data = GaussianData("SZModel",x,y,cov)

    def logp(self,**params_values):
        theory = self._get_theory(**params_values)
        return self.data.loglike(theory)

class KSZLikelihood(SZLikelihood): #this is for GNFW model

    def _get_data(self,**params_values):
        thta_arc,ksz_data = np.loadtxt(self.sz_data_file,usecols=(0,1),unpack=True)
        ksz_data /= 3282.8 * 60.**2 #units muK*sr
        cov_ksz = np.loadtxt(self.cov_ksz_file) #units muK*sr

        self.thta_arc = thta_arc
        self.ksz_data = ksz_data
        self.cov = cov_ksz
        return self.thta_arc,self.ksz_data,self.cov

    def _get_theory(self,**params_values):
        model_params = [params_values['gnfw_rho0'],params_values['gnfw_bt_ksz']
                        ,params_values['gnfw_A2h_ksz']]
        #model_params = model_params.get(self.input_model)

        rho = np.zeros(len(self.thta_arc))
        for ii in range(len(self.thta_arc)):
            rho[ii] = project_ksz(self.thta_arc[ii], self.M, self.z, self.beam_txt, model_params, self.provider) #self.input_model
        return rho

class TSZLikelihood(SZLikelihood): #this is for GNFW model

    def _get_data(self,**params_values):
        thta_arc,tsz_data = np.loadtxt(self.sz_data_file,usecols=(0,2),unpack=True)
        tsz_data /= 3282.8 * 60.**2 #units muK*sr 
        cov_tsz = np.loadtxt(self.cov_tsz_file) #units muK*sr
        
        self.thta_arc = thta_arc
        self.tsz_data = tsz_data
        self.cov = cov_tsz
        return self.thta_arc,self.tsz_data,self.cov

    def _get_theory(self,**params_values):
        model_params = [params_values['gnfw_P0'],params_values['gnfw_bt_tsz']
                        ,params_values['gnfw_A2h_tsz']]

        pth = np.zeros(len(self.thta_arc))
        for ii in range(len(self.thta_arc)):
            pth[ii] = project_tsz(self.thta_arc[ii], self.M, self.z, self.nu, self.beam_txt, model_params, self.beam_response, self.provider)
        return pth

class OBBLikelihood(SZLikelihood): #OBB model, tSZ and kSZ together

    def _get_data(self,**params_values):
        thta_arc,ksz_data,tsz_data = np.loadtxt(self.sz_data_file,usecols=(0,1,2),unpack=True)
        ksz_data /= 3282.8 * 60.**2
        tsz_data /= 3282.8 * 60.**2
        cov_ksz = np.loadtxt(self.cov_ksz_file) #different cov file? combined?
        cov_tsz = np.loadtxt(self.cov_tsz_file)

        self.thta_arc = thta_arc
        self.ksz_data = ksz_data
        self.tsz_data = tsz_data
        self.cov_ksz = cov_ksz
        self.cov_tsz = cov_tsz
        return self.thta_arc,self.ksz_data,self.tsz_data,self.cov_ksz,self.cov_tsz

    def _get_theory(self,**params_values):
        model_params = [params_values['obb_Gamma'],params_values['obb_alpha_Nth']
                ,params_values['obb_logE'],params_values['obb_Ak2h']
                ,params_values['obb_At2h']]

        rho = np.zeros(len(self.thta_arc))
        pth = np.zeros(len(self.thta_arc))
        for ii in range(len(self.thta_arc)):
            temp = project_obb(self.thta_arc[ii], self.M, self.z, self.beam_txt, model_params, self.provider)
            rho[ii] = temp[0]
            pth[ii] = temp[1]
        return rho,pth

'''
Likelihood for SZ model
'''

from .gaussian import GaussianData, GaussianLikelihood

class SZLikelihood(GaussianLikelihood):
    def initialize(self):

        self.beam=np.loadtxt(self.beam_file) #come back to this, interp function?
        self.z=self.redshift
        self.nu=self.frequency_GHz
        self.M= self.mass_halo_mean_Msol 

        x,y,dy=self._get_data()
        cov=np.diag(dy**2)
        self.data=GaussianData("SZModel",x,y,cov)

    def logp(self,**params_values):
        theory=self._get_theory(**params_values)
        return self.data.loglike(theory)

class KSZLikelihood(SZLikelihood):

    def _get_data(self,**params_values):
        thta_arc,ksz_data=np.loadtxt(self.sz_data_file,usecols=(0,1),unpack=True) #do we need two separate get data functions?
        dy_ksz=np.loadtxt(self.cov_ksz_file)
        #figure out units for dy, need to multiply by sr2sqarcmin
        self.ksz_data= ksz_data
        self.dy_ksz= dy_ksz
        return thta_arc,ksz_data,dy_ksz

    def _get_theory(self,**params_values):

        gnfw_params=[params_values['gnfw_rho0'],params_values['gnfw_al_ksz']
                ,params_values['gnfw_bt_ksz'],params_values['gnfw_A2h_ksz']]
        #define thta_arc=x somewhere
        rho = np.zeros(len(thta_arc))
        for ii in range(len(thta_arc)):
            rho[ii] = project_ksz(thta_arc[ii], self.M, self.z, fbeam, gnfw_params)

        return rho

class TSZLikelihood(SZLikelihood):

    def _get_data(self,**params_values):
        thta_arc,tsz_data=np.loadtxt(self.sz_data_file,usecols=(0,2),unpack=True) #do we need two separate get data functions?
        dy_tsz=np.loadtxt(self.cov_tsz_file)
        #figure out units for dy, need to multiply by sr2sqarcmin
        self.tsz_data= tsz_data
        self.dy_tsz= dy_tsz
        return thta_arc,tsz_data,dy_tsz

    def _get_theory(self,**params_values):

        gnfw_params=[params_values['gnfw_P0'],params_values['gnfw_xc_tsz']
                ,params_values['gnfw_bt_tsz'],params_values['gnfw_A2h_tsz']]
        #define thta_arc=x somewhere
        pth = np.zeros(len(thta_arc))
        for ii in range(len(thta_arc)):
            pth[ii] = project_tsz(thta_arc[ii], self.M, self.z, self.nu, fbeam, gnfw_params)

        return pth

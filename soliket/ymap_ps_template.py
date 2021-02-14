"""
.. module:: sz_template

:Synopsis: Definition of simplistic y-map power spectrum likelihood using pre-computed frequency dependent  templates
           for sz powerspectrum (A_sz), and foregrounds, e.g., A_cib, A_ir, A_rs, A_cn.

:Author: Boris Bolliet 2020

:running: $ /Users/boris/opt/anaconda3/bin/mpirun -np 4 /Users/boris/opt/anaconda3/bin/cobaya-run solike/ymap/input_files/sz_template_input.yaml -f
"""


from cobaya.theory import Theory
from cobaya.conventions import _packages_path
from solike.gaussian import GaussianLikelihood
import numpy as np
import os
from scipy.ndimage.interpolation import shift
from typing import Optional, Sequence
from scipy.linalg import block_diag


# Acronyms:
# ps: power spectrum
# fg: foregrounds
# cib: cosmic infrared background
# rs: radio sources
# ir: infrared sources
# f_sky: sky fraction

class SZ_PS_template_Likelihood(GaussianLikelihood):
    sz_data_directory: Optional[str] = None
    ymap_ps_file: Optional[str] = None
    f_sky: Optional[str] = None
    trispectrum_directory: Optional[str] = None
    trispectrum_ref: Optional[str] = None
    use_trispectrum: Optional[str] = None

    # Load the templates
    def initialize(self):
        self.data_directory = self.sz_data_directory
        self.datafile = self.ymap_ps_file

        D = np.loadtxt(os.path.join(self.data_directory, self.datafile))

        # fill arrays with binned Planck tSZ and FG data
        # these is the data points to fit

        # multipoles of bin centre
        self.ell = D[:,0]
        # tSZ + foregrounds
        self.y2AndFg = D[:,1]
        # Gaussian error, includes noise from the map
        self.sigma_tot = D[:,2]


        # Now start building the covariance matrix:
        # diagonal terms has gaussian contribution:
        # this includes the fsky factor.
        self.cvg = np.asarray(self.sigma_tot)**2.
        self.cvg = np.diag(self.cvg)
        # and non-gaussian contribution is the trispectrum
        # computed by class_sz  in the file
        # tSZ_trispectrum_ref_XXX.txt
        self.covmat = np.asarray(np.loadtxt(os.path.join(self.trispectrum_directory, self.trispectrum_ref)))
        # if requested:
        # Add sigm_NG (trispectrum strored in self.covmat) to sigma_G (stored in self.cvg)
        if self.use_trispectrum == 'yes':
            print('Using both Gaussian and Non-Gaussian sampling variance')
            # put the correct Omega_sky,f_sky factor for the non gaussian part
            self.covmat = self.covmat/self.f_sky/4./np.pi + self.cvg
        # if not, just use gaussian terms
        else:
            print('Using only Gaussian sampling variance')
            self.covmat = self.cvg

        # now compute the iverse and det of covariance matrix
        self.inv_covmat = np.linalg.inv(self.covmat)
        self.det_covmat = np.linalg.det(self.covmat)
        #print(np.linalg.eig(self.covmat))
        print('[reading ref tSZ files] read-in completed.')



        super().initialize()


    def get_requirements(self):
        return {"cl_yy_theory": {}}

    # this is the data to fit
    def _get_data(self):
        x_data = self.ell
        y_data = self.y2AndFg
        return x_data, y_data

    def _get_cov(self):
        cov = self.covmat
        return cov


    def _get_theory(self, **params_values):
        cl_yy_theory = self.provider.get_cl_yy_theory()
        #difference = self.y2AndFg-cl_yy_theory
        # print(difference)
        # print(self.sigma)

        #AT = np.dot(difference, np.dot(self.inv_covmat, difference))
        #chi2 = AT
        #print(- 0.5*chi2 -0.5*np.log(self.det_covmat))
        #print(chi2)
        #print(np.sum(self.I0-DI_theory))
        return cl_yy_theory



class SZ_PS_template_Theory(Theory):

    params = {"A_sz": 0, "A_cib": 0, "A_ir": 0, "A_rs": 0, "alpha_cib": 0,}
    use_2halo: Optional[str] = None
    tsz_template_file : Optional[str] = None
    tsz_template_directory : Optional[str] = None
    foreground_template_directory: Optional[str] = None
    foreground_template_file: Optional[str] = None
    ell_pivot_cib : Optional[str] = None

    def initialize(self):

        #read-in the fiducial data for tSZ
        D = np.loadtxt(os.path.join(self.tsz_template_directory, self.tsz_template_file))
        # tSZ template data points
        # e.g., from the file: tSZ_c_ell_ref_XXX.txt computed by class_sz
        # assuming these are computed at the same bin as the data
        cl_sz_1h = D[:,1]
        cl_sz_2h = D[:,2]
        if self.use_2halo == 'yes':
            print('Using both 1-halo and 2-halo terms')
            self.tSZ_template = cl_sz_1h + cl_sz_2h
        # if not, just use gaussian terms
        else:
            print('Using only 1halo term')
            self.tSZ_template = cl_sz_1h



        # similarly read in the foreground templates:

        D = np.loadtxt(os.path.join(self.foreground_template_directory, self.foreground_template_file))
        self.ell = D[:,0]
        self.A_CIB_MODEL = D[:,1]
        self.A_RS_MODEL = D[:,2]
        self.A_IR_MODEL = D[:,3]
        self.A_CN_MODEL = D[:,4]



    def calculate(self, state, want_derived=False, **params_values_dict):


        A_sz = params_values_dict['A_sz']
        A_cib = params_values_dict['A_cib']
        A_ir = params_values_dict['A_ir']
        A_rs = params_values_dict['A_rs']
        A_cn = 0.9033

        alpha_cib = params_values_dict['alpha_cib']

        tsz=A_sz*self.tSZ_template
        cib=A_cib*self.A_CIB_MODEL*(self.ell/self.ell_pivot_cib)**alpha_cib
        ir=A_ir*self.A_IR_MODEL
        rs=A_rs*self.A_RS_MODEL
        cn=A_cn*self.A_CN_MODEL




        state["cl_yy_theory"] = tsz + cib + ir + rs + cn


    def get_cl_yy_theory(self):
        return self._current_state['cl_yy_theory']

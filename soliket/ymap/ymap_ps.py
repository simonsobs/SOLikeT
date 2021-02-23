"""
.. module:: sz power spectrm likelihood (in progress)

"""

# Acronyms:
# ps: power spectrum
# fg: foregrounds
# cib: cosmic infrared background
# rs: radio sources
# ir: infrared sources
# f_sky: sky fraction


from cobaya.theory import Theory
from cobaya.conventions import _packages_path
from soliket.gaussian import GaussianLikelihood
import numpy as np
import os
from scipy.ndimage.interpolation import shift
from typing import Optional, Sequence
# from soliket.ymap.classy_sz import classy_sz


class SZLikelihood(GaussianLikelihood):
    sz_data_directory: Optional[str] = None
    ymap_ps_file: Optional[str] = None
    f_sky: Optional[str] = None
    trispectrum_directory: Optional[str] = None
    trispectrum_ref: Optional[str] = None
    use_trispectrum: Optional[str] = None


    def initialize(self):
        # print('Initialize')
        # exit(0)
        self.data_directory = self.sz_data_directory
        self.datafile = self.ymap_ps_file

        D = np.loadtxt(os.path.join(self.data_directory, self.datafile))

        #fill arrays with Planck tSZ and FG data
        self.ell_plc = D[:,0]
        self.y2AndFg_plc = D[:,1]
        self.sigma_tot_plc = D[:,2]


        #number of data points (multipole bins)
        self.num_points = np.shape(self.ell_plc)[0]


        self.fid_values_exist = False

        fiducial_file_covmat	 =  self.trispectrum_ref
        fiducial_file_cell       =  self.trispectrum_ref.replace("trispectrum","c_ell")
        fiducial_file_params = self.trispectrum_ref.replace("trispectrum","params")



        self.fiducial_file_params = fiducial_file_params
        self.fiducial_file_cell = fiducial_file_cell
        self.fiducial_file_covmat = fiducial_file_covmat
        #If reference trispectrun is available -> read in the values
        if os.path.exists(os.path.join(self.trispectrum_directory, self.fiducial_file_params)):
            self.fid_values_exist = True
            #read-in the fiducial data
            D = np.loadtxt(os.path.join(self.trispectrum_directory, self.fiducial_file_cell))
            self.ell = D[:,0]
            self.data = D[:,1]

            #check that the reference tSZ data is computed at same multipoles as planck data
            try:
                if(np.any(self.ell-self.ell_plc)):
                    print("[reading ref tSZ files] the reference multipoles do not match the planck data.")
                    exit(0)
            except ValueError:
                print("[reading ref tSZ files] the reference multipoles do not match the planck data.")
                exit(0)

            #Binning of covmat [not used for planck sz]
            #compute number of multipoles in each bin
            #[useful when we need to comput ethe gaussian sample variance]
            #[not used in the original planck likelihood as sigma_G is tabulated]
            #multipoles_bin_center = self.ell
            #mp_shift = shift(multipoles_bin_center, -1, cval=np.NaN)
            #diff_ell_shift =  mp_shift - multipoles_bin_center
            #diff_ell_up = diff_ell_shift
            #diff_ell_up[len(diff_ell_shift)-1] = diff_ell_shift[len(diff_ell_shift)-2]
            #mp_shift = shift(multipoles_bin_center, 1, cval=np.NaN)
            #diff_ell_shift =  -mp_shift + multipoles_bin_center
            #diff_ell_down = diff_ell_shift
            #diff_ell_down[0] = diff_ell_shift[1]
            #nell =  (diff_ell_up + diff_ell_down)/2.
            #Compute sigma_G^2
            #self.cvg =1./self.f_sky*(np.asarray(self.data)*np.sqrt(2./(2.*np.asarray(self.ell)+1.)))**2./nell

            #For the planck likelihood we read-in the values of sigma_G:
            self.cvg = np.asarray(self.sigma_tot_plc)**2.
            self.cvg = np.diag(self.cvg)
            self.covmat = np.asarray(np.loadtxt(os.path.join(self.trispectrum_directory, self.fiducial_file_covmat)))
            #Add sigm_NG (trispectrum strored in self.covmat) to sigma_G (stored in self.cvg)
            if self.use_trispectrum == 'yes':
                print('Using both Gaussian and Non-Gaussian sampling variance')
                self.covmat = self.covmat/self.f_sky/4./np.pi + self.cvg
            else:
                print('Using only Gaussian sampling variance')
                self.covmat = self.cvg
            #print(self.covmat)
            self.inv_covmat = np.linalg.inv(self.covmat)
            self.det_covmat = np.linalg.det(self.covmat)
            #print(np.linalg.eig(self.covmat))
            print('[reading ref tSZ files] read-in completed.')
        else:
            print('[reading ref tSZ files] reference trispectrum unavailable -> you need to create a cobaya_reference_trispectrum.')
            self.covmat = np.identity(self.num_points)
        super().initialize()


    def get_requirements(self):
        return {"Cl_sz_foreground": {}, "Cl_sz": {}}

    def _get_data(self):
        ell = self.ell_plc
        Cl_sz = self.y2AndFg_plc
        return ell, Cl_sz

    def _get_cov(self):
        cov = self.covmat
        return cov


    def _get_theory(self, **params_values):
        theory = self.theory.get_Cl_sz()
        # theory = classy_sz.get_Cl_sz()
        cl_1h_theory = theory['1h']
        cl_2h_theory = theory['2h']
        Cl_sz = np.asarray(list(cl_1h_theory)) + np.asarray(list(cl_2h_theory))
        Cl_sz_foreground = self.provider.get_Cl_sz_foreground()
        return Cl_sz + Cl_sz_foreground



class SZForegroundTheory(Theory):

    params = {"A_CIB": 0, "A_RS": 0, "A_IR": 0}

    foreground_data_directory: Optional[str] = None
    foreground_data_file: Optional[str] = None

    def initialize(self):
        self.datafile = self.foreground_data_file
        self.data_directory = self.foreground_data_directory

        D = np.loadtxt(os.path.join(self.data_directory, self.datafile))

        #fill arrays with Planck FG data
        self.A_CIB_MODEL = D[:,1]
        self.A_RS_MODEL = D[:,2]
        self.A_IR_MODEL = D[:,3]
        self.A_CN_MODEL = D[:,4]


    def calculate(self, state, want_derived=False, **params_values_dict):
        #...
        #foreground residua amplitudes [TBD]
        A_CIB = params_values_dict['A_CIB']
        A_RS = params_values_dict['A_RS']
        A_IR = params_values_dict['A_IR']
        # A_CN amplitude is set by looking at
        # SZ_and_fg_models-high_ell.txt at high multipole (ell = 2742),
        # where the correlated noise largely dominates over the other
        # components (see Bolliet et al. 1712.00788).
        A_CN = 0.9033
        #print(state)

        state["Cl_sz_foreground"] = A_CIB*self.A_CIB_MODEL + A_RS*self.A_RS_MODEL + A_IR*self.A_IR_MODEL + A_CN*self.A_CN_MODEL

    def get_Cl_sz_foreground(self):
        return self._current_state['Cl_sz_foreground']

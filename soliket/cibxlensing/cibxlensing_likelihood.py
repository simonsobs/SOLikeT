"""
.. module:: cib x kappa cross spectrum likelihood (in progress)

"""

# Acronyms:
# ps: power spectrum
# fg: foregrounds
# cib: cosmic infrared background
# rs: radio sources
# ir: infrared sources
# f_sky: sky fraction


from turtle import position
from cobaya.theory import Theory
# from cobaya.conventions import _packages_path
_packages_path = 'packages_path'
# from cobaya.likelihoods._base_classes import _InstallableLikelihood
from soliket.gaussian import GaussianLikelihood
import numpy as np
import os
from scipy.ndimage.interpolation import shift
from typing import Optional, Sequence
from pkg_resources import resource_filename
from myfuncs import alm as yalm, io as yio
from mpi4py import MPI



class CIBxKAPPA_Likelihood(GaussianLikelihood):
    cib_spectra_directory: Optional[str] = '/project/r/rbond/ymehta3/output/param-fitting/'
    cib_spectra_file: Optional[str] = 'toy_data.npy'
    cib_cov_directory: Optional[str] = '/project/r/rbond/ymehta3/output/param-fitting/'
    # cib_cov_file: Optional[str] = 'covCl_avg_v3.npy'
    cib_cov_file: Optional[str] = 'toy_cov.npy'
    project_directory: Optional[str] = '/project/r/rbond/ymehta3/' 
    cross_mcms_names_file: Optional[str] = 'input_data/names_mcms_dr6_cross.txt'
    cov_ell_info_file: Optional[str] = 'input_data/bandpower_ell_info.txt'

    def initialize(self):
        self.data_directory = self.cib_spectra_directory
        self.data_file = self.cib_spectra_file
        self.cov_directory = self.cib_cov_directory
        self.cov_file = self.cib_cov_file

        #Load the Data
        Dpoints = np.load(self.data_directory + self.data_file)
        self.cov = np.load(os.path.join(self.cov_directory, self.cov_file))
        self.mcms_paths_list = yio.readTxt( os.path.join(self.project_directory, self.cross_mcms_names_file) )

        #Get Ell Info
        self.lmax, self.binsize = np.loadtxt(os.path.join(self.project_directory, self.cov_ell_info_file))

        #Extract the Data
        self.datavector = Dpoints[:,1]
        self.ellsvector = Dpoints[:,0]  # for mock data, this is binned class_sz theory ells
        # np.save('/scratch/r/rbond/ymehta3/'+'actual_mock_data.npy', np.stack([self.ellsvector, self.datavector, np.sqrt(np.diag(self.cov))], axis=-1) )

        self.debug_index = 1
        comm = MPI.COMM_WORLD
        self.rank = comm.Get_rank()
        self.debug_theoryvector = []
        self.debug_unbinned_theory = []

        super().initialize()


    def get_requirements(self):
        return {"cl_cib_kappa": {}}

    def _get_data(self):
        ell = self.ellsvector
        Cl = self.datavector
        return ell, Cl

    def _get_cov(self):
        return self.cov


    def _get_theory(self, **params_values):
        # theory = self.theory.get_Cl_kappa_cib()
        theory = self.theory.get_cl_cib_kappa()
        theoryvector_unbinned = []

        #Extract All Cross Spectra
        freq_list = np.sort(np.array(list(theory.keys())).astype(int))
        freq_list = [str(nu) for nu in freq_list]
        for nu in freq_list:
            Dl_1h_theory = np.array(theory[str(nu)]['1h'])
            Dl_2h_theory = np.array(theory[str(nu)]['2h'])
            ells_theory = np.array(theory[str(nu)]['ell'])

            fac = ells_theory * (ells_theory+1) / 2/np.pi
            Dl_tot_theory = Dl_1h_theory + Dl_2h_theory
            Cl_kappa_cib = Dl_tot_theory / fac
            theoryvector_unbinned.append(Cl_kappa_cib)

        #Bin Theory
        theoryvector_binned = []
        for iCl, Cls_unbinned in enumerate(theoryvector_unbinned):
            #Interpolate the Theory Vector 
            full_ells = np.arange(self.lmax+1)
            interp_Cls_unbinned = np.interp(full_ells, ells_theory, Cls_unbinned)

            #Get Mode-Coupling Matrix
            mcm_path = ''
            for possible_mcm_path in self.mcms_paths_list:
                if freq_list[iCl] not in possible_mcm_path:
                    continue
                else:
                    mcm_path = possible_mcm_path

            #Confirm Mode-Coupling Matrix 
            if not mcm_path:
                raise ValueError(f"I need the mcm for {freq_list[iCl]} GHz but it's not in the provided path names, which are '{self.mcms_paths_list}'.")

            #Bin Theory Vector
            # for mock data, the data ells are the binned theory ells, so there's already perfect syncronization with the theory Cl's
            theoryvector_binned.append( yalm.binTheory(interp_Cls_unbinned, mcm_path) )

        #Create large, Multifreq Theory Vector
        theoryvector = np.array(theoryvector_binned).flatten()

        #Debugging
        self.debug_theoryvector.append(theoryvector)
        if not self.debug_index % 100:
            debugfname = self.data_directory + 'mcmcresults/steps/' + f'steps_Cls_delta_{self.rank}_{self.debug_index}.npy'
            np.save(debugfname, np.array(self.debug_theoryvector))

        # # if not os.path.isfile(debugfname):
        # #     np.savetxt(debugfname, [theoryvector])
        # # else:
        # #     # import pdb; pdb.set_trace()
        # #     all_theoryCls = np.loadtxt(debugfname)
        # #     if all_theoryCls.ndim == 1:
        # #         np.savetxt(debugfname, np.stack( (all_theoryCls, theoryvector) ))
        # #     elif all_theoryCls.ndim == 2: 
        # #         np.savetxt(debugfname, np.concatenate( (all_theoryCls, theoryvector[None, :]) ))
        # #     else:
        # #         raise ValueError('something has gone horribly wrong with saving the previous spectra!')
        self.debug_index += 1

        return theoryvector

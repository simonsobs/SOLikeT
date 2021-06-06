"""
.. module:: sz cluster counts (in progress)

"""


from cobaya.theory import Theory
from cobaya.conventions import _packages_path
from cobaya.likelihood import Likelihood
# from cobaya.likelihoods._base_classes import _InstallableLikelihood
import numpy as np
import os
from scipy.ndimage.interpolation import shift
from typing import Optional, Sequence
from pkg_resources import resource_filename
from astropy.io import fits

path_to_catalogue = '/Users/boris/Work/CLASS-SZ/SO-SZ/class_sz_external_data_and_scripts/MFMF_SOSim_3freq_tiles/'

#"fits_image_filename = fits.util.get_testdata_filepath(path_to_catalogue+'MFMF_SOSim_3freq_tiles_M500.fits')
tcat = path_to_catalogue+'MFMF_SOSim_3freq_tiles_M500.fits'
list = fits.open(tcat)
data = list[1].data
z = data.field("redshift")
snr = data.field("SNR")



class binned_cc_likelihood(Likelihood):
    data_directory: Optional[str] = resource_filename(
        "soliket", "sz_binned_cluster_counts/data/"
    )
    # ylims_file: Optional[str] = 'so_3freqs_020621_ylims.txt'
    # skyfracs_file: Optional[str] = 'so_3freqs_020621_skyfracs.txt'
    # thetas_file: Optional[str] = 'so_3freqs_020621_thetas.txt'
    tcat_file: Optional[str] = 'MFMF_SOSim_3freq_tiles_M500.fits'
    snrcut: Optional[str] = 5.



    def initialize(self):

        tcat = os.path.join(self.data_directory, self.tcat_file)
        list = fits.open(tcat)
        z = data.field("redshift")
        snr = data.field("SNR")
        self.z = z[snr > self.snrcut]
        self.snr = snr[snr > self.snrcut]
        super().initialize()


    def get_requirements(self):
        return {"sz_binned_cluster_counts": {}}

    def _get_data(self):
        snr = self.snr
        z = self.z
        return snr, z



    def _get_theory(self, **params_values):
        theory = self.theory.get_sz_binned_cluster_counts()
        # print(theory)
        dNdzdy_theoretical = theory['dndzdy']
        # z_center = M.dndzdy_theoretical()['z_center']
        z_edges = theory['z_edges']
        # log10y_center = M.dndzdy_theoretical()['log10y_center']
        log10y_edges = theory['log10y_edges']
        return dNdzdy_theoretical,z_edges,log10y_edges

    def logp(self, **params_values):
        theory = self._get_theory(**params_values)
        dNdzdy_theoretical,z_edges,log10y_edges = theory
        dNdzdy_catalog, zedges, yedges = np.histogram2d(z,np.log10(snr), bins=[z_edges,log10y_edges])
        SZCC_Cash = 0.
        N_z,N_y = np.shape(dNdzdy_theoretical)
        for index_z in range(N_z):
            for index_y in range(N_y):
                if not dNdzdy_theoretical[index_z][index_y] == 0.:
                    ln_factorial = 0.
                    if not dNdzdy_catalog[index_z,index_y] == 0.:
                        if dNdzdy_catalog[index_z,index_y] > 10.:
                            # Stirling approximation only for more than 10 elements
                            ln_factorial = 0.918939 + (dNdzdy_catalog[index_z,index_y] + 0.5) * np.log(dNdzdy_catalog[index_z,index_y]) - dNdzdy_catalog[index_z,index_y]
                        else:
                            # Direct computation of factorial
                            ln_factorial = np.log(np.math.factorial(int(dNdzdy_catalog[index_z,index_y])))
                    SZCC_Cash += (dNdzdy_catalog[index_z,index_y] * np.log(dNdzdy_theoretical[index_z][index_y]) - dNdzdy_theoretical[index_z][index_y] - ln_factorial)

        # return ln(L)
        loglkl = SZCC_Cash
        return loglkl

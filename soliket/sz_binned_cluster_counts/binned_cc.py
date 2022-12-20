"""
.. module:: sz cluster counts (in progress)

"""


from cobaya.theory import Theory
# from cobaya.conventions import _packages_path
_packages_path = 'packages_path'
from cobaya.likelihood import Likelihood
# from cobaya.likelihoods._base_classes import _InstallableLikelihood
import numpy as np
import os
from scipy.ndimage.interpolation import shift
from typing import Optional, Sequence
from pkg_resources import resource_filename
from astropy.io import fits

# path_to_catalogue = '/Users/boris/Work/CLASS-SZ/SO-SZ/class_sz_external_data_and_scripts/MFMF_SOSim_3freq_tiles/'
#
# #"fits_image_filename = fits.util.get_testdata_filepath(path_to_catalogue+'MFMF_SOSim_3freq_tiles_M500.fits')
# tcat = path_to_catalogue+'MFMF_SOSim_3freq_tiles_M500.fits'
# list = fits.open(tcat)
# data = list[1].data
# z = data.field("redshift")
# snr = data.field("SNR")



class binned_cc_likelihood(Likelihood):
    data_directory: Optional[str] = resource_filename(
        "soliket", "sz_binned_cluster_counts/data/"
    )
    # ylims_file: Optional[str] = 'so_3freqs_020621_ylims.txt'
    # skyfracs_file: Optional[str] = 'so_3freqs_020621_skyfracs.txt'
    # thetas_file: Optional[str] = 'so_3freqs_020621_thetas.txt'
    tcat_file: Optional[str] = 'MFMF_SOSim_3freq_tiles_M500.fits'
    snrcut: Optional[str] = 5.
    experiment: Optional[str] = 'Planck'
    debug: Optional[str] = True


    bin_z_min_cluster_counts: Optional[str] = 0.
    bin_z_max_cluster_counts: Optional[str] = 0.
    bin_dz_cluster_counts: Optional[str] = 0.
    bin_dlog10_snr: Optional[str] = 0.



    def initialize(self):
        if self.experiment == 'SO':
            tcat = os.path.join(self.data_directory, self.tcat_file)
            list = fits.open(tcat)
            data = list[1].data
            z = data.field("redshift")
            snr = data.field("SNR")
            self.z = z[snr > self.snrcut]
            self.snr = snr[snr > self.snrcut]
        elif self.experiment == 'Planck':
            SZ_catalog = np.loadtxt(os.path.join(self.data_directory, self.tcat_file))
            if self.debug:
                print("starting setting up catalogue data")
            # print(self.theory.extra_args['M_min'])
                print(self.bin_z_min_cluster_counts)

            # exit(0)

            z_0 = self.bin_z_min_cluster_counts;
            z_max = self.bin_z_max_cluster_counts;
            dz = self.bin_dz_cluster_counts;
            Nbins_z = int((z_max - z_0)/dz)
            z_center = np.zeros(Nbins_z)


            # class_alloc(pcsz->z_center,pcsz->Nbins_z*sizeof(double),pcsz->error_message);
            # int index_z;
            for index_z in range(Nbins_z):
                z_center[index_z] = z_0 + 0.5*dz + index_z*dz;
                if self.debug:
                    print("z_center:",z_center[index_z])
                # print("index_z=%d, z_center=%e\n"%(index_z,z_center[index_z]))

            logy_min =  0.7 # set for planck
            logy_max = 1.5 # set for planck
            dlogy = self.bin_dlog10_snr
            #
            Nbins_y = int((logy_max - logy_min)/dlogy)+1
            if self.debug:
                print('Nbins_y:',Nbins_y,self.bin_dlog10_snr)
            logy = np.zeros(Nbins_y)
            y_i = logy_min + dlogy/2.
            for index_y in range(Nbins_y):
                logy[index_y] = y_i
                y_i += dlogy
                if self.debug:
                    print("y_center:",index_y,Nbins_y,10**logy[index_y])

            self.dNdzdy_catalog = np.zeros([Nbins_z,Nbins_y])

            nrows = len(SZ_catalog[:,0])
            if self.debug:
                print('number of rows in catalogue:',nrows)


            zmin = z_0
            zmax = zmin +dz
            # Count the number of clusters in each (z,y) bin
            for i in range (Nbins_z):
                for j in range (Nbins_y):
                    y_min = logy[j]-dlogy/2.
                    y_max = logy[j]+dlogy/2.
                    y_min = 10**y_min
                    y_max = 10**y_max
                    # if j == Nbins_y:
                    #     y_max = 1e100

                    for ii in range(0,nrows):
                        if (SZ_catalog[ii][0]>=zmin) and (SZ_catalog[ii][0]<zmax)\
                           and (SZ_catalog[ii][2]<y_max) and (SZ_catalog[ii][2]>=y_min):
                            self.dNdzdy_catalog[i][j] += 1.
                # Count the number of clusters in the last y bin for each z bin
                # j = Nbins_y
                # y_min =y_max
                # for ii in range(0,nrows):
                #     if (SZ_catalog[ii][0]>=zmin) and (SZ_catalog[ii][0]<zmax)\
                #        and (SZ_catalog[ii][2]>=y_min):
                #         self.dNdzdy_catalog[i][j] += 1.
                # Change edges of the redshift bin
                zmin += dz
                zmax += dz

            # Count the number of clusters in each y bin with missing redshifts and apply normalization
            for j in range(0,Nbins_y):
                y_min = logy[j]-dlogy/2.
                y_max = logy[j]+dlogy/2.
                y_min = 10**y_min
                y_max = 10**y_max
                for ii in range(0,nrows):
                    if (SZ_catalog[ii][0]==-1) \
                       and (SZ_catalog[ii][2]<y_max) and (SZ_catalog[ii][2]>=y_min):
                        norm = 0
                        for jj in range(0,Nbins_z):
                            norm += self.dNdzdy_catalog[jj][j]
                        self.dNdzdy_catalog[:,j] *= (1.+norm)/norm

            # # Count the number of clusters in the last y bin with missing redshifts and apply normalization
            # j = Nbins_y
            # y_min =y_max
            # for ii in range(0,nrows):
            #     if (SZ_catalog[ii][0]==-1) and (SZ_catalog[ii][2]>=y_min):
            #         norm = 0.
            #         for jj in range(0,Nbins_z):
            #             norm += self.dNdzdy_catalog[jj][j]
            #         self.dNdzdy_catalog[:,j] *= (1.+norm)/norm
            if self.debug:
                print("Ntot cat:",np.sum(self.dNdzdy_catalog[:,:]))
            # exit(0)


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
        if self.experiment == 'SO':
            dNdzdy_catalog, zedges, yedges = np.histogram2d(self.z,np.log10(self.snr), bins=[z_edges,log10y_edges])
        elif self.experiment == 'Planck':
            dNdzdy_catalog = self.dNdzdy_catalog

        SZCC_Cash = 0.
        N_z,N_y = np.shape(dNdzdy_theoretical)
        if self.debug == True:
            print('N_z,N_y',N_z,N_y)
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
                    if self.debug == True:
                        print("theory: %.5e, catalogue: %.5e"%(dNdzdy_theoretical[index_z][index_y],dNdzdy_catalog[index_z,index_y]))
        if self.debug == True:
            print("Ntot cat:",np.sum(dNdzdy_catalog[:,:]))
            print("Ntot theory:",np.sum(dNdzdy_theoretical[:][:]))
        # return ln(L)
        loglkl = SZCC_Cash
        return loglkl




class unbinned_cc_likelihood(Likelihood):
    data_directory: Optional[str] = resource_filename(
        "soliket", "sz_binned_cluster_counts/data/"
    )
    # ylims_file: Optional[str] = 'so_3freqs_020621_ylims.txt'
    # skyfracs_file: Optional[str] = 'so_3freqs_020621_skyfracs.txt'
    # thetas_file: Optional[str] = 'so_3freqs_020621_thetas.txt'
    tcat_file: Optional[str] = 'MFMF_SOSim_3freq_tiles_M500.fits'
    snrcut: Optional[str] = 5.
    experiment: Optional[str] = 'Planck'
    debug: Optional[str] = True


    bin_z_min_cluster_counts: Optional[str] = 0.
    bin_z_max_cluster_counts: Optional[str] = 0.
    bin_dz_cluster_counts: Optional[str] = 0.
    bin_dlog10_snr: Optional[str] = 0.



    def initialize(self):
        if self.experiment == 'SO':
            tcat = os.path.join(self.data_directory, self.tcat_file)
            list = fits.open(tcat)
            data = list[1].data
            z = data.field("redshift")
            snr = data.field("SNR")
            self.z = z[snr > self.snrcut]
            self.snr = snr[snr > self.snrcut]
        elif self.experiment == 'Planck':
            SZ_catalog = np.loadtxt(os.path.join(self.data_directory, self.tcat_file))
            if self.debug:
                print("starting setting up catalogue data")
            # print(self.theory.extra_args['M_min'])
                print(self.bin_z_min_cluster_counts)

            # exit(0)

            z_0 = self.bin_z_min_cluster_counts;
            z_max = self.bin_z_max_cluster_counts;
            dz = self.bin_dz_cluster_counts;
            Nbins_z = int((z_max - z_0)/dz)
            z_center = np.zeros(Nbins_z)


            # class_alloc(pcsz->z_center,pcsz->Nbins_z*sizeof(double),pcsz->error_message);
            # int index_z;
            for index_z in range(Nbins_z):
                z_center[index_z] = z_0 + 0.5*dz + index_z*dz;
                if self.debug:
                    print("z_center:",z_center[index_z])
                # print("index_z=%d, z_center=%e\n"%(index_z,z_center[index_z]))

            logy_min =  0.7 # set for planck
            logy_max = 1.5 # set for planck
            dlogy = self.bin_dlog10_snr
            #
            Nbins_y = int((logy_max - logy_min)/dlogy)+1
            if self.debug:
                print('Nbins_y:',Nbins_y,self.bin_dlog10_snr)
            logy = np.zeros(Nbins_y)
            y_i = logy_min + dlogy/2.
            for index_y in range(Nbins_y):
                logy[index_y] = y_i
                y_i += dlogy
                if self.debug:
                    print("y_center:",index_y,Nbins_y,10**logy[index_y])

            self.dNdzdy_catalog = np.zeros([Nbins_z,Nbins_y])

            nrows = len(SZ_catalog[:,0])
            if self.debug:
                print('number of rows in catalogue:',nrows)


            zmin = z_0
            zmax = zmin +dz
            # Count the number of clusters in each (z,y) bin
            for i in range (Nbins_z):
                for j in range (Nbins_y):
                    y_min = logy[j]-dlogy/2.
                    y_max = logy[j]+dlogy/2.
                    y_min = 10**y_min
                    y_max = 10**y_max
                    # if j == Nbins_y:
                    #     y_max = 1e100

                    for ii in range(0,nrows):
                        if (SZ_catalog[ii][0]>=zmin) and (SZ_catalog[ii][0]<zmax)\
                           and (SZ_catalog[ii][2]<y_max) and (SZ_catalog[ii][2]>=y_min):
                            self.dNdzdy_catalog[i][j] += 1.
                # Count the number of clusters in the last y bin for each z bin
                # j = Nbins_y
                # y_min =y_max
                # for ii in range(0,nrows):
                #     if (SZ_catalog[ii][0]>=zmin) and (SZ_catalog[ii][0]<zmax)\
                #        and (SZ_catalog[ii][2]>=y_min):
                #         self.dNdzdy_catalog[i][j] += 1.
                # Change edges of the redshift bin
                zmin += dz
                zmax += dz

            # Count the number of clusters in each y bin with missing redshifts and apply normalization
            for j in range(0,Nbins_y):
                y_min = logy[j]-dlogy/2.
                y_max = logy[j]+dlogy/2.
                y_min = 10**y_min
                y_max = 10**y_max
                for ii in range(0,nrows):
                    if (SZ_catalog[ii][0]==-1) \
                       and (SZ_catalog[ii][2]<y_max) and (SZ_catalog[ii][2]>=y_min):
                        norm = 0
                        for jj in range(0,Nbins_z):
                            norm += self.dNdzdy_catalog[jj][j]
                        self.dNdzdy_catalog[:,j] *= (1.+norm)/norm

            # # Count the number of clusters in the last y bin with missing redshifts and apply normalization
            # j = Nbins_y
            # y_min =y_max
            # for ii in range(0,nrows):
            #     if (SZ_catalog[ii][0]==-1) and (SZ_catalog[ii][2]>=y_min):
            #         norm = 0.
            #         for jj in range(0,Nbins_z):
            #             norm += self.dNdzdy_catalog[jj][j]
            #         self.dNdzdy_catalog[:,j] *= (1.+norm)/norm
            if self.debug:
                print("Ntot cat:",np.sum(self.dNdzdy_catalog[:,:]))
            # exit(0)


        super().initialize()


    def get_requirements(self):
        return {"sz_unbinned_cluster_counts": {}}

    def _get_data(self):
        snr = self.snr
        z = self.z
        return snr, z



    def _get_theory(self, **params_values):
        theory = self.theory.get_sz_unbinned_cluster_counts()
        # # print(theory)
        # dNdzdy_theoretical = theory['dndzdy']
        # # z_center = M.dndzdy_theoretical()['z_center']
        # z_edges = theory['z_edges']
        # # log10y_center = M.dndzdy_theoretical()['log10y_center']
        # log10y_edges = theory['log10y_edges']
        # print(theory)
        return theory

    def logp(self, **params_values):
        theory = self._get_theory(**params_values)
        # dNdzdy_theoretical,z_edges,log10y_edges = theory
        # if self.experiment == 'SO':
        #     dNdzdy_catalog, zedges, yedges = np.histogram2d(self.z,np.log10(self.snr), bins=[z_edges,log10y_edges])
        # elif self.experiment == 'Planck':
        #     dNdzdy_catalog = self.dNdzdy_catalog
        #
        # SZCC_Cash = 0.
        # N_z,N_y = np.shape(dNdzdy_theoretical)
        # if self.debug == True:
        #     print('N_z,N_y',N_z,N_y)
        # for index_z in range(N_z):
        #     for index_y in range(N_y):
        #         if not dNdzdy_theoretical[index_z][index_y] == 0.:
        #             ln_factorial = 0.
        #             if not dNdzdy_catalog[index_z,index_y] == 0.:
        #                 if dNdzdy_catalog[index_z,index_y] > 10.:
        #                     # Stirling approximation only for more than 10 elements
        #                     ln_factorial = 0.918939 + (dNdzdy_catalog[index_z,index_y] + 0.5) * np.log(dNdzdy_catalog[index_z,index_y]) - dNdzdy_catalog[index_z,index_y]
        #                 else:
        #                     # Direct computation of factorial
        #                     ln_factorial = np.log(np.math.factorial(int(dNdzdy_catalog[index_z,index_y])))
        #             SZCC_Cash += (dNdzdy_catalog[index_z,index_y] * np.log(dNdzdy_theoretical[index_z][index_y]) - dNdzdy_theoretical[index_z][index_y] - ln_factorial)
        #             if self.debug == True:
        #                 print("theory: %.5e, catalogue: %.5e"%(dNdzdy_theoretical[index_z][index_y],dNdzdy_catalog[index_z,index_y]))
        # if self.debug == True:
        #     print("Ntot cat:",np.sum(dNdzdy_catalog[:,:]))
        #     print("Ntot theory:",np.sum(dNdzdy_theoretical[:][:]))
        # return ln(L)
        loglkl = theory
        return loglkl

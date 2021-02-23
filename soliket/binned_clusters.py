from soliket.binned_poisson import BinnedPoissonLikelihood
from scipy import interpolate, integrate, special
from typing import Optional
import numpy as np
import math as m
import time as t
import os
#from astropy.io import fits # will be used eventually!
import sys
import multiprocessing
from functools import partial

pi = 3.1415926535897932384630
rhocrit0 = 2.7751973751261264e11 # [h2 msun Mpc-3]
c_ms = 3e8 # [m s-1]
msun = 1.98892e30 # [kg]
m_pivot = 3.e14*msun
Mpc = 3.08568025e22 # [m]
G = 6.67300e-11 # [m3 kg-1 s-2]


class BinnedClusterLikelihood(BinnedPoissonLikelihood):

    name = "BinnedCluster"
    data_path: Optional[str] = None
    test_cat_file: Optional[str] = None
    test_Q_file: Optional[str] = None
    test_rms_file: Optional[str] = None
    cat_file: Optional[str] = None
    mock_file : Optional[str] = None
    mock2D_file : Optional[str] = None
    Q_file: Optional[str] = None
    meanQ_file: Optional[str] = None
    theta_file: Optional[str] = None
    rms_file: Optional[str] = None
    mock_test: Optional[str] = None
    choose_dim: Optional[str] = None
    single_tile_test: Optional[str] = None
    Q_optimise: Optional[str] = None

    def initialize(self):

        print('\r :::::: this is initialisation in binned_clusters.py')
        print('\r :::::: reading catalogue')

        self.qcut = 5.
        self.scatter = 0.2
        print("\r intrinsic scatter = ", self.scatter)

        # redshift bin for N(z)
        zarr = np.linspace(0, 2, 21)
        if zarr[0] == 0 :zarr[0] = 1e-5
        self.zarr = zarr
        print(" Nz for this analysis = ", len(zarr) - 1)

        # redshift bin for P(z,k)
        zpk = np.linspace(0, 2, 141)
        if zpk[0] == 0. : zpk[0] = 1e-5
        self.zpk = zpk
        print(" Nz for matter power spectrum = ", len(zpk))

        self.k = np.logspace(-4, np.log10(5), 200, endpoint=False)
        #self.k = np.logspace(-4, np.log10(5), 50, endpoint=False) # for comparison with f90 output
        #print(" k input check in initialisation", self.k.min(), self.k.max())

        # mass bin in natural log scale
        self.lnmmin = np.log(1e13)
        self.lnmmax = np.log(1e16)
        self.dlnm = 0.05
        self.marr = np.arange(self.lnmmin+(self.dlnm/2.), self.lnmmax, self.dlnm)

        single_tile = self.single_tile_test
        Q_opt = self.Q_optimise
        dimension = self.choose_dim

        self.data_directory = self.data_path

        if single_tile == 'yes':
            self.datafile = self.test_cat_file
            print(" SO test only for a single tile")
        else:
            self.datafile = self.cat_file
            print(" SO for a full map")

        if dimension == '2D':
            print(" 2D likelihood as a function of redshift and signal-to-noise")
        else:
            print(" 1D likelihood as a function of redshift")

        cat = np.loadtxt(os.path.join(self.data_directory, self.datafile))
        zcat = cat[:,0]
        qcat = cat[:,2]

        Ncat = len(zcat)
        print('\r Number of clusters in catalogue = ', Ncat)
        print('\r SNR cut = ', self.qcut)

        znew = []
        snrnew = []
        i = 0
        for i in range(Ncat):
            if qcat[i] > self.qcut:
                znew.append(zcat[i])
                snrnew.append(qcat[i])

        z = np.array(znew)
        snr = np.array(snrnew)

        Ncat = len(z)
        print('\r Number of clusters above the SNR cut = ', Ncat)

       # 1D catalogue
        zmin = zarr[0]
        dz = zarr[1] - zarr[0]
        zmax = zmin + dz
        delNcat = np.zeros(len(zarr))

        i = 0
        j = 0
        for i in range(len(zarr)):
            for j in range(Ncat):
                if z[j] >= zmin and z[j] < zmax :
                    delNcat[i] += 1.
            zmin = zmin + dz
            zmax = zmax + dz

        print("\r Catalogue N")
        print("", delNcat)

        if self.mock_test == 'yes' and dimension == "1D" :
            self.datafile_mock = self.mock_file
            delNmock = np.loadtxt(os.path.join(self.data_directory, self.datafile_mock))
            self.delNcat = zarr, delNmock
            print("\r Using mock catalogue")
            print("\r Mock catalogue N")
            print(delNmock)
        else:
            self.delNcat = zarr, delNcat


        # 2D catalogue
        logqmin = 0.6  # log10(4)  = 0.602 --- min snr = 4.11
        logqmax = 1.6  # log10(35) = 1.544 --- max snr = 34.73
        dlogq = 0.25

        Nq = int((logqmax - logqmin)/dlogq) + 1  ########

        if dimension == "2D":
            print("\r Number of SNR bins = ", Nq+1)

        qi = logqmin + dlogq/2.
        qarr = np.zeros(Nq+1)

        i = 0
        for i in range(Nq+1):
            qarr[i] = qi
            qi = qi + dlogq

        #if self.choose_dim == "2D":
        #    print("\r Center of SNR bins = ", 10**qarr)

        zmin = zarr[0]
        zmax = zmin + dz

        delN2Dcat = np.zeros((len(zarr), Nq+1))

        i = 0
        j = 0
        k = 0
        for i in range(len(zarr)):
           for j in range(Nq):
                qmin = qarr[j] - dlogq/2.
                qmax = qarr[j] + dlogq/2.
                qmin = 10.**qmin
                qmax = 10.**qmax

                for k in range(Ncat):
                    if z[k] >= zmin and z[k] < zmax and snr[k] >= qmin and snr[k] < qmax :
                        delN2Dcat[i,j] += 1

           j = Nq + 1 # the last bin contains all S/N greater than what in the previous bin
           qmin = qmax

           for k in range(Ncat):
               if z[k] >= zmin and z[k] < zmax and snr[k] >= qmin :
                   delN2Dcat[i,j] += 1

           zmin = zmin + dz
           zmax = zmax + dz

        if dimension == "2D":
            print("\r Catalogue N")
            j = 0
            for j in range(Nq+1):
                    print("", j, delN2Dcat[:,j].sum())

        self.Nq = Nq
        self.qarr = qarr
        self.dlogq = dlogq

        if self.mock_test == 'yes' and dimension == "2D" :
            self.datafile_mock = self.mock2D_file
            delN2Dmock = np.loadtxt(os.path.join(self.data_directory, self.datafile_mock))
            self.delN2Dcat = zarr, qarr, delN2Dmock
            print("\r Using mock catalogue N")
            print("\r Mock catalogue N (2D)")
            j = 0
            for j in range(Nq+1):
                    print(j, delN2Dmock[:,j].sum())
        else:
            self.delN2Dcat = zarr, qarr, delN2Dcat

        print('\r :::::: loading files describing selection function')
        print('\r :::::: reading theta and Q')
        if single_tile =='yes' or Q_opt == 'yes':
            self.datafile_Q = self.test_Q_file if single_tile == 'yes' else self.meanQ_file
            file_Q = np.loadtxt(os.path.join(self.data_directory, self.datafile_Q))
            self.tt500 = file_Q[:,0]
            self.Q = file_Q[:,1]
            print("\r Number of thetas = ", len(self.tt500))
            if single_tile == 'yes':
                print("\r Number of Q function = ", len(file_Q[0])-1)
            else:
                print("\r Number of Q function = ", len(file_Q[0])-1)
                print("\r Using one averaged Q function for optimisation")

        else:
            self.datafile_theta = self.theta_file
            self.tt500 = np.loadtxt(os.path.join(self.data_directory, self.datafile_theta))
            self.datafile_Q = self.Q_file
            file_Q = np.loadtxt(os.path.join(self.data_directory, self.datafile_Q))
            NQ = len(file_Q[0])
            print("\r Number of thetas = ", len(self.tt500))
            print("\r Number of Q functions = ", NQ)
            i = 0
            self.Q = np.zeros((len(self.tt500), NQ))
            for i in range(NQ):
                self.Q[:,i] = file_Q[:,i]

        print('\r :::::: reading noise data')
        if single_tile == 'yes':
            self.datafile_rms = self.test_rms_file
            file_rms = np.loadtxt(os.path.join(self.data_directory, self.datafile_rms))
            self.skyfracs = file_rms[:,0]*3.046174198e-4
            self.noise = file_rms[:,1]
        else:
            self.datafile_rms = self.rms_file
            file_rms = np.loadtxt(os.path.join(self.data_directory, self.datafile_rms))
            self.skyfracs = file_rms[:,0]*3.046174198e-4
            self.noise = file_rms[:,1]
            self.tilenames = file_rms[:,2]
            print("\r Number of tiles = ", len(np.unique(self.tilenames)))

        print("\r Number of sky patches = ", self.skyfracs.size)
        print("\r Entire survey area = ", self.skyfracs.sum()/3.046174198e-4, "deg2")

        super().initialize()

    def get_requirements(self):
        return {"Hubble":  {"z": self.zarr},
                "angular_diameter_distance": {"z": self.zarr},
                "Pk_interpolator": {"z": self.zarr,
                                    "k_max": 4.0,
                                    "nonlinear": False,
                                    "hubble_units": False,
                                    "k_hunit": False,
                                    "vars_pairs": [["delta_nonu", "delta_nonu"]]},
                "ombh2":None, "H0":None, "omch2":None, "omegam":None, "sigma8":None}
                # why are they not just in param_vals?

    def _get_data(self):
        return self.delNcat, self.delN2Dcat

    def _get_om(self):
        return (self.theory.get_param("omch2") + self.theory.get_param("ombh2"))/((self.theory.get_param("H0")/100.0)**2)

    def _get_Ez(self):
        return self.theory.get_Hubble(self.zarr)/self.theory.get_param("H0")

    def _get_DAz(self):
        return self.theory.get_angular_diameter_distance(self.zarr)

    def _get_dndlnm(self, pk_intp, **kwargs):
        print("get dndlnm kwargs = ", kwargs) # still not printing out derived params! -_-

        h = self.theory.get_param("H0")/100.0
        Ez = self._get_Ez()
        om = self._get_om()
        rhom0 = rhocrit0*om

        zarr = self.zarr
        zpk = self.zpk

        #k = self.k # why does this change?
        k = np.logspace(-4, np.log10(5), 200, endpoint=False)
        #print("k value check ", k.min(), k.max())

        # Pk_interpolator = self.theory.get_Pk_interpolator(("delta_nonu", "delta_nonu"), nonlinear=False).P
        # pks0 = Pk_interpolator(zpk, k)
        pks0 = pk_intp.P(zpk, k)
        #print("peak P(k) value = ", pks0.max(), pks0.min())

        def pks_zbins(newz):
            i = 0
            newpks = np.zeros((len(newz),len(k)))
            for i in range(k.size):
                tck = interpolate.splrep(zpk, pks0[:,i])
                newpks[:,i] = interpolate.splev(newz, tck)
            return newpks
        pks = pks_zbins(zarr)
        #print("peak P(k) value = ", pks.max(), pks.min())
        #print(pks.shape)

        pks *= h**3.
        k /= h
        #np.savetxt("pk_re_check.txt", pks)

        def radius(M):
            return (0.75*M/pi/rhom0)**(1./3.)

        def win(x):
            return 3.*(np.sin(x) - x*np.cos(x))/(x**3.)

        def win_prime(x):
            return 3.*np.sin(x)/(x**2.) - 9.*(np.sin(x) - x*np.cos(x))/(x**4.)

        marr = self.marr
        R = radius(np.exp(marr))[:,None]

        def sigma_sq(R, k):
            integral = np.zeros((len(k), len(marr), len(zarr)))
            i = 0
            # can this be faster as well? but this actually doesn't take so long
            for i in range(k.size):
                integral[i,:,:] = np.array((k[i]**2.)*pks[:,i]*(win(k[i]*R)**2.))
            return integrate.simps(integral, k, axis=0)/(2.*pi**2.)

        def sigma_sq_prime(R, k):
            integral = np.zeros((len(k), len(marr), len(zarr)))
            i = 0
            for i in range(k.size):
                integral[i,:,:] = np.array((k[i]**2.)*pks[:,i]*2.*k[i]*win(k[i]*R)*win_prime(k[i]*R))
            return integrate.simps(integral, k, axis=0)/(2.*pi**2.)

        def tinker(sgm, zarr):

            total = 9

            delta  = np.zeros(total)
            par_aa = np.zeros(total)
            par_a  = np.zeros(total)
            par_b  = np.zeros(total)
            par_c  = np.zeros(total)

            delta[0] = 200
            delta[1] = 300
            delta[2] = 400
            delta[3] = 600
            delta[4] = 800
            delta[5] = 1200
            delta[6] = 1600
            delta[7] = 2400
            delta[8] = 3200

            par_aa[0] = 0.186
            par_aa[1] = 0.200
            par_aa[2] = 0.212
            par_aa[3] = 0.218
            par_aa[4] = 0.248
            par_aa[5] = 0.255
            par_aa[6] = 0.260
            par_aa[7] = 0.260
            par_aa[8] = 0.260

            par_a[0] = 1.47
            par_a[1] = 1.52
            par_a[2] = 1.56
            par_a[3] = 1.61
            par_a[4] = 1.87
            par_a[5] = 2.13
            par_a[6] = 2.30
            par_a[7] = 2.53
            par_a[8] = 2.66

            par_b[0] = 2.57
            par_b[1] = 2.25
            par_b[2] = 2.05
            par_b[3] = 1.87
            par_b[4] = 1.59
            par_b[5] = 1.51
            par_b[6] = 1.46
            par_b[7] = 1.44
            par_b[8] = 1.41

            par_c[0] = 1.19
            par_c[1] = 1.27
            par_c[2] = 1.34
            par_c[3] = 1.45
            par_c[4] = 1.58
            par_c[5] = 1.80
            par_c[6] = 1.97
            par_c[7] = 2.24
            par_c[8] = 2.44

            delta = np.log10(delta)

            dso = 500.
            omz = om*((1. + zarr)**3.)/(Ez**2.)
            dsoz = dso/omz

            tck1 = interpolate.splrep(delta, par_aa)
            tck2 = interpolate.splrep(delta, par_a)
            tck3 = interpolate.splrep(delta, par_b)
            tck4 = interpolate.splrep(delta, par_c)

            par1 = interpolate.splev(np.log10(dsoz), tck1)
            par2 = interpolate.splev(np.log10(dsoz), tck2)
            par3 = interpolate.splev(np.log10(dsoz), tck3)
            par4 = interpolate.splev(np.log10(dsoz), tck4)

            alpha = 10.**(-((0.75/np.log10(dsoz/75.))**1.2))
            A     = par1*((1. + zarr)**(-0.14))
            a     = par2*((1. + zarr)**(-0.06))
            b     = par3*((1. + zarr)**(-alpha))
            c     = par4*np.ones(zarr.size)

            return A * (1. + (sgm/b)**(-a)) * np.exp(-c/(sgm**2.))

        dRdM = radius(np.exp(marr))/(3.*np.exp(marr))
        dRdM = dRdM[:,None]
        sigma = sigma_sq(R, k)**0.5
        sigma_prime = sigma_sq_prime(R, k)

        dndlnm = -rhom0 * tinker(sigma, zarr) * dRdM * (sigma_prime/(2.*sigma**2.))

        return dndlnm

    def _get_dVdzdO(self):

        dAz = self._get_DAz()
        Hz = self.theory.get_Hubble(self.zarr)
        h = self.theory.get_param("H0") / 100.0

        dVdzdO = (c_ms/1e3)*(((1. + self.zarr)*dAz)**2.)/Hz
        dVdzdO *= h**3.

        return dVdzdO

    def _get_integrated(self, pk_intp, **kwargs):
        #print("hellooooo this is get_integrated")

        zarr = self.zarr
        marr = np.exp(self.marr)
        dlnm = self.dlnm

        y0 = self._get_y0(marr)
        dVdzdO = self._get_dVdzdO()
        dndlnm = self._get_dndlnm(pk_intp, **kwargs)
        surveydeg2 = self.skyfracs.sum()
        intgr = dndlnm*dVdzdO*surveydeg2
        intgr = intgr.T

        c = self._get_completeness(marr, zarr, y0)

        delN = np.zeros(len(zarr))
        i = 0
        j = 0
        for i in range(len(zarr)-1):
            for j in range(len(marr)-1):
                delN[i] += 0.5*(intgr[i,j]*c[i,j] + intgr[i+1,j]*c[i+1,j])*(zarr[i+1] - zarr[i])*dlnm
                #delN[i] += 0.5*(intgr[i,j] + intgr[i+1,j])*(zarr[i+1] - zarr[i])*dlnm
            print(i, delN[i])
        print("\r Total predicted N = ", delN.sum())

        return delN


    def _get_integrated2D(self, pk_intp, **kwargs):
        #print("hellooooo this is get_integrated2D")

        zarr = self.zarr
        marr = np.exp(self.marr)
        dlnm = self.dlnm
        Nq = self.Nq

        y0 = self._get_y0(marr)
        dVdzdO = self._get_dVdzdO()
        dndlnm = self._get_dndlnm(pk_intp, **kwargs)
        surveydeg2 = self.skyfracs.sum()
        intgr = dndlnm*dVdzdO*surveydeg2
        intgr = intgr.T

        print("2D completeness calculation starts now!")
        cc = []
        kk = 0
        for kk in range(Nq+1):
            cc.append(self._get_completeness2D(marr, zarr, y0, kk))
        cc = np.asarray(cc)
        #print(cc.shape)

        delN2D = np.zeros((len(zarr), Nq+1))
        i = 0
        j = 0
        kk = 0
        for kk in range(Nq+1):
            for i in range(len(zarr)-1):
                for j in range(len(marr)-1):
                    delN2D[i,kk] += 0.5*(intgr[i,j]*cc[kk,i,j] + intgr[i+1,j]*cc[kk,i+1,j])*(zarr[i+1] - zarr[i])*dlnm
                #print(i, delN2D[i,:].sum())
            #print(kk, delN2D[:,kk], delN2D[:,kk].sum())
            print(kk, delN2D[:,kk].sum())
        print("\r Total predicted 2D N = ", delN2D.sum())

        for i in range(len(zarr)-1):
            print(i, delN2D[i,:].sum())


        return delN2D


    def _get_theory(self, pk_intp, **kwargs):
        print("hellooooo this is get_theory")

        start = t.time()

        if self.choose_dim == '1D':
            delN = self._get_integrated(pk_intp, **kwargs)
        else:
            delN = self._get_integrated2D(pk_intp, **kwargs)

        elapsed = t.time() - start
        print("\r ::: theory N calculation took %.1f seconds" %elapsed)

        return delN


    # y-m scaling relation for completeness
    def _get_y0(self, mass):

        single_tile = self.single_tile_test
        Q_opt = self.Q_optimise

        A0 = 4.95e-5
        B0 = 0.08

        Ez = self._get_Ez()

        def theta(m):

            Hz = self.theory.get_Hubble(self.zarr)
            h = self.theory.get_param("H0") / 100.0
            DAz = self._get_DAz() ###################

            #DAz[0] = 2.99999995e-05/h  # value from f90
            m = m*msun
            Hz = Hz[:,None] * 1e3/Mpc
            DAz = DAz[:,None] * Mpc
            theta = np.rad2deg((G*m/250.)**(1./3) * (Hz**(-2./3))/DAz)*60.

            theta[theta > 500.] = 500.   # check this again
            #theta[theta < 0.5] = 0.5

            return theta

        def splQ(x):

            if single_tile == 'yes' or Q_opt == 'yes':
                tck = interpolate.splrep(self.tt500, self.Q)
                newQ = interpolate.splev(x, tck)
            else:
                newQ = []
                i = 0
                for i in range(len(self.Q[0])):
                    tck = interpolate.splrep(self.tt500, self.Q[:,i])
                    newQ.append(interpolate.splev(x, tck))
                newQ = np.asarray(np.abs(newQ))
            return newQ

        def rel(m):
            Ez = self._get_Ez()
            m = m*msun/m_pivot
            t = -0.008488*(m*Ez[:,None])**(-0.585)
            rel = 1 + 3.79*t - 28.2*(t**2.)
            return rel

        if single_tile == 'yes' or Q_opt == 'yes':
            m = mass[:,None]*msun/m_pivot
            y0 = A0*(Ez**2.) * m**(1. + B0) * splQ(theta(mass)).T * rel(mass).T
            y0 = y0.T
        else:
            m = mass[:,None]*msun/m_pivot
            arg = A0*(Ez**2.) * m**(1. + B0)
            y0 = arg[:,:,None] * splQ(theta(mass)).T * rel(mass).T[:,:,None]

        return y0

    # completeness 1D
    def _get_completeness(self, marr, zarr, y0):
        print("completeness calculation starts now!")

        scatter = self.scatter
        noise = self.noise
        qcut = self.qcut
        skyfracs = self.skyfracs/self.skyfracs.sum()
        Npatches = len(skyfracs)
        single_tile = self.single_tile_test
        Q_opt = self.Q_optimise
        if single_tile == 'no' and Q_opt == 'no': tilename = self.tilenames

        if scatter == 0.:
            a_pool = multiprocessing.Pool()
            completeness = a_pool.map(partial(get_comp_zarr,
                                        Nm=len(marr),
                                        qcut=qcut,
                                        noise=noise,
                                        skyfracs=skyfracs,
                                        lnyy=None,
                                        dyy=None,
                                        yy=None,
                                        y0=y0,
                                        temp=None,
                                        single_tile=single_tile,
                                        tile=None if single_tile == 'yes' or Q_opt == 'yes' else tilename,
                                        Q_opt=Q_opt,
                                        scatter=scatter),range(len(zarr)))
        else :
            lnymin = -25.     #ln(1e-10) = -23
            lnymax = 0.       #ln(1e-2) = -4.6
            dlny = 0.05
            Ny = m.floor((lnymax - lnymin)/dlny)
            temp = []
            yy = []
            lnyy = []
            dyy = []
            i = 0
            lny = lnymin

            if single_tile == 'yes' or Q_opt == "yes":

                for i in range(Ny):
                    y = np.exp(lny)
                    arg = (y - qcut*noise)/np.sqrt(2.)/noise
                    erfunc = (special.erf(arg) + 1.)/2.
                    temp.append(np.dot(erfunc, skyfracs))
                    yy.append(y)
                    lnyy.append(lny)
                    dyy.append(np.exp(lny + dlny*0.5) - np.exp(lny - dlny*0.5))
                    lny += dlny
                temp = np.asarray(temp)
                yy = np.asarray(yy)
                lnyy = np.asarray(lnyy)
                dyy = np.asarray(dyy)

            else:
                for i in range(Ny):
                    #print("first loop", i)
                    y = np.exp(lny)
                    j = 0
                    for j in range(Npatches):
                        arg = (y - qcut*noise[j])/np.sqrt(2.)/noise[j]
                        erfunc = (special.erf(arg) + 1.)/2.
                        temp.append(erfunc*skyfracs[j])
                        yy.append(y)
                        lnyy.append(lny)
                        dyy.append(np.exp(lny + dlny*0.5) - np.exp(lny - dlny*0.5))
                    lny += dlny
                temp = np.asarray(np.array_split(temp, Npatches))
                yy = np.asarray(np.array_split(yy, Npatches))
                lnyy = np.asarray(np.array_split(lnyy, Npatches))
                dyy = np.asarray(np.array_split(dyy, Npatches))

            a_pool = multiprocessing.Pool()
            completeness = a_pool.map(partial(get_comp_zarr,
                                                Nm=len(marr),
                                                qcut=None,
                                                noise=None,
                                                skyfracs=skyfracs,
                                                lnyy=lnyy,
                                                dyy=dyy,
                                                yy=yy,
                                                y0=y0,
                                                temp=temp,
                                                single_tile=single_tile,
                                                tile=None if single_tile == 'yes' or Q_opt == 'yes' else tilename,
                                                Q_opt=Q_opt,
                                                scatter=scatter),range(len(zarr)))
        a_pool.close()
        comp = np.asarray(completeness)
        comp[comp < 0.] = 0.
        comp[comp > 1.] = 1.

        return comp

    # completeness 2D
    def _get_completeness2D(self, marr, zarr, y0, qbin):
        #print("2D completeness calculation starts now!")

        scatter = self.scatter
        noise = self.noise
        qcut = self.qcut
        skyfracs = self.skyfracs/self.skyfracs.sum()
        Npatches = len(skyfracs)
        single_tile = self.single_tile_test
        Q_opt = self.Q_optimise
        if single_tile == 'no' and Q_opt == 'no': tilename = self.tilenames

        Nq = self.Nq
        qarr = self.qarr
        dlogq = self.dlogq

        if scatter == 0.:
            a_pool = multiprocessing.Pool()
            completeness = a_pool.map(partial(get_comp_zarr2D,
                                            Nm=len(marr),
                                            qcut=qcut,
                                            noise=noise,
                                            skyfracs=skyfracs,
                                            y0=y0,
                                            Nq=Nq,
                                            qarr=qarr,
                                            dlogq=dlogq,
                                            qbin=qbin,
                                            lnyy=None,
                                            dyy=None,
                                            yy=None,
                                            temp=None,
                                            single_tile=single_tile,
                                            Q_opt=Q_opt,
                                            tile=None if single_tile == 'yes' or Q_opt == 'yes' else tilename,
                                            scatter=scatter),range(len(zarr)))


        else:
            lnymin = -25.     #ln(1e-10) = -23
            lnymax = 0.       #ln(1e-2) = -4.6
            dlny = 0.05
            Ny = m.floor((lnymax - lnymin)/dlny)
            temp = []
            yy = []
            lnyy = []
            dyy = []
            lny = lnymin
            i = 0

            if single_tile == 'yes' or Q_opt == "yes":

                for i in range(Ny):
                    yy0 = np.exp(lny)

                    kk = qbin
                    qmin = qarr[kk] - dlogq/2.
                    qmax = qarr[kk] + dlogq/2.
                    qmin = 10.**qmin
                    qmax = 10.**qmax

                    if kk == 0:
                        cc = get_erf(yy0, noise, qcut)*(1. - get_erf(yy0, noise, qmax))
                    elif kk == Nq:
                        cc = get_erf(yy0, noise, qcut)*get_erf(yy0, noise, qmin)
                    else:
                        cc = get_erf(yy0, noise, qcut)*get_erf(yy0, noise, qmin)*(1. - get_erf(yy0, noise, qmax))

                    temp.append(np.dot(cc.T, skyfracs))
                    yy.append(yy0)
                    lnyy.append(lny)
                    dyy.append(np.exp(lny + dlny*0.5) - np.exp(lny - dlny*0.5))
                    lny += dlny

                temp = np.asarray(temp)
                yy = np.asarray(yy)
                lnyy = np.asarray(lnyy)
                dyy = np.asarray(dyy)

            else:

                for i in range(Ny):
                    yy0 = np.exp(lny)

                    kk = qbin
                    qmin = qarr[kk] - dlogq/2.
                    qmax = qarr[kk] + dlogq/2.
                    qmin = 10.**qmin
                    qmax = 10.**qmax

                    j = 0
                    for j in range(Npatches):
                        if kk == 0:
                            cc = get_erf(yy0, noise[j], qcut)*(1. - get_erf(yy0, noise[j], qmax))
                        elif kk == Nq:
                            cc = get_erf(yy0, noise[j], qcut)*get_erf(yy0, noise[j], qmin)
                        else:
                            cc = get_erf(yy0, noise[j], qcut)*get_erf(yy0, noise[j], qmin)*(1. - get_erf(yy0, noise[j], qmax))

                        temp.append(cc*skyfracs[j])
                        yy.append(yy0)
                        lnyy.append(lny)
                        dyy.append(np.exp(lny + dlny*0.5) - np.exp(lny - dlny*0.5))
                    lny += dlny

                temp = np.asarray(np.array_split(temp, Npatches))
                yy = np.asarray(np.array_split(yy, Npatches))
                lnyy = np.asarray(np.array_split(lnyy, Npatches))
                dyy = np.asarray(np.array_split(dyy, Npatches))

            a_pool = multiprocessing.Pool()
            completeness = a_pool.map(partial(get_comp_zarr2D,
                                                Nm=len(marr),
                                                qcut=qcut,
                                                noise=noise,
                                                skyfracs=skyfracs,
                                                y0=y0,
                                                Nq=Nq,
                                                qarr=qarr,
                                                dlogq=dlogq,
                                                qbin=qbin,
                                                lnyy=lnyy,
                                                dyy=dyy,
                                                yy=yy,
                                                temp=temp,
                                                single_tile=single_tile,
                                                Q_opt=Q_opt,
                                                tile=None if single_tile == 'yes' or Q_opt == 'yes' else tilename,
                                                scatter=scatter),range(len(zarr)))

        a_pool.close()
        comp = np.asarray(completeness)
        comp[comp < 0.] = 0.
        comp[comp > 1.] = 1.

        return comp


def get_comp_zarr(index_z, Nm, qcut, noise, skyfracs, lnyy, dyy, yy, y0, temp, single_tile, Q_opt, tile, scatter):

    i = 0
    res = []
    for i in range(Nm):

        if scatter == 0.:

            if single_tile == 'yes' or Q_opt == 'yes':
                arg = get_erf(y0[index_z, i], noise, qcut)
            else:
                j = 0
                arg = []
                for j in range(len(skyfracs)):
                    arg.append(get_erf(y0[i, index_z, int(tile[j])-1], noise[j], qcut))
                arg = np.asarray(arg)
            res.append(np.dot(arg, skyfracs))

        else:

            fac = 1./np.sqrt(2.*pi*scatter**2)
            mu = np.log(y0)
            if single_tile == 'yes' or Q_opt == 'yes':
                arg = (lnyy - mu[index_z, i])/(np.sqrt(2.)*scatter)
                res.append(np.dot(temp, fac*np.exp(-arg**2.)*dyy/yy))
            else:
                # make here faster
                # can i get rid of a loop here?
                j = 0
                args = 0.
                for j in range(len(skyfracs)):
                    #rint("second loop", j) # most of time takes here
                    arg = (lnyy[j,:] - mu[i, index_z, int(tile[j])-1])/(np.sqrt(2.)*scatter)
                    args += np.dot(temp[j,:], fac*np.exp(-arg**2.)*dyy[j,:]/yy[j,:])
                res.append(args)

    return res

def get_comp_zarr2D(index_z, Nm, qcut, noise, skyfracs, y0, Nq, qarr, dlogq, qbin, lnyy, dyy, yy, temp, single_tile, Q_opt, tile, scatter):

    kk = qbin
    qmin = qarr[kk] - dlogq/2.
    qmax = qarr[kk] + dlogq/2.
    qmin = 10.**qmin
    qmax = 10.**qmax

    i = 0
    res = []
    for i in range(Nm):

        if scatter == 0.:

            if single_tile == 'yes' or Q_opt == "yes":
                if kk == 0:
                    erfunc = get_erf(y0[index_z,i], noise, qcut)*(1. - get_erf(y0[index_z,i], noise, qmax))
                elif kk == Nq:
                    erfunc = get_erf(y0[index_z,i], noise, qcut)*get_erf(y0[index_z,i], noise, qmin)
                else:
                    erfunc = get_erf(y0[index_z,i], noise, qcut)*get_erf(y0[index_z,i], noise, qmin)*(1. - get_erf(y0[index_z,i], noise, qmax))
            else:
                j = 0
                erfunc = []
                for j in range(len(skyfracs)):
                    if kk == 0:
                        erfunc.append(get_erf(y0[i,index_z,int(tile[j])-1], noise[j], qcut)*(1. - get_erf(y0[i,index_z,int(tile[j]-1)], noise[j], qmax)))
                    elif kk == Nq:
                        erfunc.append(get_erf(y0[i,index_z,int(tile[j])-1], noise[j], qcut)*get_erf(y0[i,index_z,int(tile[j])-1], noise[j], qmin))
                    else:
                        erfunc.append(get_erf(y0[i,index_z,int(tile[j])-1], noise[j], qcut)*get_erf(y0[i,index_z,int(tile[j])-1], noise[j], qmin)*(1. - get_erf(y0[i,index_z,int(tile[j])-1], noise[j], qmax)))
                erfunc = np.asarray(erfunc)
            res.append(np.dot(erfunc, skyfracs))

        else:

            fac = 1./np.sqrt(2.*pi*scatter**2)
            mu = np.log(y0)
            if single_tile == 'yes' or Q_opt == "yes":
                arg = (lnyy - mu[index_z,i])/(np.sqrt(2.)*scatter)
                res.append(np.dot(temp, fac*np.exp(-arg**2.)*dyy/yy))
            else:
                j = 0
                args = 0.
                for j in range(len(skyfracs)):
                    #rint("second loop", j) # most of time takes here
                    arg = (lnyy[j,:] - mu[i, index_z, int(tile[j])-1])/(np.sqrt(2.)*scatter)
                    args += np.dot(temp[j,:], fac*np.exp(-arg**2.)*dyy[j,:]/yy[j,:])
                res.append(args)

    return res

def get_erf(y, rms, cut):
    arg = (y - cut*rms)/np.sqrt(2.)/rms
    erfc = (special.erf(arg) + 1.)/2.
    return erfc



class BinnedClusterLikelihoodPlanck(BinnedPoissonLikelihood):

    name = "BinnedClusterPlanck"
    plc_data_path: Optional[str] = None
    plc_cat_file: Optional[str] = None
    plc_thetas_file: Optional[str] = None
    plc_skyfracs_file: Optional[str] = None
    plc_ylims_file: Optional[str] = None
    choose_dim: Optional[str] = None

    def initialize(self):

        print('\r :::::: this is initialisation in binned_clusters.py')
        print('\r :::::: reading Planck 2015 catalogue')

        # full sky (sky fraction handled in skyfracs file)
        self.surveydeg2 = 41253.0*3.046174198e-4
        # signal-to-noise threshold
        self.qcut = 6.

        # y-m scaling relation parameters FIXME : this should be from yaml!
        self.alpha = 1.789
        ystar = -0.186
        self.ystar = (10.**ystar)/(2.**self.alpha)*0.00472724 ##########
        self.beta = 0.6666666
        self.scatter = 0.075 ##########
        print("\r intrinsic scatter = ", self.scatter)

        self.k = np.logspace(-4, np.log10(5), 200, endpoint=False)

        self.lnmmin = 31. #np.log(1e13)
        self.lnmmax = 37. #np.log(1e16)
        self.dlnm = 0.05
        self.marr = np.arange(self.lnmmin+self.dlnm, self.lnmmax+self.dlnm, self.dlnm)
        print("\r Nm = ", len(self.marr))

        # loading the catalogue
        self.data_directory = self.plc_data_path
        self.datafile = self.plc_cat_file
        cat = np.loadtxt(os.path.join(self.data_directory, self.datafile))
        zcat = cat[:,0]
        #zErrcat = cat[:,1] ######### do we not use this at all?
        qcat = cat[:,2]

        Ncat = len(zcat)
        print('\r Number of clusters in catalogue = ', Ncat)
        print('\r SNR cut = ', self.qcut)

        znew = []
        #zErrnew = []
        snrnew= []
        i = 0
        for i in range(Ncat):
            if qcat[i] > self.qcut:
                znew.append(zcat[i])
                #zErrnew.append(zErrcat[i])
                snrnew.append(qcat[i])

        z = np.array(znew)
        #zErr = np.array(zErrnew)
        snr = np.array(snrnew)
        Ncat = len(z)
        print('\r Number of clusters above the SNR cut = ', Ncat)

        # 1D catalogue
        print('\r :::::: binning clusters according to their redshifts')

        # redshift bin for N(z)
        zarr = np.linspace(0, 1, 11) # [0, 0.1, 0.2,...,0.9,1.0] 11
        if zarr[0] == 0 :zarr[0] = 1e-5
        self.zarr = zarr

        zmin = 0.
        dz = 0.1
        zmax = zmin + dz
        delNcat = np.zeros(len(zarr))
        i = 0
        j = 0
        for i in range(len(zarr)):
            for j in range(Ncat):
                if z[j] >= zmin and z[j] < zmax :
                    delNcat[i] += 1.
            zmin = zmin + dz
            zmax = zmax + dz
            #print(i, zmin, zmax, delNcat[i].sum())

        print("\r Number of redshift bins = ", len(zarr)-1) # last bin is empty anyway
        print("\r Catalogue N = ", delNcat, delNcat.sum())

        # rescaling for missing redshift
        Nmiss = 0
        i = 0
        for i in range(Ncat):
            if z[i] < 0.:
                Nmiss += 1

        Ncat2 = Ncat - Nmiss
        print('\r Number of clusters with redshift = ', Ncat2)
        print('\r Number of clusters without redshift = ', Nmiss)

        rescale = Ncat/Ncat2

        if Nmiss != 0:
            print("\r Rescaling for missing redshifts ", rescale)

        delNcat *= rescale
        print("\r Rescaled Catalogue N = ", delNcat, delNcat.sum())

        self.delNcat = zarr, delNcat

        # 2D catalogue
        if self.choose_dim == "2D":
            print('\r :::::: binning clusters according to their SNRs')

        logqmin = 0.7  # log10(4)  = 0.778 --- min snr = 6
        logqmax = 1.5  # log10(35) = 1.505 --- max snr = 32
        dlogq = 0.25

        Nq = int((logqmax - logqmin)/dlogq) + 1  ########
        if self.choose_dim == "2D":
            print("\r Number of SNR bins = ", Nq+1)

        qi = logqmin + dlogq/2.
        qarr = np.zeros(Nq+1)

        i = 0
        for i in range(Nq+1):
            qarr[i] = qi
            qi = qi + dlogq
        if self.choose_dim == "2D":
            print("\r Center of SNR bins = ", 10**qarr)#[:-1])

        zmin = zarr[0]
        zmax = zmin + dz

        delN2Dcat = np.zeros((len(zarr), Nq+1))

        i = 0
        j = 0
        k = 0
        for i in range(len(zarr)):
           for j in range(Nq):
                qmin = qarr[j] - dlogq/2.
                qmax = qarr[j] + dlogq/2.
                qmin = 10.**qmin
                qmax = 10.**qmax

                for k in range(Ncat):
                    if z[k] >= zmin and z[k] < zmax and snr[k] >= qmin and snr[k] < qmax :
                        delN2Dcat[i,j] += 1

           j = Nq + 1 # the last bin contains all S/N greater than what in the previous bin
           qmin = qmax

           for k in range(Ncat):
               if z[k] >= zmin and z[k] < zmax and snr[k] >= qmin :
                   delN2Dcat[i,j] += 1

           zmin = zmin + dz
           zmax = zmax + dz

        if self.choose_dim == "2D":
            print("\r Catalogue 2D N = ", delN2Dcat.sum())
            j = 0
            for j in range(Nq+1):
                    print(j, delN2Dcat[:,j], delN2Dcat[:,j].sum())

        # missing redshifts
        i = 0
        j = 0
        k = 0
        for j in range(Nq):
            qmin = qarr[j] - dlogq/2.
            qmax = qarr[j] + dlogq/2.
            qmin = 10.**qmin
            qmax = 10.**qmax

            for k in range(Ncat):
                if z[k] == -1. and snr[k] >= qmin and snr[k] < qmax :
                    norm = 0.
                    for i in range(len(zarr)):
                        norm += delN2Dcat[i,j]
                    delN2Dcat[:,j] *= (norm + 1.)/norm

        j = Nq + 1 # the last bin contains all S/N greater than what in the previous bin
        qmin = qmax
        for k in range(Ncat):
            if z[k] == -1. and snr[k] >= qmin :
                norm = 0.
                for i in range(len(zarr)):
                    norm += delN2Dcat[i,j]
                delN2Dcat[:,j] *= (norm + 1.)/norm

        if self.choose_dim == "2D":
            print("\r Rescaled Catalogue 2D N = ", delN2Dcat.sum())
            j = 0
            for j in range(Nq+1):
                    print(j, delN2Dcat[:,j], delN2Dcat[:,j].sum())


        self.Nq = Nq
        self.qarr = qarr
        self.dlogq = dlogq
        self.delN2Dcat = zarr, qarr, delN2Dcat

        print('\r :::::: loading files describing selection function')

        self.datafile = self.plc_thetas_file
        thetas = np.loadtxt(os.path.join(self.data_directory, self.datafile))
        print('\r Number of size thetas = ', len(thetas))

        self.datafile = self.plc_skyfracs_file
        skyfracs = np.loadtxt(os.path.join(self.data_directory, self.datafile))
        print('\r Number of size skypatches = ', len(skyfracs))

        self.datafile = self.plc_ylims_file
        ylims0 = np.loadtxt(os.path.join(self.data_directory, self.datafile))
        print('\r Number of size ylims = ', len(ylims0))
        if len(ylims0) != len(thetas)*len(skyfracs):
            raise ValueError("Format error for ylims.txt \n" +\
                             "Expected rows : {} \n".format(len(thetas)*len(skyfracs)) +\
                             "Actual rows : {}".format(len(ylims0)))

        ylims = np.zeros((len(skyfracs), len(thetas)))

        i = 0
        j = 0
        k = 0
        for k in range(len(ylims0)):
            ylims[i,j] = ylims0[k]
            i += 1
            if i > len(skyfracs)-1:
                i = 0
                j += 1

        self.thetas = thetas
        self.skyfracs = skyfracs
        self.ylims = ylims


        # high resolution redshift bins
        minz = zarr[0]
        maxz = zarr[-1]
        if minz < 0: minz = 0.
        zi = minz

        # counting redshift bins
        Nzz = 0
        while zi <= maxz :
            zi = self._get_hres_z(zi)
            Nzz += 1

        Nzz += 1
        zi = minz
        zz = np.zeros(Nzz)
        for i in range(Nzz): # [0-279]
            zz[i] = zi
            zi = self._get_hres_z(zi)
        if zz[0] == 0. : zz[0] = 1e-5 # = steps_z(Nz) in f90
        self.zz = zz
        print(" Nz for higher resolution = ", len(zz))


        # redshift bin for P(z,k)
        zpk = np.linspace(0, 1, 100) #[0, 0.01, 0.02,...,0.99] 100
        if zpk[0] == 0. : zpk[0] = 1e-5
        self.zpk = zpk
        print(" Nz for matter power spectrum = ", len(zpk))


        super().initialize()

    def get_requirements(self):
        return {"Hubble":  {"z": self.zz},
                "angular_diameter_distance": {"z": self.zz},
                "Pk_interpolator": {"z": self.zpk,
                                    "k_max": 5,  # is this transfer kmax?
                                    "nonlinear": False,
                                    "hubble_units": False,
                                    "k_hunit": False,
                                    "vars_pairs": [["delta_nonu", "delta_nonu"]]},
                "ombh2":None, "H0":None, "omch2":None, "omegam":None, "sigma8":None}

    def _get_data(self):
        return self.delNcat, self.delN2Dcat

    def _get_om(self):
        return (self.theory.get_param("omch2") + self.theory.get_param("ombh2"))/((self.theory.get_param("H0")/100.0)**2)

    def _get_Hz(self, z):
        return self.theory.get_Hubble(z)

    def _get_Ez(self, z):
        return self.theory.get_Hubble(z)/self.theory.get_param("H0")

    def _get_DAz(self, z):
        return self.theory.get_angular_diameter_distance(z)

    def _get_hres_z(self, zi):
        # bins in redshifts are defined with higher resolution for z < 0.2
        hr = 0.2
        if zi < hr :
            dzi = 1e-3
            #dzi = 1e-2
        else:
            dzi = 1e-2
        hres_z = zi + dzi
        return hres_z

    def _get_dndlnm(self, z, pk_intp, **kwargs):

        h = self.theory.get_param("H0")/100.0
        Ez = self._get_Ez(z)
        om = self._get_om()
        rhom0 = rhocrit0*om

        #zarr = self.zarr
        #k = self.k # why does this change?
        k = np.logspace(-4, np.log10(5), 200, endpoint=False)
        zpk = self.zpk
        pks0 = pk_intp.P(zpk, k)

        def pks_zbins(newz):
            i = 0
            newpks = np.zeros((len(newz),len(k)))
            for i in range(k.size):
                tck = interpolate.splrep(zpk, pks0[:,i])
                newpks[:,i] = interpolate.splev(newz, tck)
            return newpks
        pks = pks_zbins(z)

        pks *= h**3.
        k /= h

        def radius(M):
            return (0.75*M/pi/rhom0)**(1./3.)

        def win(x):
            return 3.*(np.sin(x) - x*np.cos(x))/(x**3.)

        def win_prime(x):
            return 3.*np.sin(x)/(x**2.) - 9.*(np.sin(x) - x*np.cos(x))/(x**4.)

        marr = self.marr
        R = radius(np.exp(marr))[:,None]

        def sigma_sq(R, k):
            integral = np.zeros((len(k), len(marr), len(z)))
            i = 0
            for i in range(k.size):
                integral[i,:,:] = np.array((k[i]**2.)*pks[:,i]*(win(k[i]*R)**2.))
            return integrate.simps(integral, k, axis=0)/(2.*pi**2.)

        def sigma_sq_prime(R, k):
            integral = np.zeros((len(k), len(marr), len(z)))
            i = 0
            for i in range(k.size):
                integral[i,:,:] = np.array((k[i]**2.)*pks[:,i]*2.*k[i]*win(k[i]*R)*win_prime(k[i]*R))
            return integrate.simps(integral, k, axis=0)/(2.*pi**2.)

        def tinker(sgm, z):

            total = 9

            delta  = np.zeros(total)
            par_aa = np.zeros(total)
            par_a  = np.zeros(total)
            par_b  = np.zeros(total)
            par_c  = np.zeros(total)

            delta[0] = 200
            delta[1] = 300
            delta[2] = 400
            delta[3] = 600
            delta[4] = 800
            delta[5] = 1200
            delta[6] = 1600
            delta[7] = 2400
            delta[8] = 3200

            par_aa[0] = 0.186
            par_aa[1] = 0.200
            par_aa[2] = 0.212
            par_aa[3] = 0.218
            par_aa[4] = 0.248
            par_aa[5] = 0.255
            par_aa[6] = 0.260
            par_aa[7] = 0.260
            par_aa[8] = 0.260

            par_a[0] = 1.47
            par_a[1] = 1.52
            par_a[2] = 1.56
            par_a[3] = 1.61
            par_a[4] = 1.87
            par_a[5] = 2.13
            par_a[6] = 2.30
            par_a[7] = 2.53
            par_a[8] = 2.66

            par_b[0] = 2.57
            par_b[1] = 2.25
            par_b[2] = 2.05
            par_b[3] = 1.87
            par_b[4] = 1.59
            par_b[5] = 1.51
            par_b[6] = 1.46
            par_b[7] = 1.44
            par_b[8] = 1.41

            par_c[0] = 1.19
            par_c[1] = 1.27
            par_c[2] = 1.34
            par_c[3] = 1.45
            par_c[4] = 1.58
            par_c[5] = 1.80
            par_c[6] = 1.97
            par_c[7] = 2.24
            par_c[8] = 2.44

            # use derivatives as well?

            delta = np.log10(delta)

            dso = 500.
            omz = om*((1. + z)**3.)/(Ez**2.)
            dsoz = dso/omz

            tck1 = interpolate.splrep(delta, par_aa)
            tck2 = interpolate.splrep(delta, par_a)
            tck3 = interpolate.splrep(delta, par_b)
            tck4 = interpolate.splrep(delta, par_c)

            par1 = interpolate.splev(np.log10(dsoz), tck1)
            par2 = interpolate.splev(np.log10(dsoz), tck2)
            par3 = interpolate.splev(np.log10(dsoz), tck3)
            par4 = interpolate.splev(np.log10(dsoz), tck4)

            alpha = 10.**(-((0.75/np.log10(dsoz/75.))**1.2))
            A     = par1*((1. + z)**(-0.14))
            a     = par2*((1. + z)**(-0.06))
            b     = par3*((1. + z)**(-alpha))
            c     = par4*np.ones(z.size)

            #print("hereeeeeeee", dsoz)

            return A * (1. + (sgm/b)**(-a)) * np.exp(-c/(sgm**2.))

        dRdM = radius(np.exp(marr))/(3.*np.exp(marr))
        dRdM = dRdM[:,None]
        sigma = sigma_sq(R, k)**0.5
        sigma_prime = sigma_sq_prime(R, k)

        #print("inside tinker", sigma.shape)

        dndlnm = -rhom0 * tinker(sigma, z) * dRdM * (sigma_prime/(2.*sigma**2.))

        #print("hereeeeeeeeeeeeeee = ", tinker(sigma,z))

        return dndlnm

    def _get_dVdzdO(self, z):

        h = self.theory.get_param("H0") / 100.0
        DAz = self._get_DAz(z)
        Hz = self._get_Hz(z)

        dVdzdO = (c_ms/1e3)*(((1. + z)*DAz)**2.)/Hz
        dVdzdO *= h**3.

        return dVdzdO

    def _get_integrated(self, pk_intp, **kwargs):

        print("hellooooo this is get_integrated")

        qcut = self.qcut

        marr = np.exp(self.marr)
        dlnm = self.dlnm
        lnmmin = self.lnmmin

        zarr = self.zarr
        zz = self.zz

        y500 = self._get_y500(marr, zz)
        theta500 = self._get_theta500(marr, zz)
        dVdzdO = self._get_dVdzdO(zz)
        dndlnm = self._get_dndlnm(zz, pk_intp, **kwargs)

        surveydeg2 = 41253.0*3.046174198e-4
        intgr = dndlnm*dVdzdO*surveydeg2
        intgr = intgr.T

        c = self._get_completeness(marr, zz, y500, theta500)

        delN = np.zeros(len(zarr))
        i = 0
        for i in range(len(zarr)-1):

            test = np.abs(zz - zarr[i])
            i1 = np.argmin(test)
            test = np.abs(zz - zarr[i+1])
            i2 = np.argmin(test)
            zs = np.arange(i1, i2)

            sum = 0.
            sumzs = np.zeros(len(zz))
            ii = 0
            for ii in zs:
                j = 0
                for j in range(len(marr)-1):
                    sumzs[ii] += 0.5*(intgr[ii,j]*c[ii,j] + intgr[ii,j+1]*c[ii,j+1])*dlnm
                sum += sumzs[ii]*(zz[ii+1] - zz[ii])

            delN[i] = sum

            print(i, delN[i])

        print("\r Total predicted N = ", delN.sum())

        return delN


    def _get_integrated2D(self, pk_intp, **kwargs):

        print("hellooooo this is get_integrated2D")

        zarr = self.zarr
        zz = self.zz
        marr = np.exp(self.marr)
        dlnm = self.dlnm
        lnmmin = self.lnmmin
        qcut = self.qcut

        Nq = self.Nq
        qarr = self.qarr
        dlogq = self.dlogq

        y500 = self._get_y500(marr, zz)
        theta500 = self._get_theta500(marr, zz)
        dVdzdO = self._get_dVdzdO(zz)
        dndlnm = self._get_dndlnm(zz, pk_intp, **kwargs)

        surveydeg2 = 41253.0*3.046174198e-4
        intgr = dndlnm*dVdzdO*surveydeg2
        intgr = intgr.T

        cc = self._get_completeness2D(marr, zz, y500, theta500)

        delN2D = np.zeros((len(zarr), Nq+1))
        kk = 0
        for kk in range(Nq+1):
            i = 0
            for i in range(len(zarr)-1):

                test = np.abs(zz - zarr[i])
                i1 = np.argmin(test)
                test = np.abs(zz - zarr[i+1])
                i2 = np.argmin(test)
                zs = np.arange(i1, i2)
                #print(i1, i2)

                sum = 0.
                sumzs = np.zeros((len(zz), Nq+1))
                ii = 0
                for ii in zs:
                    j = 0
                    for j in range(len(marr)-1):
                        sumzs[ii,kk] += 0.5*(intgr[ii,j]*cc[ii,j,kk] + intgr[ii,j+1]*cc[ii,j+1,kk])*dlnm
                        #sumzs[ii,kk] += 0.5*(intgr[ii,j]*cc[kk,ii,j] + intgr[ii,j+1]*cc[kk,ii,j+1])*dlnm
                    sum += sumzs[ii,kk]*(zz[ii+1] - zz[ii])
                delN2D[i,kk] = sum

            #print(kk, delN2D[:,kk], delN2D[:,kk].sum())
            print(kk, delN2D[:,kk].sum())

        print("\r Total predicted 2D N = ", delN2D.sum())

        return delN2D


    def _get_theory(self, pk_intp, **kwargs):
        print("hellooooo this is get_theory")

        start = t.time()

        if self.choose_dim == '1D':
            res = self._get_integrated(pk_intp, **kwargs)
        else:
            res = self._get_integrated2D(pk_intp, **kwargs)

        elapsed = t.time() - start
        print("\r ::: theory N calculation took %.1f seconds" %elapsed)

        return res


    # y-m scaling relation for completeness

    def _get_theta500(self, m, z):

        thetastar = 6.997
        alpha_theta = 1./3.
        bias = 1

        #Hz = Hz[:,None] * 1e3/Mpc
        #DAz = DAz[:,None] * Mpc

        H0 = self.theory.get_param("H0")
        h = self.theory.get_param("H0") / 100.0
        Ez = self._get_Ez(z)
        DAz = self._get_DAz(z)*h

        m *=bias
        m = m[:,None]
        thetastar *= (H0/70.)**(-2./3.)
        theta500 = thetastar*(m/3e14*(100./H0))**alpha_theta * Ez**(-2./3.) * (100.*DAz/500/H0)**(-1.)

        return theta500

    def _get_y500(self, m, z): # come back here

        bias = 1
        ystar = self.ystar
        alpha = self.alpha
        beta = self.beta

        H0 = self.theory.get_param("H0")
        h = self.theory.get_param("H0") / 100.0
        Ez = self._get_Ez(z)
        DAz = self._get_DAz(z)*h

        m *= bias
        m = m[:,None]
        ystar *= (H0/70.)**(alpha - 2.)
        y500 = ystar*(m/3e14*(100./H0))**alpha * Ez**beta * (100.*DAz/500./H0)**(-2.)

        return y500

    # completeness

    def _get_completeness(self, marr, zarr, y500, theta500):

        scatter = self.scatter
        qcut = self.qcut
        thetas = self.thetas
        ylims = self.ylims
        skyfracs = self.skyfracs
        fsky = skyfracs.sum()

        lnymin = -11.5     #ln(1e-10) = -23
        lnymax = 10.       #ln(1e-2) = -4.6
        dlny = 0.05
        Ny = m.floor((lnymax - lnymin)/dlny)

        yylims = []
        yy = []
        lnyy = []
        dyy = []
        lny = lnymin
        i = 0
        for i in range(Ny):
            yy0 = np.exp(lny)
            arg = (yy0 - qcut*ylims)/np.sqrt(2.)/ylims
            erfunc = (special.erf(arg) + 1.)/2.
            yylims.append(np.dot(erfunc.T, skyfracs))
            yy.append(yy0)
            lnyy.append(lny)
            dyy.append(np.exp(lny + dlny*0.5) - np.exp(lny - dlny*0.5))
            lny += dlny

        yylims = np.asarray(yylims)
        yy = np.asarray(yy)
        lnyy = np.asarray(lnyy)
        dyy = np.asarray(dyy)

        a_pool = multiprocessing.Pool()
        completeness = a_pool.map(partial(get_comp_zarr_plc,
                                            Nm=len(marr),
                                            qcut=qcut,
                                            thetas=thetas,
                                            ylims=ylims,
                                            skyfracs=skyfracs,
                                            y500=y500,
                                            theta500=theta500,
                                            lnyy=lnyy,
                                            dyy=dyy,
                                            yy=yy,
                                            yylims=yylims,
                                            scatter=scatter),range(len(zarr)))
        a_pool.close()
        comp = np.asarray(completeness)
        comp[comp < 0.] = 0.
        comp[comp > fsky] = fsky

        return comp


    def _get_completeness2D(self, marr, zarr, y500, theta500):

        scatter = self.scatter
        qcut = self.qcut
        thetas = self.thetas
        skyfracs = self.skyfracs
        ylims = self.ylims
        fsky = skyfracs.sum()

        Nq = self.Nq
        qarr = self.qarr
        dlogq = self.dlogq

        th0 = theta500.T
        y0 = y500.T

        if scatter == 0:
            a_pool = multiprocessing.Pool()
            completeness = a_pool.map(partial(get_comp_zarr_plc2D,
                                                Nm=len(marr),
                                                qcut=qcut,
                                                thetas=thetas,
                                                ylims=ylims,
                                                skyfracs=skyfracs,
                                                y500=y500,
                                                theta500=theta500,
                                                qqarr=qqarr,
                                                qqmin=qqmin,
                                                qqmax=qqmax,
                                                lnyy=None,
                                                dyy=None,
                                                yy=None,
                                                yylims=None,
                                                scatter=scatter),range(len(zarr)))
        else:


            start0 = t.time()

            b_pool = multiprocessing.Pool()
            yylims, yy, lnyy, dyy = zip(*b_pool.map(partial(get_comp_qarr_plc2D,
                                                qarr=qarr,
                                                dlogq=dlogq,
                                                ylims=ylims,
                                                qcut=qcut,
                                                zarr=zarr,
                                                marr=marr,
                                                skyfracs=skyfracs,
                                                thetas=thetas),range(Nq+1)))

            b_pool.close()


            #print(yylims.shape, yy.shape) #(430, 5, 80) (430,)
            elapsed0 = t.time() - start0
            print("\r ::: here took %.1f seconds" %elapsed0)

            yylims = np.asarray(yylims)
            yy = np.asarray(yy)
            lnyy = np.asarray(lnyy)
            dyy = np.asarray(dyy)
            print(yylims.shape, lnyy.shape, yy.shape) #(4, 430, 80) (4, 430) (4, 430)

            a_pool = multiprocessing.Pool()
            completeness = a_pool.map(partial(get_comp_zarr_plc2D,
                                                Nm=len(marr),
                                                qcut=qcut,
                                                thetas=thetas,
                                                ylims=ylims,
                                                skyfracs=skyfracs,
                                                y500=y500,
                                                theta500=theta500,
                                                qqarr=None,
                                                qqmin=None,
                                                qqmax=None,
                                                lnyy=lnyy,
                                                dyy=dyy,
                                                yy=yy,
                                                yylims=yylims,
                                                scatter=scatter),range(len(zarr)))
        a_pool.close()
        comp = np.asarray(completeness)
        comp[comp < 0.] = 0.
        comp[comp > fsky] = fsky
        #comp[comp > 1.] = 1.
        #print("hmmmm", comp.shape)

        return comp

def get_comp_qarr_plc2D(qbin, qarr, dlogq, ylims, qcut, zarr, marr, skyfracs, thetas):

    lnymin = -11.5     #ln(1e-10) = -23
    lnymax = 10.       #ln(1e-2) = -4.6
    dlny = 0.05
    Ny = m.floor((lnymax - lnymin)/dlny)

    yylims = []
    yy = []
    lnyy = []
    dyy = []
    lny = lnymin
    i = 0

    #cc = np.zeros((len(zarr), len(marr), len(qarr)))
    #cc = np.zeros((len(skyfracs), len(thetas), len(qarr)))

    for i in range(Ny): # okay most time takes here - let's make here faster!
        yy0 = np.exp(lny) # what can i do here ... can i get rid of this loop? hmmm
        #maybe trying to optimise this cc computation first?

        kk = qbin
        qmin = qarr[kk] - dlogq/2.
        qmax = qarr[kk] + dlogq/2.
        qmin = 10.**qmin
        qmax = 10.**qmax
        Nq = len(qarr)

        #aaa = get_erf(yy0, ylims, qcut)*get_erf(yy0, ylims, qmin)*(1. - get_erf(yy0, ylims, qmax))
        #aaa.shape = 417, 80 -> Nskypatches, Nthetas
        # 417, 80, 5 -> transposed - 5, 80, 417

        #print("yo yo yo i am")

        cc = get_erf(yy0, ylims, qcut)*get_erf(yy0, ylims, qmin)*(1. - get_erf(yy0, ylims, qmax))
        if kk == 0:
            cc = get_erf(yy0, ylims, qcut)*(1. - get_erf(yy0, ylims, qmax))
        if kk == Nq:
            cc = get_erf(yy0, ylims, qcut)*get_erf(yy0, ylims, qmin)

        ##yylims.append(np.dot(cc.transpose(0,2,1), skyfracs))
        yylims.append(np.dot(cc.T, skyfracs))
        yy.append(yy0)
        lnyy.append(lny)
        dyy.append(np.exp(lny + dlny*0.5) - np.exp(lny - dlny*0.5))
        lny += dlny

    yylims = np.asarray(yylims)
    yy = np.asarray(yy)
    lnyy = np.asarray(lnyy)
    dyy = np.asarray(dyy)

    #print("heyyyyyy i am", yylims.shape, yy.shape, lnyy.shape, qbin) #(430, 80) (430,)
    #print("helllllllllooooo", yylims.shape, (430, 5, 80) yy.shape, lnyy.shape, dyy.shape)

    return yylims, yy, lnyy, dyy


def get_comp_zarr_plc(index_z, Nm, qcut, thetas, ylims, skyfracs, y500, theta500, lnyy, dyy, yy, yylims, scatter):

    Nthetas = len(thetas)
    min_thetas = thetas.min()
    max_thetas = thetas.max()
    dif_theta = np.zeros(Nthetas)
    th0 = theta500.T
    y0 = y500.T
    mu = np.log(y0)

    res = []
    i = 0
    for i in range(Nm):
        if th0[index_z,i] > max_thetas:
            l1 = Nthetas - 1
            l2 = Nthetas - 2
            th1 = thetas[l1]
            th2 = thetas[l2]
        elif th0[index_z,i] < min_thetas:
            l1 = 0
            l2 = 1
            th1 = thetas[l1]
            th2 = thetas[l2]
        else:
            dif_theta = np.abs(thetas - th0[index_z,i])
            l1 = np.argmin(dif_theta)
            th1 = thetas[l1]
            l2 = l1 + 1
            if th1 > th0[index_z,i] : l2 = l1 - 1
            th2 = thetas[l2]

        if scatter == 0:
            y1 = ylims[:,l1]
            y2 = ylims[:,l2]
            y = y1 + (y2 - y1)/(th2 - th1)*(th0[index_z,i] - th1)
            arg = get_erf(y0[index_z, i], y, qcut)
            res.append(np.dot(arg, skyfracs))
        else:
            y1 = yylims[:,l1]
            y2 = yylims[:,l2]
            y = y1 + (y2 - y1)/(th2 - th1)*(th0[index_z,i] - th1)
            fac = 1./np.sqrt(2.*pi*scatter**2)
            arg = (lnyy - mu[index_z,i])/(np.sqrt(2.)*scatter)
            res.append(np.dot(y, fac*np.exp(-arg**2.)*dyy/yy))

    return res

def get_comp_zarr_plc2D(index_z, Nm, qcut, thetas, ylims, skyfracs, y500, theta500, qqarr, qqmin, qqmax, lnyy, dyy, yy, yylims, scatter):
    Nthetas = len(thetas)
    min_thetas = thetas.min()
    max_thetas = thetas.max()
    dif_theta = np.zeros(Nthetas)
    th0 = theta500.T
    y0 = y500.T
    mu = np.log(y0)

    res = []
    i = 0
    for i in range(Nm):
        if th0[index_z,i] > max_thetas:
            l1 = Nthetas - 1
            l2 = Nthetas - 2
            th1 = thetas[l1]
            th2 = thetas[l2]
        elif th0[index_z,i] < min_thetas:
            l1 = 0
            l2 = 1
            th1 = thetas[l1]
            th2 = thetas[l2]
        else:
            dif_theta = np.abs(thetas - th0[index_z,i])
            l1 = np.argmin(dif_theta)
            th1 = thetas[l1]
            l2 = l1 + 1
            if th1 > th0[index_z,i] : l2 = l1 - 1
            th2 = thetas[l2]

        if scatter == 0:
            y1 = ylims[:,l1]
            y2 = ylims[:,l2]
            y = y1 + (y2 - y1)/(th2 - th1)*(th0[index_z,i] - th1)
            erf0 = get_erf(y0[index_z,i], y, qcut)*(1. - get_erf(y0[index_z,i], y, qqarr[1]))
            erf1 = get_erf(y0[index_z,i], y, qcut)*get_erf(y0[index_z,i], y, qqmin[:,None])*(1. -  get_erf(y0[index_z,i], y, qqmax[:,None]))
            erf2 = get_erf(y0[index_z,i], y, qcut)*get_erf(y0[index_z,i], y, qqarr[-2])
            erfunc = np.column_stack((erf0, erf1.T, erf2))
            res.append(np.dot(erfunc.T, skyfracs))

        else: # another 1-2 sec takes here but here is already optimised
            y1 = yylims[:,:,l1]
            y2 = yylims[:,:,l2]
            y = y1 + (y2 - y1)/(th2 - th1)*(th0[index_z,i] - th1)
            fac = 1./np.sqrt(2.*pi*scatter**2)
            arg = (lnyy - mu[index_z,i])/(np.sqrt(2.)*scatter)
            #print(y.shape, arg.shape, dyy.shape) # (4, 430) (4, 430) (4, 430)
            #res.append(np.dot(y.T, fac*np.exp(-arg**2.)*dyy/yy))
            fac *= np.exp(-arg**2.)*dyy/yy
            res.append(np.diagonal(np.dot(y, fac.T))) #-> 430, 430

            #print(res.shape, res.diagonal())

    return res

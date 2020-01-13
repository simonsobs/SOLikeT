import numpy as np
from scipy import interpolate
from astropy.cosmology import FlatLambdaCDM

#from nemo import signals

class szutils(object):
    def __init__(self,Survey):
        self.LgY = np.arange(-6, -2.5, 0.01)
        self.Survey = Survey

    def P_Yo(self, LgY, M, z, param_vals):
        Om = param_vals['om']
        Ob = param_vals['ob']
        OL = 1. - Om
        H0 = param_vals['H0']
        
        cosmoModel = FlatLambdaCDM(H0=H0, Om0=Om, Ob0=Ob, Tcmb0=2.725)
        
        Ytilde, theta0, Qfilt = y0FromLogM500(
            np.log10(param_vals['massbias']*M / (H0/100.)),
            z,
            self.Survey.tckQFit['Q'],
            sigma_int=param_vals['scat'],
            B0=param_vals['B0'],
            cosmoModel=cosmoModel,
            )
        Y = 10**LgY
        
        Ytilde = np.repeat(Ytilde[:, :, np.newaxis], LgY.shape[2], axis=2)
        
        numer = -1.*(np.log(Y/Ytilde))**2
        ans = 1./(param_vals['scat'] * np.sqrt(2*np.pi)) * np.exp(numer/(2.*param_vals['scat']**2))
        return ans

    def Y_erf(self, Y, Ynoise):
        qmin = self.Survey.qmin
        ans = Y*0.0
        ans[Y - qmin*Ynoise > 0] = 1.
        return ans

    def P_of_gt_SN(self, LgY, MM, zz, Ynoise, param_vals):
        Y = 10**LgY

        sig_tr = np.outer(np.ones([MM.shape[0], MM.shape[1]]), self.Y_erf(Y, Ynoise))
        sig_thresh = np.reshape(sig_tr, (MM.shape[0], MM.shape[1], len(self.Y_erf(Y, Ynoise))))

        LgYa = np.outer(np.ones([MM.shape[0], MM.shape[1]]), LgY)
        LgYa2 = np.reshape(LgYa, (MM.shape[0], MM.shape[1], len(LgY)))

        P_Y = np.nan_to_num(self.P_Yo(LgYa2, MM, zz, param_vals))

        ans = np.trapz(P_Y*sig_thresh, x=LgY, axis=2) * np.log(10)
        return ans

    def PfuncY(self, YNoise, M, z_arr, param_vals):
        LgY = self.LgY

        P_func = np.outer(M, np.zeros([len(z_arr)]))
        M_arr = np.outer(M, np.ones([len(z_arr)]))

        P_func = self.P_of_gt_SN(LgY, M_arr, z_arr, YNoise, param_vals)
        return P_func

    def P_of_Y_per(self, LgY, MM, zz, Y_c, Y_err, param_vals):
        P_Y_sig = np.outer(np.ones(len(MM)), self.Y_prob(Y_c, LgY, Y_err))
        LgYa = np.outer(np.ones(len(MM)), LgY)

        LgYa = np.outer(np.ones([MM.shape[0], MM.shape[1]]), LgY)
        LgYa2 = np.reshape(LgYa, (MM.shape[0], MM.shape[1], len(LgY)))

        P_Y = np.nan_to_num(self.P_Yo(LgYa2, MM, zz, param_vals))
        ans = np.trapz(P_Y*P_Y_sig, LgY, np.diff(LgY), axis=1) * np.log(10)
        return ans


###
'''Routines from nemo (author: Matt Hilton ) to limit dependencies'''

#------------------------------------------------------------------------------------------------------------
def calcR500Mpc(z, M500, cosmoModel):
    """Given z, M500 (in MSun), returns R500 in Mpc, with respect to critical density.
    
    """

    if type(M500) == str:
        raise Exception("M500 is a string - check M500MSun in your .yml config file: use, e.g., 1.0e+14 (not 1e14 or 1e+14)")

    Ez=cosmoModel.efunc(z)
    criticalDensity=cosmoModel.critical_density(z).value
    criticalDensity=(criticalDensity*np.power(Mpc_in_cm, 3))/MSun_in_g
    R500Mpc=np.power((3*M500)/(4*np.pi*500*criticalDensity), 1.0/3.0)
        
    return R500Mpc

#------------------------------------------------------------------------------------------------------------
def calcTheta500Arcmin(z, M500, cosmoModel):
    """Given z, M500 (in MSun), returns angular size equivalent to R500, with respect to critical density.
    
    """
    
    R500Mpc=calcR500Mpc(z, M500, cosmoModel)
    theta500Arcmin=np.degrees(np.arctan(R500Mpc/cosmoModel.angular_diameter_distance(z).value))*60.0
    
    return theta500Arcmin

#------------------------------------------------------------------------------------------------------------
def calcQ(theta500Arcmin, tck):
    """Returns Q, given theta500Arcmin, and a set of spline fit knots for (theta, Q).
    
    """
    
    #Q=np.poly1d(coeffs)(theta500Arcmin)
    Q=interpolate.splev(theta500Arcmin, tck)
    
    return Q

#------------------------------------------------------------------------------------------------------------
def calcFRel(z, M500, cosmoModel, obsFreqGHz = 148.0):
    """Calculates relativistic correction to SZ effect at specified frequency, given z, M500 in MSun.
       
    This assumes the Arnaud et al. (2005) M-T relation, and applies formulae of Itoh et al. (1998)
    
    As for H13, we return fRel = 1 + delta_SZE (see also Marriage et al. 2011)
    """
    
    # NOTE: we should define constants somewhere else...
    h=6.63e-34
    kB=1.38e-23
    sigmaT=6.6524586e-29
    me=9.11e-31
    e=1.6e-19
    c=3e8
    
    # Using Arnaud et al. (2005) M-T to get temperature
    A=3.84e14
    B=1.71
    #TkeV=5.*np.power(((cosmoModel.efunc(z)*M500)/A), 1/B)   # HMF/Astropy
    TkeV=5.*np.power(((cosmoModel.Ez(z)*M500)/A), 1/B)   # Colossus
    TKelvin=TkeV*((1000*e)/kB)

    # Itoh et al. (1998) eqns. 2.25 - 2.30
    thetae=(kB*TKelvin)/(me*c**2)
    X=(h*obsFreqGHz*1e9)/(kB*TCMB)
    Xtw=X*(np.cosh(X/2.)/np.sinh(X/2.))
    Stw=X/np.sinh(X/2.)

    Y0=-4+Xtw

    Y1=-10. + (47/2.)*Xtw - (42/5.)*Xtw**2 + (7/10.)*Xtw**3 + np.power(Stw, 2)*(-(21/5.) + (7/5.)*Xtw)

    Y2=-(15/2.) +  (1023/8.)*Xtw - (868/5.)*Xtw**2 + (329/5.)*Xtw**3 - (44/5.)*Xtw**4 + (11/30.)*Xtw**5 \
        + np.power(Stw, 2)*(-(434/5.) + (658/5.)*Xtw - (242/5.)*Xtw**2 + (143/30.)*Xtw**3) \
        + np.power(Stw, 4)*(-(44/5.) + (187/60.)*Xtw)

    Y3=(15/2.) + (2505/8.)*Xtw - (7098/5.)*Xtw**2 + (14253/10.)*Xtw**3 - (18594/35.)*Xtw**4 + (12059/140.)*Xtw**5 - (128/21.)*Xtw**6 + (16/105.)*Xtw**7 \
        + np.power(Stw, 2)*(-(7098/10.) + (14253/5.)*Xtw - (102267/35.)*Xtw**2 + (156767/140.)*Xtw**3 - (1216/7.)*Xtw**4 + (64/7.)*Xtw**5) \
        + np.power(Stw, 4)*(-(18594/35.) + (205003/280.)*Xtw - (1920/7.)*Xtw**2 + (1024/35.)*Xtw**3) \
        + np.power(Stw, 6)*(-(544/21.) + (992/105.)*Xtw)

    Y4=-(135/32.) + (30375/128.)*Xtw - (62391/10.)*Xtw**2 + (614727/40.)*Xtw**3 - (124389/10.)*Xtw**4 \
        + (355703/80.)*Xtw**5 - (16568/21.)*Xtw**6 + (7516/105.)*Xtw**7 - (22/7.)*Xtw**8 + (11/210.)*Xtw**9 \
        + np.power(Stw, 2)*(-(62391/20.) + (614727/20.)*Xtw - (1368279/20.)*Xtw**2 + (4624139/80.)*Xtw**3 - (157396/7.)*Xtw**4 \
        + (30064/7.)*Xtw**5 - (2717/7.)*Xtw**6 + (2761/210.)*Xtw**7) \
        + np.power(Stw, 4)*(-(124389/10.) + (6046951/160.)*Xtw - (248520/7.)*Xtw**2 + (481024/35.)*Xtw**3 - (15972/7.)*Xtw**4 + (18689/140.)*Xtw**5) \
        + np.power(Stw, 6)*(-(70414/21.) + (465992/105.)*Xtw - (11792/7.)*Xtw**2 + (19778/105.)*Xtw**3) \
        + np.power(Stw, 8)*(-(682/7.) + (7601/210.)*Xtw)

    deltaSZE=((X**3)/(np.exp(X)-1)) * ((thetae*X*np.exp(X))/(np.exp(X)-1)) * (Y0 + Y1*thetae + Y2*thetae**2 + Y3*thetae**3 + Y4*thetae**4)

    fRel=1+deltaSZE
    
    return fRel


#------------------------------------------------------------------------------------------------------------
def y0FromLogM500(log10M500, z, tckQFit, tenToA0 = 4.95e-5, B0 = 0.08, Mpivot = 3e14, sigma_int = 0.2,
                  cosmoModel = None, fRelWeightsDict = {148.0: 1.0}):
    """Predict y0~ given logM500 (in MSun) and redshift. Default scaling relation parameters are A10 (as in
    H13).
    
    Use cosmoModel (astropy.cosmology object) to change/specify cosmological parameters.
    
    fRelWeightsDict is used to account for the relativistic correction when y0~ has been constructed
    from multi-frequency maps. Weights should sum to 1.0; keys are observed frequency in GHz.
    
    Returns y0~, theta500Arcmin, Q
    
    """

    if type(Mpivot) == str:
        raise Exception("Mpivot is a string - check Mpivot in your .yml config file: use, e.g., 3.0e+14 (not 3e14 or 3e+14)")
        
    # Filtering/detection was performed with a fixed fiducial cosmology... so we don't need to recalculate Q
    # We just need to recalculate theta500Arcmin and E(z) only
    M500=np.power(10, log10M500)
    theta500Arcmin=calcTheta500Arcmin(z, M500, cosmoModel)
    Q=calcQ(theta500Arcmin, tckQFit)
    
    # Relativistic correction: now a little more complicated, to account for fact y0~ maps are weighted sum
    # of individual frequency maps, and relativistic correction size varies with frequency
    fRels=[]
    freqWeights=[]
    for obsFreqGHz in fRelWeightsDict.keys():
        fRels.append(calcFRel(z, M500, cosmoModel, obsFreqGHz = obsFreqGHz))
        freqWeights.append(fRelWeightsDict[obsFreqGHz])
    fRel=np.average(np.array(fRels), axis = 0, weights = freqWeights)
    
    # UPP relation according to H13
    # NOTE: m in H13 is M/Mpivot
    # NOTE: this goes negative for crazy masses where the Q polynomial fit goes -ve, so ignore those
    y0pred=tenToA0*np.power(cosmoModel.efunc(z), 2)*np.power(M500/Mpivot, 1+B0)*Q*fRel
    
    return y0pred, theta500Arcmin, Q

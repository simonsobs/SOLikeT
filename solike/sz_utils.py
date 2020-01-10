import numpy as np
from nemo import signals

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
        
        Ytilde, theta0, Qfilt = signals.y0FromLogM500(
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

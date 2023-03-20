import numpy as np
from scipy.special import factorial


def cash_c_logpdf(theory, data, usestirling=True, name="binned"):

    __, __, delN2Dcat, zcut = data

    obs = np.asarray(delN2Dcat, dtype=int)
    ln_fac = np.zeros_like(obs, dtype=float)
    zcut_arr = np.arange(zcut)

    if zcut > 0:
        theory = np.delete(theory, zcut_arr, 0)
        obs = np.delete(obs, zcut_arr, 0)
        ln_fac = np.delete(ln_fac, zcut_arr, 0)
        print("\r ::: Excluding first {} redshift bins in likelihood.".format(zcut))

        for i in range(theory.shape[0]):
            print('\r Number of clusters in redshift bin {}: {}.'.format(i, theory[i,:].sum()))
        print('------------')
        for i in range(theory.shape[1]):
            print('\r Number of clusters in SNR bin {}: {}.'.format(i, theory[:,i].sum()))
        print('------------')
        print('\r Total predicted N = {}'.format(theory.sum()))
        print('\r Total observed N = {}'.format(obs.sum()))

    if usestirling: # use Stirling's approximation for N > 10
        ln_fac[obs > 10] = 0.918939 + (obs[obs > 10] + 0.5) * np.log(obs[obs > 10]) - obs[obs > 10]
        ln_fac[obs <= 10] = np.log(factorial(obs[obs <= 10]))
    else: # direct compuation of factorial
        ln_fac[obs > 0] = np.log(factorial(obs[obs > 0]))
    ln_fac[obs == 0] = 0.

    loglike = obs * np.log(theory) - theory - ln_fac
    #loglike = obs * np.log(obs) - obs - ln_fac

    print("\r ::: 2D ln likelihood = ", np.nansum(loglike[np.isfinite(loglike)]))

    return np.nansum(loglike[np.isfinite(loglike)])



class CashCData:
    """Named multi-dimensional Cash-C distributed data
    """

    def __init__(self, name, N, usestirling=True):
        self.name = str(name)
        self.data = N
        self.usestirling = usestirling

    # def __len__(self):
    #     return len(self.data)

    def loglike(self, theory):
        return cash_c_logpdf(theory, self.data, name=self.name)

import numpy as np
from scipy.special import factorial
import math as m


def cash_c_logpdf(theory, data, usestirling=True, name = "Unbinned"):

    # ## This is how it needs to be!!!!
    # data = np.asarray(data, dtype=int)
    #
    # ln_fac = np.zeros_like(data, dtype=float)
    #
    # if usestirling: # use Stirling's approximation for N > 10
    #     ln_fac[data > 10] = 0.918939 + (data[data > 10] + 0.5) \
    #                                 * np.log(data[data > 10]) - data[data > 10]
    #     ln_fac[data <= 10] = np.log(factorial(data[data <= 10]))
    # else:
    #     ln_fac[data > 0] = np.log(factorial(data[data > 0]))
    # ln_fac[data == 0] = 0.
    #
    # loglike = data * np.log(theory) - theory - ln_fac

    ### Not well written, but for now ok:
    delN2D = theory
    zarr, qarr, delN2Dcat = data

    szcc = 0
    i = 0
    j = 0
    ii = 0

    for i in range(len(zarr)):
        for j in range(len(qarr)):
            if delN2D[i,j] != 0. :
                ln_fac = 0.
                if delN2Dcat[i,j] != 0. :
                    if delN2Dcat[i,j] > 10. : # Stirling approximation only for more than 10 elements
                        ln_fac = 0.918939 + (delN2Dcat[i,j] + 0.5) * np.log(delN2Dcat[i,j]) - delN2Dcat[i,j]
                    else: # direct compuation of factorial
                        ln_fac = np.log(m.factorial(int(delN2Dcat[i,j])))

                szcc += delN2Dcat[i,j] * np.log(delN2D[i,j]) - delN2D[i,j] - ln_fac

    print("\r ::: 2D ln likelihood = ", -szcc)

    loglike = szcc

    return np.nansum(loglike[np.isfinite(loglike)])


class CashCData:
    """Named multi-dimensional Cash-C distributed data
    """

    def __init__(self, name, N, usestirling=True):

        self.name = str(name)
        self.data = N
        self.usestirling = usestirling

    def __len__(self):
        return len(self.data)

    def loglike(self, theory):
        return cash_c_logpdf(theory, self.data, name = self.name)

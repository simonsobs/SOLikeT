import numpy as np
from scipy.special import factorial
import math as m


def cash_c_logpdf(theory, data, usestirling=True):

    data = np.asarray(data, dtype=int)

    ln_fac = np.zeros_like(data, dtype=float)

    if usestirling: # use Stirling's approximation for N > 10
        ln_fac[data > 10] = 0.918939 + (data[data > 10] + 0.5) \
                                    * np.log(data[data > 10]) - data[data > 10]
        ln_fac[data <= 10] = np.log(factorial(data[data <= 10]))
    else:
        ln_fac[data > 0] = np.log(factorial(data[data > 0]))
    ln_fac[data == 0] = 0.

    loglike = data * np.log(theory) - theory - ln_fac

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
        return cash_c_logpdf(theory, self.data)

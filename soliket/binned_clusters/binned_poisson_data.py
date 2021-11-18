import numpy as np
import math as m

class BinnedPoissonData:

    def __init__(self, delNcat, delN2Dcat):

        self.delNcat = delNcat
        self.delN2Dcat = delN2Dcat

    def loglike(self, theory):

        if theory.ndim == 1:

            delN = theory
            zarr, delNcat = self.delNcat

            szcc = 0
            i = 0
            for i in range(len(zarr)):
                if delN[i] != 0. :
                    ln_fac = 0.
                    if delNcat[i] != 0. :
                        if delNcat[i] > 10.: # Stirling approximation only for more than 10 elements
                            ln_fac = 0.918939 + (delNcat[i] + 0.5) * np.log(delNcat[i]) - delNcat[i]
                        else: # direct computation of factorial
                            ln_fac = np.log(m.factorial(int(delNcat[i])))
                    szcc += delNcat[i] * np.log(delN[i]) - delN[i] - ln_fac

            print("\r ln likelihood = ", -szcc)

        else:

            delN2D = theory
            zarr, qarr, delN2Dcat = self.delN2Dcat

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

        return szcc

import pandas as pd

from cobaya.likelihood import Likelihood

from .poisson_data import PoissonData
import scipy.interpolate
import numpy as np
# import multiprocessing
# from functools import partial
import time
import sys

class PoissonLikelihood(Likelihood):
    name = "Poisson"
    data_path = None
    columns = None

    def initialize(self):
        self.data = PoissonData(self.name, self.catalog, self.columns)
        return {}

    def get_requirements(self):
        return {}


    def _get_rate_fn(self, pk_intp,dn_dzdm_intp,**kwargs):
        """Returns a callable rate function that takes each of 'columns' as kwargs.
        """
        raise NotImplementedError

    def _get_n_expected(self, **kwargs):
        """Computes and returns the integral of the rate function
        """
        raise NotImplementedError

    def logp(self, **params_values):

        start0 = time.time()

        pk_intp = self.theory.get_Pk_interpolator(("delta_nonu", "delta_nonu"), nonlinear=False)
        dndlnm = self._get_dndlnm(self.zz, pk_intp, **params_values)
        dn_dzdm_intp = scipy.interpolate.interp2d(self.zz, self.lnmarr, np.log(dndlnm), kind='linear',
                                                  copy=True, bounds_error=False, fill_value=-np.inf)

        n_expected = self._get_n_expected(pk_intp, **params_values)


        # a_pool = multiprocessing.Pool()
        # rate_densities = a_pool.map(partial(Prob_per_cluster,
        #                                     # self = self,
        #                                     pk_intp = pk_intp,
        #                                     dn_dzdm_intp = dn_dzdm_intp,
        #                                     params_values = params_values
        #                                     ),
        #                                     range(ncat))
        # a_pool.close()
        # rate_densities = np.asarray(rate_densities)
        # exit(0)

        # vectorize implementation
        # apparently faster than parallel implementation

        start = time.time()

        Prob_per_cluster_vec = np.vectorize(Prob_per_cluster)
        rate_densities = Prob_per_cluster_vec(np.arange(self.N_cat),
                             self,
                             pk_intp,
                             dn_dzdm_intp,
                             params_values)

        rate_densities = np.asarray(rate_densities)
        # exit(0)

        elapsed = time.time() - start
        self.log.info("::: prob_per_cluster calculation took {:.3f} seconds.".format(elapsed))

        # start = time.time()


        elapsed = time.time() - start0
        self.log.info("::: total unbinned took {:.3f} seconds.".format(elapsed))

        return self.data.loglike(rate_densities, n_expected)


def Prob_per_cluster(cat_index,
                     self,
                     pk_intp,
                     dn_dzdm_intp,
                     params_values):

    tsz_signal, z, tsz_signal_err, tile_name = [self.catalog[c].values[cat_index] for c in self.columns]
    # self.log.info('computing prob per cluster for cluster: %.5e %.5e %.5e %s'%(z,tsz_signal,tsz_signal_err,tile_name))


    #if self.tiles_dwnsmpld is not None:
    tile_index = self.tiles_dwnsmpld[tile_name]
    #else:
    #    tile_index = None

    Pfunc_ind = self.Pfunc_per(
        tile_index,
        np.exp(self.lnmarr),
        z,
        tsz_signal * 1e-4,
        tsz_signal_err * 1e-4,
        params_values,
    )

    dn_dzdm = np.exp(dn_dzdm_intp(z, self.lnmarr))

    dn_dzdm = np.squeeze(dn_dzdm)

    ans = np.trapz(dn_dzdm * Pfunc_ind, dx=np.diff(self.lnmarr, axis=0), axis=0)

    if ans == 0:
        print("Pfunc_per_cluster is zero at", cat_index)
        ans = 0.00001

    return ans

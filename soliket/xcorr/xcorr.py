r""" Likelihood for cross-correlation of CMB lensing and galaxy clustering probes.
Based on the original xcorr code [1]_ used in Krolewski et al (2021) [2]_.

    References
    ----------
    .. [1] https://github.com/simonsobs/xcorr
    .. [2] Krolewski, Ferraro and White, 2021, arXiv:2105.03421

"""

from typing import Optional, Tuple
from cobaya.theory import Provider
import numpy as np
import sacc
from scipy.interpolate import InterpolatedUnivariateSpline as Spline

from soliket import utils
from soliket.gaussian import GaussianData, GaussianLikelihood

from .limber import do_limber


class XcorrLikelihood(GaussianLikelihood):
    """Cross-correlation Likelihood for CMB lensing and galaxy clustering probes.

    Accepts data files containing the two spectra from either text files or a sacc file.

    Parameters
    ----------

    datapath : str, optional
        sacc file containing the redshift distribtion, galaxy-galaxy and galaxy-kappa
        observed spectra. Default: soliket/tests/data/unwise_g-so_kappa.sim.sacc.fits
    k_tracer_name : str, optional
        sacc file tracer name for kappa. Default: ck_so
    gc_tracer_name : str, optional
        sacc file tracer name for galaxy clustering. Default: gc_unwise

    dndz_file : str, optional
        Text file containing the redshift distribution.
    auto_file : str, optional
        Text file containing the galaxy-galaxy observed spectra.
    cross_file : str, optional
        Text file containing the galaxy-kappa observed spectra.

    high_ell : int
        Maximum multipole to be computed for all spectra. Default: 600
    nz : int
        Resolution of redshift grid used for Limber computations. Default: 149
    Nchi : int
        Resolution of Chi grid used for lensing kernel computations. Default: 149
    Nchi_mag : int
        Resolution of Chi grid used for magnification kernel computations. Default: 149

    Pk_interp_kmax : float
        Maximum k  value for the Pk interpolator, units Mpc^-1. Default: 10.0

    b1 : float
        Linear galaxy bias value for the galaxy sample.
    s1 : float
        Magnification bias slope for the galaxy sample.

    """

    provider: Provider

    def initialize(self):
        name: str = "Xcorr"  # noqa F841
        self.log.info('Initialising.')

        if self.datapath is None:

            dndz_file: Optional[str]  # noqa F821
            auto_file: Optional[str]  # noqa F821
            cross_file: Optional[str]  # noqa F821

            self.dndz = np.loadtxt(self.dndz_file)

            self.x, self.y, self.dy = self._get_data()
            if self.covpath is None:
                self.log.info('No xcorr covariance specified. Using diag(dy^2).')
                self.cov = np.diag(self.dy**2)
            else:
                self.cov = self._get_cov()

        else:

            self.k_tracer_name: Optional[str]
            self.gc_tracer_name: Optional[str]
            # tracer_combinations: Optional[str] # TODO: implement with keep_selection

            self.sacc_data = self._get_sacc_data()

            self.x = self.sacc_data['x']
            self.y = self.sacc_data['y']
            self.cov = self.sacc_data['cov']
            self.dndz = self.sacc_data['dndz']
            self.ngal = self.sacc_data['ngal']

        # TODO is this resolution limit on zarray a CAMB problem?
        self.nz: Optional[int]
        assert self.nz <= 149, "CAMB limitations requires nz <= 149"
        self.zarray = np.linspace(self.dndz[:, 0].min(), self.dndz[:, 0].max(), self.nz)
        self.zbgdarray = np.concatenate([self.zarray, [1100]]) # TODO: unfix zstar
        self.Nchi: Optional[int]
        self.Nchi_mag: Optional[int]

        #self.use_zeff: Optional[bool]

        self.Pk_interp_kmax: Optional[float]

        self.high_ell: Optional[float]
        self.ell_range = np.linspace(1, self.high_ell, int(self.high_ell + 1))

        # TODO expose these defaults
        self.alpha_auto = 0.9981
        self.alpha_cross = 0.9977

        self.data = GaussianData(self.name, self.x, self.y, self.cov)


    def get_requirements(self) -> dict:
        return {
                'Cl': {'lmax': self.high_ell,
                        'pp': self.high_ell},
                "Pk_interpolator": {
                                    "z": self.zarray[:-1],
                                    "k_max": self.Pk_interp_kmax,
                                    #"extrap_kmax": 20.0,
                                    "nonlinear": False,
                                    "hubble_units": False,  # cobaya told me to
                                    "k_hunit": False,  # cobaya told me to
                                    "vars_pairs": [["delta_nonu", "delta_nonu"]],
                                    },
                "Hubble": {"z": self.zarray},
                "angular_diameter_distance": {"z": self.zbgdarray},
                "comoving_radial_distance": {"z": self.zbgdarray},
                'H0': None,
                'ombh2': None,
                'omch2': None,
                'omk': None,
                'omegam': None,
                'zstar': None,
                'As': None,
                'ns': None
                }

    def _bin(
        self, theory_cl: np.ndarray, lmin: np.ndarray, lmax: np.ndarray
    ) -> np.ndarray:
        binned_theory_cl: np.ndarray = np.zeros_like(lmin)
        for i in range(len(lmin)):
            binned_theory_cl[i] = np.mean(theory_cl[(self.ell_range >= lmin[i])
                                                     & (self.ell_range < lmax[i])])
        return binned_theory_cl

    def _get_sacc_data(self, **params_values: dict) -> dict:
        data_sacc = sacc.Sacc.load_fits(self.datapath)

        # TODO: would be better to use keep_selection
        data_sacc.remove_selection(tracers=(self.k_tracer_name, self.k_tracer_name))

        ell_auto, cl_auto = data_sacc.get_ell_cl('cl_00',
                                                 self.gc_tracer_name,
                                                 self.gc_tracer_name)
        ell_cross, cl_cross = data_sacc.get_ell_cl('cl_00',
                                                   self.gc_tracer_name,
                                                   self.k_tracer_name) # TODO: check order
        cov = data_sacc.covariance.covmat

        x = np.concatenate([ell_auto, ell_cross])
        y = np.concatenate([cl_auto, cl_cross])

        dndz = np.column_stack([data_sacc.tracers[self.gc_tracer_name].z,
                                data_sacc.tracers[self.gc_tracer_name].nz])
        ngal = data_sacc.tracers[self.gc_tracer_name].metadata['ngal']

        data = {'x': x,
                'y': y,
                'cov': cov,
                'dndz': dndz,
                'ngal': ngal}

        return data


    def _get_data(
        self, **params_values: dict
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

        data_auto = np.loadtxt(self.auto_file)
        data_cross = np.loadtxt(self.cross_file)

        # Get data
        self.ell_auto = data_auto[0]
        cl_auto = data_auto[1]
        cl_auto_err = data_auto[2]

        self.ell_cross = data_cross[0]
        cl_cross = data_cross[1]
        cl_cross_err = data_cross[2]

        x = np.concatenate([self.ell_auto, self.ell_cross])
        y = np.concatenate([cl_auto, cl_cross])
        dy = np.concatenate([cl_auto_err, cl_cross_err])

        return x, y, dy

    def _setup_chi(self) -> dict:

        chival = self.provider.get_comoving_radial_distance(self.zarray)
        zatchi = Spline(chival, self.zarray)
        chiatz = Spline(self.zarray, chival)

        chimin = np.min(chival) + 1.e-5
        chimax = np.max(chival)
        chival = np.linspace(chimin, chimax, self.Nchi)
        zval = zatchi(chival)
        chistar = \
            self.provider.get_comoving_radial_distance(self.provider.get_param('zstar'))
        chivalp = \
            np.array(list(map(lambda x: np.linspace(x, chistar, self.Nchi_mag), chival)))
        chivalp = chivalp.transpose()[0]
        zvalp = zatchi(chivalp)

        chi_result = {'zatchi': zatchi,
                      'chiatz': chiatz,
                      'chival': chival,
                      'zval': zval,
                      'chivalp': chivalp,
                      'zvalp': zvalp}

        return chi_result

    def _get_theory(self, **params_values: dict) -> np.ndarray:

        setup_chi_out = self._setup_chi()

        Pk_interpolator = self.provider.get_Pk_interpolator(("delta_nonu", "delta_nonu"),
                                                          extrap_kmax=1.e8,
                                                          nonlinear=False).P

        cl_gg, cl_kappag = do_limber(self.ell_range,
                                     self.provider,
                                     self.dndz,
                                     self.dndz,
                                     params_values['s1'],
                                     params_values['s1'],
                                     Pk_interpolator,
                                     params_values['b1'],
                                     params_values['b1'],
                                     self.alpha_auto,
                                     self.alpha_cross,
                                     setup_chi_out,
                                     Nchi=self.Nchi,
                                     #use_zeff=self.use_zeff,
                                     dndz1_mag=self.dndz,
                                     dndz2_mag=self.dndz)

        # TODO: this is not the correct binning,
        # but there needs to be a consistent way to specify it
        bin_edges = np.linspace(20, self.high_ell, self.data.x.shape[0] // 2 + 1)

        ell_gg, clobs_gg = utils.binner(self.ell_range, cl_gg, bin_edges)
        ell_kappag, clobs_kappag = utils.binner(self.ell_range, cl_kappag, bin_edges)
        #ell_kappakappa, clobs_kappakappa = utils.binner(self.ell_range, cl_kappakappa, bin_edges) # noqa E501

        return np.concatenate([clobs_gg, clobs_kappag])

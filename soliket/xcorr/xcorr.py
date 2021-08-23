import numpy as np
import sacc
import pdb

from scipy.interpolate import InterpolatedUnivariateSpline as Spline

from ..gaussian import GaussianData, GaussianLikelihood
from .. import utils
from .limber import do_limber

class XcorrLikelihood(GaussianLikelihood):

    def initialize(self):
        name: str = "Xcorr"
        self.log.info('Initialising.')

        if self.datapath is None:

            dndz_file: Optional[str]
            auto_file: Optional[str]
            cross_file: Optional[str]

            self.dndz = np.loadtxt(self.dndz_file)

            x, y, dy = self._get_data()
            if self.covpath is None:
                self.log.info('No xcorr covariance specified. Using diag(dy^2).')
                cov = np.diag(dy**2)
            else:
                cov = self._get_cov()

        else:
            
            k_tracer_name: Optional[str]
            gc_tracer_name: Optional[str]
            # tracer_combinations: Optional[str] # TODO: implement this along with keep_selection
            
            sacc_data = self._get_sacc_data()

            x = sacc_data['x']
            y = sacc_data['y']
            cov = sacc_data['cov']
            self.dndz = sacc_data['dndz']
            self.ngal = sacc_data['ngal']

        # TODO is this resolution limit on zarray a CAMB problem?
        self.zarray = np.linspace(self.dndz[:,0].min(), self.dndz[:,0].max(), 149)
        self.zbgdarray = np.concatenate([self.zarray, [1100]]) # TODO: unfix zstar
        self.Nchi = 149
        self.Nchi_mag = 149

       # TODO expose these defaults
        self.high_ell = 600
        self.ell_range = np.linspace(1, self.high_ell, int(self.high_ell+1))

        # TODO expose these defaults
        self.alpha_auto = 0.9981
        self.alpha_cross = 0.9977

        self.data = GaussianData(self.name, x, y, cov)


    def get_requirements(self):
        return {
                'Cl': {'lmax': self.high_ell,
                        'pp': self.high_ell},
                "Pk_interpolator": {
                    "z": self.zarray[:-1],
                    "k_max": 10.0, #TODO fix this
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

    def _bin(self, theory_cl, lmin, lmax):
        binned_theory_cl = np.zeros_like(lmin)
        for i in range(len(lmin)):
            binned_theory_cl[i] = np.mean(theory_cl[(self.ell_range >= lmin[i]) & (self.ell_range < lmax[i])])
        return binned_theory_cl

    def _get_sacc_data(self, **params_values):

        data_sacc = sacc.Sacc.load_fits(self.datapath)

        data_sacc.remove_selection(tracers=(self.k_tracer_name, self.k_tracer_name)) # TODO: would be better to use keep_selection

        ell_auto, cl_auto = data_sacc.get_ell_cl('cl_00', self.gc_tracer_name, self.gc_tracer_name)
        ell_cross, cl_cross = data_sacc.get_ell_cl('cl_00', self.gc_tracer_name, self.k_tracer_name) #TODO: be robust to ordering
        cov = data_sacc.covariance.covmat

        x = np.concatenate([ell_auto, ell_cross])
        y = np.concatenate([cl_auto, cl_cross])

        dndz = np.column_stack([data_sacc.tracers[self.gc_tracer_name].z, data_sacc.tracers[self.gc_tracer_name].nz])
        ngal = data_sacc.tracers[self.gc_tracer_name].metadata['ngal']

        data = {'x' : x,
                'y' : y,
                'cov': cov,
                'dndz': dndz,
                'ngal': ngal}

        return data


    def _get_data(self, **params_values):

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

    def _setup_chi(self):

        chival = self.provider.get_comoving_radial_distance(self.zarray)
        zatchi = Spline(chival, self.zarray)
        chiatz = Spline(self.zarray, chival)

        chimin = np.min(chival) + 1.e-5
        chimax = np.max(chival)
        chival   = np.linspace(chimin,chimax,self.Nchi)
        zval = zatchi(chival)
        chistar = self.provider.get_comoving_radial_distance(self.provider.get_param('zstar'))
        chivalp = np.array(list(map(lambda x: np.linspace(x,chistar,self.Nchi_mag),chival))).transpose()[0]
        zvalp = zatchi(chivalp)

        return zatchi, chiatz, chival, zval, chivalp, zvalp

    def _get_theory(self, **params_values):

        setup_chi_out = self._setup_chi()

        Pk_interpolator = self.theory.get_Pk_interpolator(("delta_nonu", "delta_nonu"), extrap_kmax=1.e8, nonlinear=False).P

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
                                       Nchi=self.Nchi,
                                       autoCMB=False,
                                       use_zeff=False,
                                       dndz1_mag=self.dndz,
                                       dndz2_mag=self.dndz,
                                       setup_chi_flag=True,
                                       setup_chi_out=setup_chi_out)

        # TODO: this is not the correct binning, but there needs to be a consistent way to specify it
        bin_edges = np.linspace(20, self.high_ell, self.data.x.shape[0]//2 + 1)

        ell_gg, clobs_gg = utils.binner(self.ell_range, cl_gg, bin_edges)
        ell_kappag, clobs_kappag = utils.binner(self.ell_range, cl_kappag, bin_edges)
        #ell_kappakappa, clobs_kappakappa = utils.binner(self.ell_range, cl_kappakappa, bin_edges)

        return np.concatenate([clobs_gg, clobs_kappag])

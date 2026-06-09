r"""Likelihood for cross-correlation of CMB lensing and galaxy clustering probes.
Based on the original xcorr code [1]_ used in Krolewski et al (2021) [2]_.

    References
    ----------
    .. [1] https://github.com/simonsobs/xcorr
    .. [2] Krolewski, Ferraro and White, 2021, arXiv:2105.03421

"""

import numpy as np
import sacc
from cobaya.theory import Provider
from scipy.interpolate import InterpolatedUnivariateSpline as Spline

from soliket.gaussian import GaussianData, GaussianLikelihood

from .limber import do_limber


class XcorrLikelihood(GaussianLikelihood):
    """Cross-correlation Likelihood for CMB lensing and galaxy clustering probes.

    Accepts data files containing the two spectra from either text files or a sacc file.

    Parameters
    ----------

    datapath : str, optional
        sacc file containing the redshift distribtion, galaxy-galaxy and galaxy-kappa
        observed spectra. Default: tests/data/unwise_g-so_kappa.sim.sacc.fits
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

    name = "Xcorr"
    use_spectra: str | tuple[str, str] | list[tuple[str, str]] | None
    datapath: str | None
    k_tracer_name: str | None
    gc_tracer_name: str | None
    high_ell: int | None
    nz: int | None
    Nchi: int | None
    Nchi_mag: int | None
    Pk_interp_kmax: int | float | None
    b1: int | float
    s1: int | float

    provider: Provider

    _allowable_tracers = ("cmb_convergence", "galaxy_density")

    def initialize(self):

        super().initialize()
        _, self.binning_matrix = self.get_binning(self.tracer_comb)
        assert self.gc_tracer_name in self.sacc_data.tracers, (
            f"Galaxy clustering tracer {self.gc_tracer_name} not found in sacc data!"
        )
        assert self.k_tracer_name in self.sacc_data.tracers, (
            f"CMB lensing tracer {self.k_tracer_name} not found in sacc data!"
        )

        self.dndz = self._get_dndz()
        self.ngal = self._get_ngal()

        # TODO is this resolution limit on zarray a CAMB problem?
        assert self.nz <= 149, "CAMB limitations requires nz <= 149"
        self.zarray = np.linspace(self.dndz[:, 0].min(), self.dndz[:, 0].max(), self.nz)
        self.zbgdarray = np.concatenate([self.zarray, [1100]])  # TODO: unfix zstar

        # self.use_zeff: bool | None = None

        self.ell_range = np.linspace(1, self.high_ell, int(self.high_ell + 1))

        self.data = GaussianData(self.name, self.x, self.y, self.cov)

    def _get_dndz(self) -> np.ndarray:
        tracers = self.sacc_data.tracers
        tracer: sacc.tracers.NZTracer = tracers[self.gc_tracer_name]

        dndz = tracer.nz
        z = tracer.z
        assert len(z) == len(dndz), "dndz and z have different lengths!"
        return np.array([z, dndz]).T

    def _get_ngal(self) -> float:
        tracers = self.sacc_data.tracers
        tracer: sacc.tracers.NZTracer = tracers[self.gc_tracer_name]
        if "ngal" not in tracer.metadata:
            raise ValueError(
                f"Tracer {self.gc_tracer_name} does not have ngal in metadata!"
            )
        ngal = tracer.metadata["ngal"]
        return ngal

    def get_requirements(self):
        return {
            "Cl": {"lmax": self.high_ell, "pp": self.high_ell},
            "Pk_interpolator": {
                "z": self.zarray[:-1],
                "k_max": self.Pk_interp_kmax,
                # "extrap_kmax": 20.0,
                "nonlinear": False,
                "hubble_units": False,  # cobaya told me to
                "k_hunit": False,  # cobaya told me to
                "vars_pairs": [["delta_nonu", "delta_nonu"]],
            },
            "Hubble": {"z": self.zarray},
            "angular_diameter_distance": {"z": self.zbgdarray},
            "comoving_radial_distance": {"z": self.zbgdarray},
            "H0": None,
            "ombh2": None,
            "omch2": None,
            "omk": None,
            "omegam": None,
            "zstar": None,
            "As": None,
            "ns": None,
        }

    def _setup_chi(self) -> dict:
        chival = self.provider.get_comoving_radial_distance(self.zarray)
        zatchi = Spline(chival, self.zarray)
        chiatz = Spline(self.zarray, chival)

        chimin = np.min(chival) + 1.0e-5
        chimax = np.max(chival)
        chival = np.linspace(chimin, chimax, self.Nchi)
        zval = zatchi(chival)
        chistar = self.provider.get_comoving_radial_distance(
            self.provider.get_param("zstar")
        )
        chivalp = np.array(
            list(map(lambda x: np.linspace(x, chistar, self.Nchi_mag), chival))
        )
        chivalp = chivalp.transpose()[0]
        zvalp = zatchi(chivalp)

        chi_result = {
            "zatchi": zatchi,
            "chiatz": chiatz,
            "chival": chival,
            "zval": zval,
            "chivalp": chivalp,
            "zvalp": zvalp,
        }

        return chi_result

    def _get_theory(self, **params_values) -> np.ndarray:
        setup_chi_out = self._setup_chi()

        Pk_interpolator = self.provider.get_Pk_interpolator(
            ("delta_nonu", "delta_nonu"), extrap_kmax=1.0e8, nonlinear=False
        ).P

        cl_gg, cl_kappag = do_limber(
            self.ell_range,
            self.provider,
            self.dndz,
            self.dndz,
            params_values["s1"],
            params_values["s1"],
            Pk_interpolator,
            params_values["b1"],
            params_values["b1"],
            params_values["alpha_auto"],
            params_values["alpha_cross"],
            setup_chi_out,
            Nchi=self.Nchi,
            # use_zeff=self.use_zeff,
            dndz1_mag=self.dndz,
            dndz2_mag=self.dndz,
        )

        clobs_gg = self.binning_matrix.dot(cl_gg)
        clobs_kappag = self.binning_matrix.dot(cl_kappag)

        return np.concatenate([clobs_gg, clobs_kappag])

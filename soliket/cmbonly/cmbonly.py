import os
import numpy as np
from typing import Optional
from cobaya.likelihood import Likelihood


class CMBonly(Likelihood):
    """
    Likelihood for SO foreground-marginalized (cmb-only).

    Author: Hidde T. Jense
    """
    file_base_name: str = "so_cmb"

    input_file: str
    data_folder: str = "SOCMBonly"
    ell_cuts: dict = {
        "TT": [0, 7000],
        "TE": [0, 7000],
        "EE": [0, 7000]
    }
    lmax_theory: Optional[int] = None

    def initialize(self):
        if self.packages_path is None:
            self.data_folder = os.path.join(
                os.path.dirname(os.path.realpath(__file__)),
                "data")
        else:
            from cobaya.conventions import data_path
            self.data_folder = os.path.join(
                self.packages_path,
                data_path,
                self.data_folder)

        import sacc
        input_filename = os.path.join(self.data_folder, self.input_file)
        self.log.debug(f"Searching for data in {input_filename}.")

        input_file = sacc.Sacc.load_fits(input_filename)

        self.log.info(f"Loading data from {input_filename}.")

        pol_dt = {"t": "0", "e": "e", "b": "b"}

        self.ell_cuts = self.ell_cuts or {}
        self.lmax_theory = self.lmax_theory or -1

        self.spec_meta = []
        self.cull = []
        idx_max = 0

        for pol in ["TT", "TE", "EE"]:
            p1, p2 = pol.lower()
            t1, t2 = pol_dt[p1], pol_dt[p2]
            dt = f"cl_{t1}{t2}"

            tracers = input_file.get_tracer_combinations(dt)

            for tr1, tr2 in tracers:
                lmin, lmax = self.ell_cuts.get(pol, (-np.inf, np.inf))
                ls, mu, ind = input_file.get_ell_cl(dt, tr1, tr2,
                                                    return_ind=True)
                mask = np.logical_and(ls >= lmin, ls <= lmax)

                if not np.all(mask):
                    self.log.debug(
                        f"Cutting {pol} data to the range [{lmin}-{lmax}]."
                    )
                    self.cull.append(ind[~mask])

                window = input_file.get_bandpower_windows(ind[mask])

                self.spec_meta.append({
                    "data_type": dt,
                    "tracer1": tr1,
                    "tracer2": tr2,
                    "pol": pol.lower(),
                    "ell": ls[mask],
                    "spec": mu[mask],
                    "idx": ind[mask],
                    "window": window
                })

                idx_max = max(idx_max, max(ind))
                self.lmax_theory = max(self.lmax_theory,
                                       int(window.values.max())+1)

        self.data_vec = np.zeros((idx_max+1,))
        for m in self.spec_meta:
            self.data_vec[m["idx"]] = m["spec"]

        self.covmat = input_file.covariance.covmat
        for culls in self.cull:
            self.covmat[culls, :] = 0.0
            self.covmat[:, culls] = 0.0
            self.covmat[culls, culls] = 1e10

        self.inv_cov = np.linalg.inv(self.covmat)
        self.logp_const = np.log(2.0 * np.pi) * -0.5 * len(self.data_vec)
        self.logp_const -= 0.5 * np.linalg.slogdet(self.covmat)[1]

        self.log.debug(f"log(P) = {self.logp_const}")
        self.log.debug(f"len(data vec) = {len(self.data_vec)}")

    def get_requirements(self):
        return dict(Cl={
            k: self.lmax_theory+1 for k in ["TT", "TE", "EE"]
        })

    def chi_square(self, cl):
        ps_vec = np.zeros_like(self.data_vec)

        for m in self.spec_meta:
            idx = m["idx"]
            win = m["window"].weight.T
            ls = m["window"].values
            pol = m["pol"]
            dat = cl[pol][ls]

            ps_vec[idx] = win @ dat

        delta = self.data_vec - ps_vec

        chisquare = delta @ self.inv_cov @ delta
        self.log.debug(f"Chisqr = {chisquare:.3f}")
        return chisquare

    def loglike(self, cl):
        logp = -0.5 * self.chi_square(cl)
        return self.logp_const + logp

    def logp(self, **param_values):
        cl = self.provider.get_Cl(ell_factor=True)
        return self.loglike(cl)

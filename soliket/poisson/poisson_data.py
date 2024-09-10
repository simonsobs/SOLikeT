from typing import List, Optional, Callable, Dict
import pandas as pd
import numpy as np


class PoissonData:
    def __init__(
        self,
        name: str,
        catalog: pd.DataFrame,
        columns: List[str],
        samples: Optional[Dict[str, np.ndarray]] = None,
    ):
        self.name = str(name)

        self.catalog = pd.DataFrame(catalog)[columns]
        self.columns = columns

        if samples is not None:
            for c in columns:
                if c not in samples:
                    raise ValueError(f"If providing samples, must have samples \
                                     for all columns: {columns}")

            if "prior" not in samples:
                raise ValueError('Must provide value of interim prior \
                                  for all samples, under "prior" key!')

            assert all(
                samples[k].shape == samples["prior"].shape for k in samples
            ), "Samples all need same shape!"
            self.N_k = samples["prior"].shape[1]
            self._len = samples["prior"].shape[0]

        else:
            self._len = len(self.catalog)

        self.samples = samples

    def __len__(self) -> int:
        return self._len

    def loglike(
        self,
        rate_fn: Callable,
        n_expected: float,
        broadcastable: bool = False,
    ) -> float:
        """Computes log-likelihood of data under poisson process model

        rate_fn returns the *observed rate* as a function of self.columns
        (must be able to take all of self.columns as keywords, and be broadcastable
        (though could make this an option))

        n_expected is predicted total number
        """
        # Simple case; no uncertainties
        if self.samples is None:
            if broadcastable:
                rate_densities = rate_fn(**{c: self.catalog[c].values for
                                            c in self.columns})
            else:
                rate_densities = np.array(
                    [
                        rate_fn(**{c: self.catalog[c].values[i] for c in self.columns})
                        for i in range(len(self))
                    ]
                )

            return -n_expected + sum(np.log(rate_densities))

        else:
            # Eqn (11) of DFM, Hogg & Morton (https://arxiv.org/pdf/1406.3020.pdf)
            summand = rate_fn(**{c: self.samples[c] for
                                 c in self.columns}) / self.samples["prior"]
            l_k = 1 / self.N_k * summand.sum(axis=1)
            assert l_k.shape == (self._len,)
            return -n_expected + sum(np.log(l_k))

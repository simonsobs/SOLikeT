import numpy as np
import pandas as pd
from functools import partial

from soliket.poisson_data import PoissonData

x_min = 0
x_max = 10


def rate_density(x, a):
    """simple linear rate density
    """
    return a * x


def n_expected(a):
    return 0.5 * a * (x_max ** 2 - x_min ** 2)  # integral(rate_density, x_min, x_max)


def generate_data(a, with_samples=False, unc=0.3, Nk=64):
    # Generate total number
    n = np.random.poisson(n_expected(a))

    # Generate x values according to rate density (normalized as PDF)
    u = np.random.random(n)

    # From inverting CDF of above normalized density
    x = np.sqrt(u * (x_max ** 2 - x_min ** 2) + x_min ** 2)

    if not with_samples:
        return x
    else:
        return x[:, None] * (1 + np.random.normal(0, unc, size=(n, Nk)))


def test_poisson_experiment(a_true=3, N=100, with_samples=False, Nk=64):
    a_maxlikes = []
    for i in range(N):
        observations = generate_data(a_true, with_samples=with_samples, Nk=Nk)
        if not with_samples:
            catalog = pd.DataFrame({"x": observations})
            data = PoissonData("toy", catalog, ["x"])
        else:
            catalog = pd.DataFrame({"x": observations.mean(axis=1)})
            samples = {"x": observations, "prior": np.ones(observations.shape)}
            data = PoissonData("toy_samples", catalog, ["x"], samples=samples)

        a_grid = np.arange(0.1, 10, 0.1)
        lnl = [data.loglike(partial(rate_density, a=a), n_expected(a)) for a in a_grid]
        a_maxlike = a_grid[np.argmax(lnl)]

        a_maxlikes.append(a_maxlike)

    assert abs(np.mean(a_maxlikes) - a_true) < 0.1

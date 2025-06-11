import importlib

import numpy as np
from cobaya.theory import Theory

from soliket.cash import CashCData


class cash_theory_calculator(Theory):
    def calculate(self, state, want_derived=False, **params_values_dict):
        state["cash_theory"] = np.arange(params_values_dict["param_test_cash"])

    def get_cash_theory(self):
        return self.current_state["cash_theory"]


def toy_data():
    x = np.arange(20)
    y = np.arange(20)

    xx, yy = np.meshgrid(x, y)

    return x, y, xx, yy


def test_cash_import():
    _ = importlib.import_module("soliket.cash").CashCLikelihood


def test_cash_read_data(request):
    import os

    from soliket.cash import CashCLikelihood

    cash_data_path = os.path.join(
        request.config.rootdir, "soliket/tests/data/cash_data.txt"
    )

    cash_lkl = CashCLikelihood({"datapath": cash_data_path})
    cash_data = cash_lkl._get_data()
    assert np.allclose(cash_data[1], np.arange(20))


def test_cash_logp(request):
    import os

    from soliket.cash import CashCLikelihood

    params = {"cash_test_logp": 20}
    cash_data_path = os.path.join(
        request.config.rootdir, "soliket/tests/data/cash_data.txt"
    )

    cash_lkl = CashCLikelihood({"datapath": cash_data_path})
    cash_logp = cash_lkl.logp(**params)
    assert np.allclose(cash_logp, -37.3710640070228)


def test_cash():
    data1d, theory1d, data2d, theory2d = toy_data()

    cashdata1d = CashCData("toy 1d", data1d)
    cashdata2d = CashCData("toy 2d", data2d)

    assert np.allclose(cashdata1d.loglike(theory1d), -37.3710640070228)
    assert np.allclose(cashdata2d.loglike(theory2d), -2349.5353718742294)

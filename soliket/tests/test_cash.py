import numpy as np

from soliket.cash import CashCData


def toy_data():
    x = np.arange(20)
    y = np.arange(20)

    xx, yy = np.meshgrid(x, y)

    return x, y, xx, yy

def test_cash_import():
    from soliket.cash import CashCLikelihood

def test_cash_read_data():
    import os
    from soliket.cash import CashCLikelihood
    from cobaya.model import get_model

    soliket_dir = os.getcwd()
    cash_data_path = soliket_dir + "/soliket/tests/data/cash_data.txt"

    info = {"likelihood": { "soliket.cash.CashCLikelihood": {"datapath": cash_data_path}}}
    info["params"] = {"param_test_cash": 20}

    model = get_model(info)

    #theory = np.arange(20)

    #assert np.allclose(CashCLikelihood.data.loglike(theory), -37.3710640070228)

def test_cash():

    data1d, theory1d, data2d, theory2d = toy_data()

    cashdata1d = CashCData("toy 1d", data1d)
    cashdata2d = CashCData("toy 2d", data2d)

    assert np.allclose(cashdata1d.loglike(theory1d), -37.3710640070228)
    assert np.allclose(cashdata2d.loglike(theory2d), -2349.5353718742294)

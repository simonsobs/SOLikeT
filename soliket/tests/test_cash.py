import numpy as np

from soliket.cash import CashCData


def toy_data():
    x = np.arange(20)
    y = np.arange(20)

    xx, yy = np.meshgrid(x, y)

    return x, y, xx, yy


def test_cash():

    data1d, theory1d, data2d, theory2d = toy_data()

    cashdata1d = CashCData("toy 1d", data1d)
    cashdata2d = CashCData("toy 2d", data2d)

    assert np.allclose(cashdata1d.loglike(theory1d), -37.3710640070228)
    assert np.allclose(cashdata2d.loglike(theory2d), -2349.5353718742294)

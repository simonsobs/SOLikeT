import numpy as np

from soliket.utils import binner


def naive_binner(bmin, bmax, x, tobin):

    binned = list()
    bcent = list()
    # All but the last bins are open to the right
    for bm, bmx in zip(bmin[:-1], bmax[:-1]):
        bcent.append(0.5 * (bmx + bm))
        binned.append(np.mean(tobin[np.where((x >= bm) & (x < bmx))[0]]))
    # The last bin is closed to the right
    bcent.append(0.5 * (bmax[-1] + bmin[-1]))
    binned.append(np.mean(tobin[np.where((x >= bmin[-1]) & (x <= bmax[-1]))[0]]))

    return (np.array(bcent), np.array(binned))


def test_binning():

    #bmin = np.arange(10, step=3)
    #bmax = np.array([2, 5, 8, 12])
    binedge = np.arange(13, step=3)
    bmin = binedge[:-1]
    bmax = binedge[1:]
    ell = np.arange(13)
    cell = np.arange(13)

    centers_test, values_test = naive_binner(bmin, bmax, ell, cell)

    bincent, binval = binner(ell, cell, binedge)

    assert np.allclose(bincent, centers_test)
    assert np.allclose(binval, values_test)

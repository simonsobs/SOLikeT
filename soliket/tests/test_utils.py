import numpy as np

from soliket.utils import binner


def naive_binner(bmin, bmax, x, tobin):

	binned = list()
	bcent = list()
	for bm, bmx in zip(bmin, bmax):
		bcent.append(0.5 * (bmx + bm))
		binned.append(np.mean(tobin[np.where((x >= bm) & (x <= bmx))[0]]))

	return (np.array(bcent), np.array(binned))


def test_binning():

	bmin = np.arange(10, step=3)
	bmax = np.array([2, 5, 8, 12])
	binedge = np.arange(13, step=3)
	ell = np.arange(13)
	cell = np.arange(13)

	centers_test, values_test = naive_binner(bmin, bmax, ell, cell)

	bincent, binval = binner(ell, cell, binedge)

	#assert np.allclose(bincent, centers_test)
	assert np.allclose(binval, values_test)

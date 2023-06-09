import numpy as np

from soliket.utils import binner


def test_binning():

	values_test = np.array([1.,  4.,  7., 10.5])
	centers_test = np.array([1.5, 4.5, 7.5, 10.5])

	bincent, binval = binner(np.arange(13), np.arange(13), np.arange(13, step=3))

	assert np.allclose(bincent, centers_test)
	assert np.allclose(binval, values_test)

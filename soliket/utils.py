r"""
.. module:: utils

:Synopsis: Compilation of some useful classes and functions for use in SOLikeT.

"""

from typing import Optional, Tuple, Dict, Union, get_args, get_origin
from importlib import import_module

import numpy as np
from cobaya.likelihood import Likelihood
from cobaya.likelihoods.one import one
from scipy.stats import binned_statistic as binnedstat


def binner(
    ls: np.ndarray, cls: np.ndarray, bin_edges: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    r"""
    Simple function intended for binning :math:`\ell`-by-:math:`\ell` data into
    band powers with a top hat window function.

    Note that the centers are computed as :math:`0.5({\rm LHE}+{\rm RHE})`,
    where :math:`{\rm LHE}` and :math:`{\rm RHE}` are the bin edges.
    While this is ok for plotting purposes, the user may need
    to recompute the bin center in case of integer ``ls``
    if the correct baricenter is needed.

    :param ls: Axis along which to bin
    :param cls: Values to be binned
    :param bin_edges: The edges of the bins. Note that all but the last bin
                      are open to the right. The last bin is closed.

    :return: The centers of the bins and the average of ``cls`` within the bins.
    """
    x = ls.copy()
    y = cls.copy()
    cents = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    bin_edges_min = bin_edges.min()
    bin_edges_max = bin_edges.max()
    y[x < bin_edges_min] = 0
    y[x > bin_edges_max] = 0
    bin_means = binnedstat(x, y, bins=bin_edges, statistic=np.nanmean)[0]
    return cents, bin_means


def get_likelihood(name: str, options: Optional[dict] = None) -> Likelihood:
    parts = name.split(".")
    module = import_module(".".join(parts[:-1]))
    t = getattr(module, parts[-1])
    if not issubclass(t, Likelihood):
        raise ValueError(f"{name} is not a Likelihood!")
    if options is None:
        options = {}
    return t(options)


class OneWithCls(one):
    r"""
    Extension of
    `cobaya.likelihoods.one
    <https://cobaya.readthedocs.io/en/latest/likelihood_one.html>`_
    which creates a dummy :math:`C_\ell` requirements dictionary with an
    :math:`\ell_{\rm max}` of 1000 to force computation of ``pp``, ``tt``, ``te``, ``ee``
    and ``bb`` :math:`C_\ell` s.
    """
    lmax = 10000

    def get_requirements(self) -> Dict[str, Dict[str, int]]:
        return {"Cl": {"pp": self.lmax,
                       "tt": self.lmax,
                       "te": self.lmax,
                       "ee": self.lmax,
                       "bb": self.lmax, }}


def check_collection_type(value, expected_type, name):
    origin_type = get_origin(expected_type)
    if origin_type is list:
        if isinstance(value, list):
            item_type = get_args(expected_type)[0]
            for item in value:
                check_type(item, item_type, f"{name} item")
            return True
    elif origin_type is dict:
        if isinstance(value, dict):
            key_type, val_type = get_args(expected_type)
            for key, val in value.items():
                check_type(key, key_type, f"{name} key")
                check_type(val, val_type, f"{name} value")
            return True
    return False


def check_type(value, expected_types, name: str) -> None:
    if not isinstance(expected_types, tuple):
        expected_types = (expected_types,)

    if value is None:
        return

    for expected_type in expected_types:
        origin_type = get_origin(expected_type)
        if origin_type is None:
            if isinstance(value, expected_type):
                return
        elif check_collection_type(value, expected_type, name):
            return

    msg = f"{name} must be of type {expected_types}, not {type(value)}"
    raise TypeError(msg)


def check_yaml_types(
    instance, attributes: Dict[str, Union[type, Tuple[type, ...]]]
) -> None:
    for attr_name, expected_types in attributes.items():
        value = getattr(instance, attr_name, None)
        if value is None and isinstance(instance, dict):
            value = instance.get(attr_name, None)
        check_type(value, expected_types, attr_name)

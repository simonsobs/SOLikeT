import numpy as np
from scipy.stats import binned_statistic as binnedstat

def binner(ls,cls,bin_edges):
    x = ls.copy()
    y = cls.copy()
    numbins = len(bin_edges)-1
    cents = (bin_edges[:-1]+bin_edges[1:])/2.
    bin_edges_min = bin_edges.min()
    bin_edges_max = bin_edges.max()
    y[x<bin_edges_min] = 0
    y[x>bin_edges_max] = 0
    bin_means = binnedstat(x,y,bins=bin_edges,statistic=np.nanmean)[0]
    return cents,bin_means

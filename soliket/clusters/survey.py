"""
.. module:: survey

This module contains useful functions to internally required by the cluster likelihood to
navigate cluster catalogues. The ``SurveyData`` class contains information about the
specific survey.

"""

import os

import astropy.io.fits as pyfits
import astropy.table as atpy
import numpy as np
from astropy.io import fits

# from astLib import astWCS
from astropy.wcs import WCS
from scipy import interpolate


def read_clust_cat(fitsfile, qmin):
    list = fits.open(fitsfile)
    data = list[1].data
    SNR = data.field("SNR2p4")
    z = data.field("z")
    zerr = data.field("zErr")
    Y0 = data.field("y0tilde")
    Y0err = data.field("y0tilde_err")
    ind = np.where(SNR >= qmin)[0]
    print("num clust ", np.shape(ind), qmin)
    return z[ind], zerr[ind], Y0[ind], Y0err[ind]


def read_mock_cat(fitsfile, qmin):
    list = fits.open(fitsfile)
    data = list[1].data
    SNR = data.field("fixed_SNR")
    z = data.field("redshift")
    zerr = data.field("redshiftErr")
    Y0 = data.field("fixed_y_c")
    Y0err = data.field("err_fixed_y_c")
    ind = np.where(SNR >= qmin)[0]
    return z[ind], zerr[ind], Y0[ind], Y0err[ind]


def read_matt_mock_cat(fitsfile, qmin):
    list = fits.open(fitsfile)
    data = list[1].data
    # ra = data.field("RADeg")
    # dec = data.field("decDeg")
    z = data.field("redshift")
    zerr = data.field("redshiftErr")
    Y0 = data.field("fixed_y_c")
    Y0err = data.field("fixed_err_y_c")
    SNR = data.field("fixed_SNR")
    # M = data.field("true_M500")
    ind = np.where(SNR >= qmin)[0]
    return z[ind], zerr[ind], Y0[ind], Y0err[ind]


def read_matt_cat(fitsfile, qmin):
    list = fits.open(fitsfile)
    data = list[1].data
    z = data.field("redshift")
    zerr = data.field("redshiftErr")
    Y0 = data.field("fixed_y_c")
    Y0err = data.field("fixed_err_y_c")
    SNR = data.field("fixed_SNR")
    ind = np.where(SNR >= qmin)[0]
    return z[ind], zerr[ind], Y0[ind], Y0err[ind]


def loadAreaMask(extName, DIR):
    """Loads the survey area mask (i.e., after edge-trimming and point source masking,
     produced by nemo).
    Returns map array, wcs
    """
    areaImg = pyfits.open(os.path.join(DIR, "areaMask%s.fits.gz" % extName))
    areaMap = areaImg[0].data
    wcs = WCS(areaImg[0].header)  # , mode="pyfits")
    areaImg.close()

    return areaMap, wcs


def loadRMSmap(extName, DIR):
    """Loads the survey RMS map (produced by nemo).
    Returns map array, wcs
    """
    areaImg = pyfits.open(
        os.path.join(DIR, "RMSMap_Arnaud_M2e14_z0p4%s.fits.gz" % extName)
    )
    areaMap = areaImg[0].data
    wcs = WCS(areaImg[0].header)  # , mode="pyfits")
    areaImg.close()

    return areaMap, wcs


def loadQ(source, tileNames=None):
    """Load the filter mismatch function Q as a dictionary of spline fits.
    Args:
        source (NemoConfig or str): Either the path to a .fits table (containing Q fits
            for all tiles - this is normally selFn/QFit.fits), or a NemoConfig object
            (from which the path and tiles to use will be inferred).
        tileNames (optional, list): A list of tiles for which the Q function will be
        extracted. If source is a NemoConfig object, this should be set to None.
    Returns:
        A dictionary (with tile names as keys), containing spline knots for the Q
        function for each tile.
    """
    if type(source) == str:
        combinedQTabFileName = source
    else:
        # We should add a check to confirm this is actually a NemoConfig object
        combinedQTabFileName = os.path.join(source.selFnDir, "QFit.fits")
        tileNames = source.tileNames
    tckDict = {}
    if os.path.exists(combinedQTabFileName):
        combinedQTab = atpy.Table().read(combinedQTabFileName)
        for key in combinedQTab.keys():
            if key != "theta500Arcmin":
                tckDict[key] = interpolate.splrep(
                    combinedQTab["theta500Arcmin"], combinedQTab[key]
                )
    else:
        if tileNames is None:
            raise Exception(
                "If source does not point to a complete QFit.fits file,\
                you need to supply tileNames."
            )
        for tileName in tileNames:
            tab = atpy.Table().read(
                combinedQTabFileName.replace(".fits", "#%s.fits" % tileName)
            )
            tckDict[tileName] = interpolate.splrep(tab["theta500Arcmin"], tab["Q"])
    return tckDict


class SurveyData:
    def __init__(
        self,
        nemoOutputDir,
        ClusterCat,
        qmin=5.6,
        szarMock=False,
        MattMock=False,
        tiles=False,
        num_noise_bins=20,
    ):
        self.nemodir = nemoOutputDir

        self.tckQFit = loadQ(self.nemodir + "/QFit.fits")
        self.qmin = qmin
        self.tiles = tiles
        self.num_noise_bins = num_noise_bins

        if szarMock:
            print("mock catalog")
            self.clst_z, self.clst_zerr, self.clst_y0, self.clst_y0err = read_mock_cat(
                ClusterCat, self.qmin
            )
        elif MattMock:
            print("Matt mock catalog")
            self.clst_z, self.clst_zerr, self.clst_y0, self.clst_y0err = read_matt_cat(
                ClusterCat, self.qmin
            )
        else:
            print("real catalog")
            self.clst_z, self.clst_zerr, self.clst_y0, self.clst_y0err = read_clust_cat(
                ClusterCat, self.qmin
            )

        if tiles:
            self.filetile = self.nemodir + "/tileAreas.txt"
            self.tilenames = np.loadtxt(
                self.filetile, dtype=np.str, usecols=0, unpack=True
            )
            self.tilearea = np.loadtxt(
                self.filetile, dtype=np.float, usecols=1, unpack=True
            )

            self.fsky = []
            self.mask = []
            self.mwcs = []
            self.rms = []
            self.rwcs = []
            self.rmstotal = np.array([])

            for i in range(len(self.tilearea)):
                self.fsky.append(self.tilearea[i] / 41252.9612)
                tempmask, tempmwcs = loadAreaMask("#" + self.tilenames[i], self.nemodir)
                self.mask.append(tempmask)
                self.mwcs.append(tempmwcs)
                temprms, temprwcs = loadRMSmap("#" + self.tilenames[i], self.nemodir)
                self.rms.append(temprms)
                self.rwcs.append(temprwcs)
                self.rmstotal = np.append(self.rmstotal, temprms[temprms > 0])

            self.fskytotal = np.sum(self.fsky)
        else:
            self.rms, self.rwcs = loadRMSmap("", self.nemodir)
            self.mask, self.mwcs = loadAreaMask("", self.nemodir)
            self.rmstotal = self.rms[self.rms > 0]
            self.fskytotal = 987.5 / 41252.9612

        count_temp, bin_edge = np.histogram(
            np.log10(self.rmstotal), bins=self.num_noise_bins
        )

        self.frac_of_survey = count_temp * 1.0 / np.sum(count_temp)
        self.Ythresh = 10 ** ((bin_edge[:-1] + bin_edge[1:]) / 2.0)

    @property
    def Q(self):
        if self.tiles:
            return self.tckQFit["Q"]
        else:
            return self.tckQFit["PRIMARY"]

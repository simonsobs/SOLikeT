import os
import numpy as np
import scipy
from scipy import interpolate
import astropy.io.fits as pyfits

from astropy.wcs import WCS
from astropy.io import fits
import astropy.table as atpy
import nemo as nm # needed for reading Q-functions

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
    ra = data.field("RADeg")
    dec = data.field("decDeg")
    z = data.field("redshift")
    zerr = data.field("redshiftErr")
    Y0 = data.field("fixed_y_c") # tsz_signal
    Y0err = data.field("fixed_err_y_c") # tsz_signal_err
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
    areaImg = pyfits.open(os.path.join(DIR, "areaMask%s.fits.gz" % (extName)))
    areaMap = areaImg[0].data
    wcs = WCS(areaImg[0].header)
    areaImg.close()

    return areaMap, wcs


def loadRMSmap(extName, DIR):
    """Loads the survey RMS map (produced by nemo).
    Returns map array, wcs
    """
    areaImg = pyfits.open(
        os.path.join(DIR, "RMSMap_Arnaud_M2e14_z0p4%s.fits.gz" % (extName))
    )
    areaMap = areaImg[0].data
    wcs = WCS(areaImg[0].header)
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
                combinedQTabFileName.replace(".fits", "#%s.fits" % (tileName))
            )
            tckDict[tileName] = interpolate.splrep(tab["theta500Arcmin"], tab["Q"])
    return tckDict


class SurveyData:
    def __init__(
        self,
        lkl,
        nemoOutputDir,
        ClusterCat,
        qmin=5.6,
        szarMock=False,
        MattMock=False,
        tiles=False,
        num_noise_bins=2,
    ):
        self.nemodir = nemoOutputDir

        # self.tckQFit = loadQ(self.nemodir + "/QFit.fits")
        print(lkl.data['Q_file'])
        self.datafile_Q = lkl.data['Q_file']
        filename_Q, ext = os.path.splitext(self.datafile_Q)
        datafile_Q_dwsmpld = os.path.join(lkl.data_directory,
                             filename_Q + 'dwsmpld_nbins={}'.format(lkl.selfunc['dwnsmpl_bins']) + '.npz')
        if os.path.exists(datafile_Q_dwsmpld):
            lkl.log.info('Reading in binned Q function from file.')
            Qfile = np.load(datafile_Q_dwsmpld)
            lkl.allQ = Qfile['Q_dwsmpld']
            lkl.tt500 = Qfile['tt500']
        # exit(0)

        else:
            lkl.log.info('Reading full Q function.')
            tile_area = np.genfromtxt(os.path.join(lkl.data_directory, lkl.data['tile_file']), dtype=str)
            tilename = tile_area[:, 0]
            QFit = nm.signals.QFit(QFitFileName=os.path.join(lkl.data_directory, self.datafile_Q), tileNames=tilename)
            Nt = len(tilename)
            lkl.log.info("Number of tiles = {}.".format(Nt))

            hdulist = fits.open(os.path.join(lkl.data_directory, self.datafile_Q))
            data = hdulist[1].data
            tt500 = data.field("theta500Arcmin")

            # reading in all Q functions
            allQ = np.zeros((len(tt500), Nt))
            for i in range(Nt):
                allQ[:, i] = QFit.getQ(tt500, tileName=tile_area[:, 0][i])
            assert len(tt500) == len(allQ[:, 0])
            lkl.tt500 = tt500
            lkl.allQ = allQ

        lkl.log.info('Reading full RMS.')
        self.datafile_rms = lkl.datafile_rms
        filename_rms, ext = os.path.splitext(self.datafile_rms)
        datafile_rms_dwsmpld = os.path.join(lkl.data_directory,
                filename_rms + 'dwsmpld_nbins={}'.format(lkl.selfunc['dwnsmpl_bins']) + '.npz')
        datafile_tiles_dwsmpld = os.path.join(lkl.data_directory,
                'tile_names' + 'dwsmpld_nbins={}'.format(lkl.selfunc['dwnsmpl_bins']) + '.npy')
        # if (self.selfunc['mode'] == 'downsample' and self.selfunc['save_dwsmpld'] is False)  or (
        #     self.selfunc['mode'] == 'downsample' and self.selfunc['save_dwsmpld'] and not os.path.exists(datafile_rms_dwsmpld)):

        if os.path.exists(datafile_rms_dwsmpld):
            rms = np.load(datafile_rms_dwsmpld)
            # print(len(rms['noise']))
            # exit(0)
            lkl.noise = rms['noise']
            lkl.skyfracs = rms['skyfracs']
            lkl.log.info("Number of rms bins = {}.".format(lkl.skyfracs.size))

            lkl.tiles_dwnsmpld = np.load(datafile_tiles_dwsmpld,allow_pickle='TRUE').item()
            print(lkl.tiles_dwnsmpld)
            # exit(0)
        else:
            lkl.log.info('Reading in full RMS table.')

            list = fits.open(os.path.join(lkl.data_directory, self.datafile_rms))
            file_rms = list[1].data

            self.noise = file_rms['y0RMS']
            self.skyfracs = lkl.skyfracs#file_rms['areaDeg2']*np.deg2rad(1.)**2
            self.tname = file_rms['tileName']
            lkl.log.info("Number of tiles = {}. ".format(len(np.unique(self.tname))))
            lkl.log.info("Number of sky patches = {}.".format(self.skyfracs.size))
            # exit(0)

            lkl.log.info('Downsampling RMS and Q function using {} bins.'.format(lkl.selfunc['dwnsmpl_bins']))
            binned_stat = scipy.stats.binned_statistic(self.noise, self.skyfracs, statistic='sum',
                                                       bins=lkl.selfunc['dwnsmpl_bins'])
            binned_area = binned_stat[0]
            binned_rms_edges = binned_stat[1]

            bin_ind = np.digitize(self.noise, binned_rms_edges)
            tiledict = dict(zip(tilename, np.arange(tile_area[:, 0].shape[0])))

            Qdwnsmpld = np.zeros((lkl.allQ.shape[0], lkl.selfunc['dwnsmpl_bins']))
            tiles_dwnsmpld = {}

            for i in range(lkl.selfunc['dwnsmpl_bins']):
                tempind = np.where(bin_ind == i + 1)[0]
                if len(tempind) == 0:
                    lkl.log.info('Found empty bin.')
                    Qdwnsmpld[:, i] = np.zeros(lkl.allQ.shape[0])
                else:
                    print('dowsampled rms bin ',i)
                    temparea = self.skyfracs[tempind]
                    print('areas of tiles in bin',temparea)
                    temptiles = self.tname[tempind]
                    print('names of tiles in bin',temptiles)
                    for t in temptiles:
                        tiles_dwnsmpld[t] = i

                    test = [tiledict[key] for key in temptiles]
                    Qdwnsmpld[:, i] = np.average(lkl.allQ[:, test], axis=1, weights=temparea)

            lkl.noise = 0.5*(binned_rms_edges[:-1] + binned_rms_edges[1:])
            lkl.skyfracs = binned_area
            lkl.allQ = Qdwnsmpld
            lkl.tiles_dwnsmpld = tiles_dwnsmpld
            print('len(tiles_dwnsmpld)',tiles_dwnsmpld)
            lkl.log.info("Number of downsampled sky patches = {}.".format(lkl.skyfracs.size))

            assert lkl.noise.shape[0] == lkl.skyfracs.shape[0] and lkl.noise.shape[0] == lkl.allQ.shape[1]

            if lkl.selfunc['save_dwsmpld']:
                np.savez(datafile_Q_dwsmpld, Q_dwsmpld=Qdwnsmpld, tt500=lkl.tt500)
                np.savez(datafile_rms_dwsmpld, noise=lkl.noise, skyfracs=lkl.skyfracs)
                np.save(datafile_tiles_dwsmpld, lkl.tiles_dwnsmpld)

        # exit(0)
        self.qmin = lkl.qcut
        # self.tiles = tiles
        self.num_noise_bins = lkl.skyfracs.size

        # if szarMock:
        #     print("mock catalog, using read_matt_mock_cat")
        #     self.clst_z, self.clst_zerr, self.clst_y0, self.clst_y0err = read_matt_mock_cat(
        #         ClusterCat, self.qmin
        #     )
        # elif MattMock:
        #     print("Matt mock catalog")
        #     self.clst_z, self.clst_zerr, self.clst_y0, self.clst_y0err = read_matt_cat(
        #         ClusterCat, self.qmin
        #     )
        # else:
        #     print("real catalog")
        #     self.clst_z, self.clst_zerr, self.clst_y0, self.clst_y0err = read_clust_cat(
        #         ClusterCat, self.qmin
        #     )
        #
        # if tiles:
        #     self.filetile = self.nemodir + "/tileAreas.txt"
        #     self.tilenames = np.loadtxt(
        #         self.filetile, dtype=np.str, usecols=0, unpack=True
        #     )
        #     self.tilearea = np.loadtxt(
        #         self.filetile, dtype=np.float, usecols=1, unpack=True
        #     )
        #
        #     self.fsky = []
        #     self.mask = []
        #     self.mwcs = []
        #     self.rms = []
        #     self.rwcs = []
        #     self.rmstotal = np.array([])
        #
        #     for i in range(len(self.tilearea)):
        #         self.fsky.append(self.tilearea[i] / 41252.9612)
        #         tempmask, tempmwcs = loadAreaMask("#" + self.tilenames[i], self.nemodir)
        #         self.mask.append(tempmask)
        #         self.mwcs.append(tempmwcs)
        #         temprms, temprwcs = loadRMSmap("#" + self.tilenames[i], self.nemodir)
        #         self.rms.append(temprms)
        #         self.rwcs.append(temprwcs)
        #         self.rmstotal = np.append(self.rmstotal, temprms[temprms > 0])
        #
        #     self.fskytotal = np.sum(self.fsky)
        # else:
        #     # self.rms, self.rwcs = loadRMSmap("", self.nemodir)
        #     # self.mask, self.mwcs = loadAreaMask("", self.nemodir)
        #     # tcat = '/Users/boris/Work/CLASS-SZ/SO-SZ/SOLikeT/soliket/clusters/data/selFn_SO/stitched_RMSMap_Arnaud_M2e14_z0p4.fits'
        #     tcat = os.path.join(self.nemodir, "stitched_RMSMap_Arnaud_M2e14_z0p4.fits")
        #     list = pyfits.open(tcat)
        #     self.rms = list[1].data
        #
        #     self.rmstotal = self.rms[self.rms > 0]
        #     self.fskytotal = 987.5 / 41252.9612
        #
        # count_temp, bin_edge = np.histogram(
        #     np.log10(self.rmstotal), bins=self.num_noise_bins
        # )

        # self.frac_of_survey = count_temp * 1.0 / np.sum(count_temp)
        # self.Ythresh = 10 ** ((bin_edge[:-1] + bin_edge[1:]) / 2.0)

        self.frac_of_survey  = lkl.skyfracs
        self.fskytotal = lkl.skyfracs.sum()
        self.Ythresh = lkl.noise
        print('self.Ythresh',len(self.Ythresh),self.Ythresh)

    @property
    def Q(self):
        # if self.tiles:
        return self.tckQFit["Q"]
        # else:
        #     print(self.tckQFit.keys())
        #     return self.tckQFit["PRIMARY"]

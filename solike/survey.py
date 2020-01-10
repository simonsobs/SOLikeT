import numpy as np
import astropy.io.fits as pyfits
from astLib import astWCS
from nemo import signals
from astropy.io import fits

def read_clust_cat(fitsfile,qmin):
    list = fits.open(fitsfile)
    data = list[1].data
    SNR = data.field('SNR2p4')
    z = data.field('z')
    zerr = data.field('zErr')
    Y0 = data.field('y0tilde')
    Y0err = data.field('y0tilde_err')
    ind = np.where(SNR >= qmin)[0]
    return z[ind],zerr[ind],Y0[ind],Y0err[ind]

def read_mock_cat(fitsfile,qmin):
    list = fits.open(fitsfile)
    data = list[1].data
    SNR = data.field('fixed_SNR')
    z = data.field('redshift')
    zerr = data.field('redshiftErr')
    Y0 = data.field('fixed_y_c')
    Y0err = data.field('err_fixed_y_c')
    ind = np.where(SNR >= qmin)[0]
    return z[ind],zerr[ind],Y0[ind],Y0err[ind]

def read_matt_mock_cat(fitsfile,qmin):
    list = fits.open(fitsfile)
    data = list[1].data
    ra = data.field('RADeg')
    dec = data.field('decDeg')
    z = data.field('redshift')
    zerr = data.field('redshiftErr')
    Y0 = data.field('fixed_y_c')
    Y0err = data.field('fixed_err_y_c')
    SNR = data.field('fixed_SNR')
    M = data.field('true_M500')
    ind = np.where(SNR >= qmin)[0]
    return z[ind],zerr[ind],Y0[ind],Y0err[ind]

def loadAreaMask(extName, DIR):
    """Loads the survey area mask (i.e., after edge-trimming and point source masking, produced by nemo).
    Returns map array, wcs
    """
    areaImg=pyfits.open(DIR+"areaMask%s.fits.gz" % (extName))
    areaMap=areaImg[0].data
    wcs=astWCS.WCS(areaImg[0].header, mode = 'pyfits')
    areaImg.close()

    return areaMap, wcs

def loadRMSmap(extName, DIR):
    """Loads the survey RMS map (produced by nemo).
    Returns map array, wcs
    """
    areaImg=pyfits.open(DIR+"RMSMap_Arnaud_M2e14_z0p4%s.fits.gz" % (extName))
    areaMap=areaImg[0].data
    wcs=astWCS.WCS(areaImg[0].header, mode = 'pyfits')
    areaImg.close()

    return areaMap, wcs

class SurveyData(object):
    def __init__(self,nemoOutputDir,ClusterCat,qmin=5.6,szarMock=False,tiles=False):
        self.nemodir = nemoOutputDir
        self.tckQFit=signals.loadQ(self.nemodir + '/QFit.fits')
        self.qmin = qmin

        if (szarMock):
            print("mock catalog")
            self.clst_z,self.clst_zerr,self.clst_y0,self.clst_y0err = read_mock_cat(ClusterCat,self.qmin)
        else:
            print("real catalog")
            self.clst_z,self.clst_zerr,self.clst_y0,self.clst_y0err = read_clust_cat(ClusterCat,self.qmin)

        if (tiles):
            self.filetile = self.nemodir + 'tileAreas.txt'
            self.tilenames = np.loadtxt(self.filetile,dtype=np.str,usecols = 0,unpack=True)
            self.tilearea = np.loadtxt(self.filetile,dtype=np.float,usecols = 1,unpack=True)

            self.fsky = []
            self.mask = []
            self.mwcs = []
            self.rms  = []
            self.rwcs = []
            self.rmstotal = np.array([])

            for i in range(len(self.tilearea)):
                self.fsky.append(self.tilearea[i]/41252.9612)
                tempmask,tempmwcs = loadAreaMask('#'+self.tilenames[i],self.nemodir)
                self.mask.append(tempmask)
                self.mwcs.append(tempmwcs)
                temprms,temprwcs = loadRMSmap('#'+self.tilenames[i],self.nemodir)
                self.rms.append(temprms)
                self.rwcs.append(temprwcs)
                self.rmstotal = np.append(self.rmstotal,temprms[temprms > 0])
            
            self.fskytotal = np.sum(self.fsky)
        else:
            self.rms, self.rwcs = loadRMSmap('',self.nemodir)
            self.mask, self.mwcs = loadAreaMask('',self.nemodir)
            self.rmstotal = self.rms[self.rms > 0]
            self.fskytotal = 987.5/41252.9612

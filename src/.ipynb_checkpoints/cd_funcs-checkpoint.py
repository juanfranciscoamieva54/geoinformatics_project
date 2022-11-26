import copy
import numpy as np
from scipy.spatial import distance
from sklearn.metrics.pairwise import cosine_similarity
# https://stackoverflow.com/questions/61490351/scipy-cosine-similarity-vs-sklearn-cosine-similarity

def cva(zb1, zb2, zcube1, zcube2, zbs1, zbs2):
    """
    Compute magnitude and angle with 2-band 
    change vector analysis (CVA) in the 2D case

    zb1: ndarray, bands of the first data cube
    zb2: ndarray, bands of the second cube (must be equal zb1)
    zcube1: ndarray, data for first cube
    zcube2: ndarray, data for second cube
    zbs1: float, first band selected
    zbs2: float, second band selected

    mag: ndarray, amplitude of change
    ang: ndarray, direction of change
    """

    if (zb1 != zb2).any():
        raise ValueError('Unexpected difference in bands')

    if zbs1 == zbs2:
        raise ValueError('Two bands for difference cannot be equal')

    # Compute spectral difference of cubes
    dcube = zcube1 - zcube2
    dcube1 = dcube[:,:,np.abs(zb1-zbs1).argmin()].squeeze()
    dcube2 = dcube[:,:,np.abs(zb1-zbs2).argmin()].squeeze()

    mag = (dcube1**2 + dcube2**2)**0.5

    ang = np.arctan2(dcube1,dcube2) / np.pi * 180
    negang = ang < 0
    ang[negang] = 360 + ang[negang] 

    return mag, ang


def c2va(zb1, zb2, zcube1, zcube2):
    """
    Compute magnitude and angle of change with the
    compressed change vector analysis (C2VA) technique

    """
    
    # First calculate the difference cube matrix
    xdif = zcube2 - zcube1

    # Define the reference vector in the B-dimensional space
    # (B being the number of bands)
    xref = np.ones(len(zb1))*np.sqrt(len(zb1))/len(zb1)

    # Calculate difference magnitude (sum over bands)
    mag = np.sqrt(np.sum((xdif**2),axis=-1))

    # Calculate angle between difference and reference vectors,
    # it must be within 0 and pi. Avoid division by 0.
    denom =  np.sqrt(np.sum(xdif**2,axis=-1)*np.sum(xref[None,None,:]**2,axis=-1))
    denom[denom == 0.] = 1.
    ang = np.arccos(np.sum(xdif*xref[None,None,:],axis=-1)/denom)

    return mag, ang


def diffs(zb, zcube1, zcube2, mode=None, bsel=None):
    """
    Non-CVA difference methods for registered cubes

    Parameters:
    ----------
    zb: ndarray, data bands
    zcube1: ndarray, first data cube
    zcube2: ndarray, target data cube
    mode: str, difference method (sdif, logr, glrt or coss)
    bsel: float, band to be used for difference

    Returns:
    -------
    zdiff: ndarray, difference data cube (sized as inputs)
    """

    # The indices of 10.1109/IGARSS.2013.6723707 were used on SAR images,
    # here adapted to single bands from multispectral data.
    # The band should be selected depending on the case under consideration.

    if bsel != None:
        idx = np.abs(zb-bsel).argmin()

    # Note that the magnitude can be easily computed in the calling code

    if mode not in ['sdif', 'logr', 'glrt', 'coss']:
        raise KeyError('{} method not implemented'.format(mode))

    # Standard, plain difference
    if mode == 'sdif':
        zdiff = (zcube2[:,:,idx] - zcube1[:,:,idx]).squeeze()

    # Compute arrays with padded zero for use in divisions
    if mode in ['logr', 'glrt']:
        zcube1p = zcube1[:,:,idx].squeeze()
        zcube1p[zcube1p==0] = 1.
        zcube2p = zcube2[:,:,idx].squeeze()
        zcube2p[zcube2p==0] = 1.
        zmask = zcube1[:,:,idx].squeeze()
        zmask[zmask>0] = 1.

    # Retain bands for the cosine distance case
    if mode in ['coss']:
        zcube1p = copy.deepcopy(zcube1)
        zcube1p[zcube1p==0] = 1.
        zcube2p = copy.deepcopy(zcube2)
        zcube2p[zcube2p==0] = 1.
        # Multiplying by mask makes no difference, to be checked TODO
        zmask = np.amax(zcube1,axis=-1) # find pixels inside image
        zmask[zmask>0] = 1. 

    # Log-ratio method
    if mode == 'logr':
        zdiff = np.minimum(zcube2[:,:,idx]/zcube1p,zcube1[:,:,idx]/zcube2p)

    # Generalized likelihood ratio test
    if mode == 'glrt':
        zdiff = 2.*np.sqrt(zcube1[:,:,idx]*zcube2[:,:,idx])/(zcube1p+zcube2p)
        
    # Cosine similarity, implemented as spectral function.
    if mode == 'coss':
        shp = (np.shape(zcube1)[0],np.shape(zcube1)[1])
        zdiff = np.zeros(shp)
        #zdiff = cosine_similarity(zcube1p, zcube2p)
        for ii in range(shp[0]):
            for jj in range(shp[1]):
                zdiff[ii,jj] = distance.cosine(zcube1p[ii,jj,:], zcube2p[ii,jj,:])
        zdiff = np.ones((np.shape(zcube1)[0],np.shape(zcube1)[1])) - zdiff
        #zdiff = np.multiply(zdiff,zmask)

    return zdiff
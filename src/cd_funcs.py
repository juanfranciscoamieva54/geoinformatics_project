import copy
import numpy as np
from scipy.spatial import distance
from sklearn.metrics.pairwise import cosine_similarity

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

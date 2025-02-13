import numpy as np
import h5py
from skimage import data, draw, io
import matplotlib.pyplot as plt
from img_list import use_cases


def spherdist(zlon1,zlat1,zlon2,zlat2):
    """ Haversine distance between two points

    zlon1: float, longitude of first point in degrees
    zlat1: float, latitude of first point
    zlon2: float, longitude of second point
    zlat2: float, latitude of second point

    zd: float, distance (km) between points 1 and 2
    """
    d2r = np.pi/180.
    zhavelon = np.sin(0.5*(zlon2-zlon1)*d2r)**2
    zhavelat = np.sin(0.5*(zlat2-zlat1)*d2r)**2
    zcosfac = np.cos(zlat1*d2r)*np.cos(zlat2*d2r)
    zc = 2*np.arcsin(np.sqrt(zhavelat+(zcosfac*zhavelon)))
    zd = zc*6371000./1000.
    return zd

# Find corner indices in grids
def idx_mesh(x, y, X, Y, maxdist=None):
    """
    Find indices in mesh nearest to coordinates

    x: float, first coordinate target value
    y: float, second coordinate target value
    X: ndarray, first array of the mesh
    Y: ndarray, second array of the mesh
    maxdist: float, difference allowed (in meters)
    
    out_idx: list, indices for lon and lat
    """
    distance = (X-x)**2 + (Y-y)**2
    idxs = np.argwhere(distance == np.amin(distance))[0]
    out_idx = (idxs[0], idxs[1])
    
    # Calculate distance in meters with the haversine formula
    diffm = 1000.*spherdist(x, y, X[idxs[0], idxs[1]], Y[idxs[0], idxs[1]])

    if maxdist is not None and diffm > maxdist:
        print('Tolerance {} m exceeded!'.format(maxdist))

    return out_idx

def make_rgb(ztstart, ztend, path):
    """
    Helper function to make RGB-like images
    from the visible cubes
    """
    wl_rgb = [640., 560., 480.]
    yy, xx, vwl, _, vrf, _, info = read_prs_l2d(path, ztstart, ztend)
    img = vrf.astype(float)

    # Get indices for the RGB channels
    idx_rgb = []
    for wl in wl_rgb:
        idx_rgb.append(np.abs(vwl-wl).argmin())

    rgb = img[:,:, idx_rgb]

    # Normalize frames for better visualization. 
    # Using np.max() is not good if there are clouds or bright features.
    for iw in range(len(wl_rgb)):
        rgb[:,:,iw] /= np.percentile(rgb[:,:,iw],99.5)
    
    return rgb

def make_rgb_dc(img, vwl):
    """
    Helper function to make RGB-like images
    from the visible cubes already coregistered
    """
    wl_rgb = [640., 560., 480.]

    img = img.astype(float) #vrf datacube here

    # Get indices for the RGB channels
    idx_rgb = []
    for wl in wl_rgb:
        idx_rgb.append(np.abs(vwl-wl).argmin())

    rgb = img[:,:, idx_rgb]

    # Normalize frames for better visualization. 
    # Using np.max() is not good if there are clouds or bright features.
    for iw in range(len(wl_rgb)):
        rgb[:,:,iw] /= np.percentile(rgb[:,:,iw],99.5)
    
    return rgb

def find_nearest(array, value):
    idx = (np.abs(array - value)).argmin()
    return idx

def reflectance_norm(path_l2d,t1_0,t2_0,tstart1,tstart2,tend1,tend2):
    
    pf1 = h5py.File(path_l2d+'PRS_L2D_STD_'+tstart1+'_'+tend1+'_0001.he5','r')
    pf2 = h5py.File(path_l2d+'PRS_L2D_STD_'+tstart2+'_'+tend2+'_0001.he5','r')
    attrs1 = pf1.attrs
    L2ScaleVnirMax1 = attrs1['L2ScaleVnirMax']
    L2ScaleVnirMin1 = attrs1['L2ScaleVnirMin']

    attrs2 = pf2.attrs
    L2ScaleVnirMax2 = attrs2['L2ScaleVnirMax']
    L2ScaleVnirMin2 = attrs2['L2ScaleVnirMin'] 
    
    t1=L2ScaleVnirMin1 + t1_0*(L2ScaleVnirMax1-L2ScaleVnirMin1)/65535 # scaling to get reflectance
    t2=L2ScaleVnirMin2 + t2_0*(L2ScaleVnirMax2-L2ScaleVnirMin2)/65535
    
    return t1,t2

def get_coord_id(path_l2d,img,tstart,tend):
    
    pf1 = h5py.File(path_l2d+'PRS_L2D_STD_'+tstart1+'_'+tend1+'_0001.he5','r')
    attrs1 = pf1.attrs
    lat = attrs1['Product_center_lat']
    lon = attrs1['Product_center_long']
    img_id = attrs1['Image_ID']

    return lat,lon,img_id

def AOI(t1,t2,vwl,long=None,lat=None,key=None,coreg=True):
    #select an AOI in which to perform CVA
    y='n'
    rgb1 = make_rgb_dc(t1,vwl)
    rgb2 = make_rgb_dc(t2,vwl)
    while y=='n':
        if key==None:
            xtopleft,ytopleft,xbottomright,ybottomright=input('insert the 4 indexes of the square AOI separated by a space (order: xtopleft,ytopleft,xbottomright,ybottomright): ').split() 
        elif coreg:
            xtopleft,ytopleft,xbottomright,ybottomright=use_cases[key][1]
            y='y'
        else:
            xtopleft,ytopleft=use_cases[key][3]
            xbottomright=xtopleft+512
            ybottomright=ytopleft+512
            y='y'
        start=[int(ytopleft),int(xtopleft)]
        end=[ int(ybottomright), int(xbottomright)]
        row, col = draw.rectangle_perimeter(start=start, end=end)   
        fig, ax = plt.subplots(1, 1,figsize=(10, 10))
        ax.imshow(rgb1)
        ax.plot(col, row, "--y",linewidth=7,color='red')
        if y!='y':
            y=input('do you want to confirm the indices? (y/n): ')

    crop1=rgb1[start[0]:end[0],start[1]:end[1],:]
    crop2=rgb2[start[0]:end[0],start[1]:end[1],:]
    plt.figure(figsize=(20,20))
    plt.subplot(1,2,1)
    plt.imshow(crop1)
    plt.subplot(1,2,2)
    plt.imshow(crop2)
    t1_crop=t1[start[0]:end[0],start[1]:end[1],:]
    t2_crop=t2[start[0]:end[0],start[1]:end[1],:]

    print('the datacube is now cropped to the new AOI')
    print('those are the new sizes:')
    print(t1_crop.shape)
    print(t2_crop.shape)
    
    if np.any(long) and np.any(lat):
        xcrop=long[start[0]:end[0],start[1]:end[1]]
        ycrop=lat[start[0]:end[0],start[1]:end[1]]
        return crop1,crop2, t1_crop, t2_crop,xcrop, ycrop
    
    else:
        return crop1,crop2, t1_crop, t2_crop

def append_new_line(file_name, text_to_append):
    """Append given text as a new line at the end of file"""
    # Open the file in append & read mode ('a+')
    with open(file_name, "a+") as file_object:
        # Move read cursor to the start of file.
        file_object.seek(0)
        # If file is not empty then append '\n'
        data = file_object.read()
        if len(data) > 0:
            file_object.write("\n")
        # Append text at the end of file
        file_object.write(text_to_append)
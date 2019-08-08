import numpy as np
from . import interpolation
from . import observation

def make_ker2D(shape, coord_lr):
    '''Builds a coordinate kernel for 2D sinc interpolation
    This function is used in the making of the resampling convolution operator.
    It create a kernel based on the sinc of the difference between coordinates in a high resolution frame and reference
    coordinate (coord_lr)
    Parameters
    ----------
    shape: tuple
        shape of the high resolution frame
    coord_lr: arrays
        coordinate of the low resolution location where to compute mapping
    Returns
    -------
    result: array
        vector for convolution and resampling of the high resolution plane into pixel (xm,ym) at low resolution
    '''

    B, Ny, Nx = shape

    y_lr, x_lr = coord_lr

    N_lr = y_lr.size
    ker = np.zeros((N_lr,Ny, Nx))
    y, x = np.where(ker[0] == 0)

    for m in range(N_lr):
        ker[m, y, x] = interpolation.sinc2D((y_lr[m] - y), (x_lr[m] - x))

    return ker

def make_ker1D(coord_hr, coord_lr, axis = 0):
    '''Builds a coordinate kernel for 1D sinc interpolation
    This function is used in the making of the resampling convolution operator.
    It creates a kernel based on the sinc of the difference between coordinates in a high resolution frame and reference
    coordinate (ym,xm)
    Parameters
    ----------
    Ny: tuple
        size of the high resolution frame along the y-axis
    coord_lr: arrays
        coordinate of the low resolution location where to compute mapping
    axis: int
        axis along which the kernel is being built
    Returns
    -------
    result: array
        vector for convolution and resampling of the high resolution plane into pixel (xm,ym) at low resolution
    '''
    assert axis in [0,1]

    z_lr = coord_lr[axis]
    z = coord_hr[axis]

    if axis == 0:
        ker = np.sinc(z_lr[:,np.newaxis]-z[np.newaxis, :])
    elif axis == 1:
        ker = np.sinc(z_lr[np.newaxis, :] - z[:,np.newaxis])

    return ker


def match_patches(shape_hr, shape_lr, wcs_hr, wcs_lr, isrot = True, perimeter  = 'overlap', psf = False):
    '''Matches datasets at different resolutions

    Finds the region of overlap between two datasets and creates a mask for the region as well as the pixel coordinates
    for the dataset pixels inside the overlap.

    Parameters
    ----------
    shape_hr, shape_lr: tuples
        shapes of the two datasets
    wcs_hr, wcs_lr: WCS objects
        WCS of the Low and High resolution fields respectively
    perimeter: string
        returns the coordinates in the intersection or union of both frames if set to 'overlap' or 'union' respectively

    Returns
    -------
    coordlr_over_lr: array
        coordinates of the overlap in low resolution.
    coordlr_over_hr: array
        coordinates of the overlaps at low resolution in the high resolution frame.
    coordhr_hr: array
        coordinates of the high resolution pixels in the overlap. Necessary for psf matching
    '''

    assert perimeter in ['overlap', 'union'], 'perimeter should be either overlap or union.'

    if np.size(shape_hr) == 3:
        B_hr, Ny_hr, Nx_hr = shape_hr
    elif np.size(shape_hr) == 2:
        Ny_hr, Nx_hr = shape_hr
    else:
        raise ValueError('Wrong dimensions for reference image')

    if np.size(shape_lr) == 3:
        B_lr, Ny_lr, Nx_lr = shape_lr
    elif np.size(shape_lr) == 2:
        Ny_lr, Nx_lr = shape_lr
    else:
        raise ValueError('Wrong dimensions for low resolution image')

    assert wcs_hr != None
    assert wcs_lr != None



    if psf is True:
        # Coordinates of all high resolution pixels. This is only needed for PSF matches.
        y_hr, x_hr = np.indices((Ny_hr, Nx_hr))

        x_hr = x_hr.flatten()
        y_hr = y_hr.flatten()
    else:
        # Coordinates of the high resolution pixels
        y_hr, x_hr = np.array(range(Ny_hr)), np.array(range(Nx_hr))

    # Capital letters are for coordinates of low-resolution pixels
    if (isrot is True) or (psf is True):

        # Coordinates of all low resolution pixels. All are needed if frames are rotated.
        Y_lr, X_lr = np.indices((Ny_lr, Nx_lr))

        X_lr = X_lr.flatten()
        Y_lr = Y_lr.flatten()

    else:
        Y_lr, X_lr = np.array(range(Ny_lr)), np.array(range(Nx_lr))

    #Corresponding angular positions
    #of low resolution pixels
    if np.size(wcs_lr.array_shape) == 2:
        ra_lr, dec_lr = wcs_lr.all_pix2world(X_lr, Y_lr, 0, ra_dec_order=True)
    elif np.size(wcs_lr.array_shape) == 3:
        ra_lr, dec_lr = wcs_lr.all_pix2world(X_lr, Y_lr, 0, 0, ra_dec_order=True)
    #of high resolution pixels
    if np.size(wcs_hr.array_shape) == 2:
        ra_hr, dec_hr = wcs_hr.all_pix2world(x_hr, y_hr, 0, ra_dec_order=True)
    elif np.size(wcs_hr.array_shape) == 3:
        ra_hr, dec_hr, eggbenedict = wcs_hr.all_pix2world(x_hr, y_hr, 0, 0, ra_dec_order=True)

    # Coordinates of the low resolution pixels in the high resolution frame
    if np.size(wcs_hr.array_shape) == 2:
        X_hr, Y_hr = wcs_hr.all_world2pix(ra_lr, dec_lr, 0, ra_dec_order=True)
    elif np.size(wcs_hr.array_shape) == 3:
        X_hr, Y_hr, eggbenedict = wcs_hr.all_world2pix(ra_lr, dec_lr, 0, 0, ra_dec_order=True)

    # Coordinates of the high resolution pixels in the low resolution frame
    if np.size(wcs_lr.array_shape) == 2:
        x_lr, y_lr = wcs_lr.all_world2pix(ra_hr, dec_hr, 0, ra_dec_order=True)
    elif np.size(wcs_lr.array_shape) == 3:
        x_lr, y_lr, eggsbenedict = wcs_lr.all_world2pix(ra_hr, dec_hr, 0, 0, ra_dec_order=True)



    #mask of low resolution pixels at high resolution in the overlap:
    over_lr = ((X_hr > 0) * (X_hr < Nx_hr) * (Y_hr > 0) * (Y_hr < Ny_hr))
    #mask of high resolution pixels at high resolution in the overlap (needed for psf matching)
    over_hr = ((x_lr > 0) * (x_lr < Nx_lr) * (y_lr > 0) * (y_lr < Ny_lr))

    #pixels of the high resolution pixels in the overlap at high resolution (needed for PSF only)
    coordhr_hr = (y_hr[(over_hr == 1)], x_hr[(over_hr == 1)])

    class SourceInitError(Exception):
        """
        Datasets do not match, no overlap found. Check the coordinates of the observations or the WCS.
        """
        pass

    if np.sum(over_lr) == 0:
        raise SourceInitError

    if perimeter is 'overlap':
        # Coordinates of low resolution pixels in the overlap at low resolution:
        ylr_lr = Y_lr[(over_lr == 1)]
        xlr_lr = X_lr[(over_lr == 1)]
        coordlr_lr = (ylr_lr, xlr_lr)
        # Coordinates of low resolution pixels in the overlap at high resolution:
        ylr_hr = Y_hr[(over_lr == 1)]
        xlr_hr = X_hr[(over_lr == 1)]

        coordlr_hr = (ylr_hr, xlr_hr)

    elif perimeter is 'union':

        # Coordinates of low resolution pixels at low resolution:
        coordlr_lr = (Y_lr, X_lr)

        # Coordinates of low resolution pixels at high resolution:
        coordlr_hr = (Y_hr, X_hr)

    return coordlr_lr, coordlr_hr, coordhr_hr


def factor_operator(mat, shape_lr, shape_hr):
    ''' Factorises the resampling operator to speed-up operations



    '''

    nl1, nl2 = shape_lr
    nh1, nh2 = shape_lr

    #Making images out of resampling matrix
    Kern = (mat).reshape(nh1*nh2, nl1, nl2)

    #Coordinates of highres grid
    loc = np.zeros((nh1, nh2))
    xl, yl = np.where(loc == 0)
    xl = xl.reshape(nh1, nh2)
    yl = yl.reshape(nh1, nh2)
    #High res grid in the low res frame
    Xl = (xl * nl1 / (nh1)).flatten()
    Yl = (yl * nl2 / (nh2)).flatten()
    #coordinates at high res that are in the middle of the frame
    loc[(np.where((xl > nh1 / 4) * (xl < 3 * nh1 / 4) * (yl > nh2 / 4) * (yl < 3 * nh2 / 4)))] = 1

    #Slicing the matrix to take only vectors in the middle of the frame
    mat = Kern[np.where(loc.flatten() == 1), :, :]

    #Corresponding coordinates in the middle of the frame in low res reference
    Xl = Xl[np.where(loc.flatten() == 1)].astype(np.intp)
    Yl = Yl[np.where(loc.flatten() == 1)].astype(np.intp)


    xl1, xl2, yl1, yl2 = (Xl.astype(int) - nl1 / 4).astype(int), (Xl.astype(int) + nl1 / 4).astype(int), (
                Yl.astype(int) - nl2 / 4).astype(int), (Yl.astype(int) + nl2 / 4).astype(int)

    M = np.zeros((np.int(nh1 * nh2 / 4), np.int(nl1 / 2) - 1, np.int(nl2 / 2) - 1))

    #for i in range(np.int(n1 * n2 / 4)):
    #    M[i, :, :] = mat[0, i, xl1[i]:xl2[i], yl1[i]:yl2[i]]

#    U, E, V = np.linalg.svd(M.reshape(np.int(N1 / 2) * np.int(N2 / 2), np.int(n1 * n2 / 4)), full_matrices=False)

 #   Ev = np.dot(np.diag(E), V)
    pass



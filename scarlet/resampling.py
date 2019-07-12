import numpy as np
from . import interpolation

def conv2D_fft(shape, coord_lr):
    '''performs a convolution of a coordinate kernel by a psf
    This function is used in the making of the resampling convolution operator.
    It create a kernel based on the sinc of the difference between coordinates in a high resolution frame and reference
    coordinate (ym,xm)
    Parameters
    ----------
    shape: tuple
        shape of the high resolution frame
    ym, xm: arrays
        coordinate of the low resolution location where to compute mapping
    p: array
        PSF kernel
    h: float
        pixel size
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

def match_patches(shape_hr, shape_lr, wcs_hr, wcs_lr):
    '''Matches datasets at different resolutions

    Finds the region of overlap between two datasets and creates a mask for the region as well as the pixel coordinates
    for the dataset pixels inside the overlap.

    Parameters
    ----------
    shape_hr, shape_lr: tuples
        shapes of the two datasets
    wcs_hr, wcs_lr: WCS objects
        WCS of the Low and High resolution fields respectively

    Returns
    -------
    mask: array
        mask of overlapping pixel in the high resolution frame.
    coordlr_over_lr: array
        coordinates of the overlap in low resolution.
    coordlr_over_hr: array
        coordinates of the overlaps at low resolution in the high resolution frame.
    '''

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

    im_hr = np.zeros((Ny_hr, Nx_hr))
    im_lr = np.zeros((Ny_lr, Nx_lr))

    # Coordinates of pixels in both frames
    y_hr, x_hr = np.where(im_hr == 0)
    Y_lr, X_lr = np.where(im_lr == 0)

    #Corresponding angular positions
    if np.size(wcs_lr.array_shape) == 2:
        ra_lr, dec_lr = wcs_lr.all_pix2world(X_lr, Y_lr, 0, ra_dec_order=True)
    elif np.size(wcs_lr.array_shape) == 3:
        ra_lr, dec_lr = wcs_lr.all_pix2world(X_lr, Y_lr, 0, 0, ra_dec_order=True)
    if np.size(wcs_hr.array_shape) == 2:
        ra_hr, dec_hr = wcs_hr.all_pix2world(y_hr, x_hr, 0, ra_dec_order=True)
    elif np.size(wcs_hr.array_shape) == 3:
        ra_hr, dec_hr = wcs_hr.all_pix2world(y_hr, x_hr, 0, 0, ra_dec_order=True)

    # Coordinates of the low resolution pixels in the high resolution frame
    if np.size(wcs_hr.array_shape) == 2:
        X_hr, Y_hr = wcs_hr.all_world2pix(ra_lr, dec_lr, 0, ra_dec_order=True)
    elif np.size(wcs_hr.array_shape) == 3:
        X_hr, Y_hr = wcs_hr.all_world2pix(ra_lr, dec_lr, 0, 0, ra_dec_order=True)

    # Coordinates of the high resolution pixels in the low resolution frame
    if np.size(wcs_lr.array_shape) == 2:
        x_lr, y_lr = wcs_lr.all_world2pix(ra_hr, dec_hr, 0, ra_dec_order=True)
    # Coordinates of the high resolution pixels in the low resolution frame
    elif np.size(wcs_lr.array_shape) == 3:
        x_lr, y_lr, l = wcs_lr.all_world2pix(ra_hr, dec_hr, 0, 0, ra_dec_order=True)

    # Mask of low resolution pixels in the overlap at low resolution:
    over_lr = ((X_hr > 0) * (X_hr < Nx_hr) * (Y_hr > 0) * (Y_hr < Ny_hr))
    # Mask of low resolution pixels in the overlap at high resolution:
    over_hr = ((x_lr > 0) * (x_lr < Nx_lr) * (y_lr > 0) * (y_lr < Ny_lr))

    mask = over_hr.reshape(Ny_hr, Nx_hr)

    class SourceInitError(Exception):
        """
        Datasets do not match, no overlap found. Check the coordinates of the observations or the WCS.
        """
        pass

    if np.sum(mask) == 0:
        raise SourceInitError

    # Coordinates of low resolution pixels in the overlap at high resolution:
    ylr_lr = X_lr[(over_lr == 1)]
    xlr_lr = Y_lr[(over_lr == 1)]
    coordlr_lr = (xlr_lr, ylr_lr)
    # Coordinates of low resolution pixels in the overlap at low resolution:
    ylr_hr = X_hr[(over_lr == 1)]
    xlr_hr = Y_hr[(over_lr == 1)]
    coordlr_hr = (xlr_hr, ylr_hr)


    return mask, coordlr_lr, coordlr_hr


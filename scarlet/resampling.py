import autograd.numpy as np
from . import interpolation


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

    # Coordinates of low resolution pixels in the overlap at low resolution:
    ylr_over_lr = X_lr[(over_lr == 1)]
    xlr_over_lr = Y_lr[(over_lr == 1)]
    coordlr_over_lr = (xlr_over_lr, ylr_over_lr)
    # Coordinates of low resolution pixels in the overlap at high resolution:
    ylr_over_hr = X_hr[(over_lr == 1)]
    xlr_over_hr = Y_hr[(over_lr == 1)]
    coordlr_over_hr = (xlr_over_hr, ylr_over_hr)

    return mask, coordlr_over_lr, coordlr_over_hr


def match_psfs(psf_hr, psf_lr, wcs_hr, wcs_lr):
    '''psf matching between different dataset

    Matches PSFS at different resolutions by interpolating psf_lr on the same grid as psf_hr

    Parameters
    ----------
    psf_hr: array
        centered psf of the high resolution scene
    psf_lr: array
        centered psf of the low resolution scene
    wcs_hr: WCS object
        wcs of the high resolution scene
    wcs_lr: WCS object
        wcs of the low resolution scene
    Returns
    -------
    psf_match_hr: array
        high rresolution psf at mactching size
    psf_match_lr: array
        low resolution psf at matching size and resolution
    '''

    ny_hr, nx_hr = psf_hr.shape
    ny_lr, nx_lr = psf_lr.shape

    psf_wcs_hr = wcs_hr.deepcopy()
    psf_wcs_lr = wcs_lr.deepcopy()
    if np.size(wcs_hr.array_shape) == 2:
        psf_wcs_hr.wcs.crval = 0., 0.
        psf_wcs_hr.wcs.crpix = ny_hr / 2., nx_hr / 2.
    elif np.size(psf_wcs_hr.array_shape) == 3:
        psf_wcs_hr.wcs.crval = 0., 0., 0.
        psf_wcs_hr.wcs.crpix = ny_hr / 2., nx_hr / 2., 0.
    if np.size(psf_wcs_lr.array_shape) == 2:
        psf_wcs_lr.wcs.crval = 0., 0.
        psf_wcs_lr.wcs.crpix = ny_lr / 2., nx_lr / 2.
    elif np.size(psf_wcs_lr.array_shape) == 3:
        psf_wcs_lr.wcs.crval = 0., 0., 0.
        psf_wcs_lr.wcs.crpix = ny_lr / 2., nx_lr / 2., 0

    mask, p_lr, p_hr = match_patches(psf_hr.shape, psf_lr.data.shape, psf_wcs_hr, psf_wcs_lr)

    cmask = np.where(mask == 1)

    n_p = np.int((np.size(cmask[0])) ** 0.5)
    psf_match_lr = interpolation.sinc_interp(cmask, p_hr[::-1], (psf_lr).flatten()).reshape(n_p, n_p)

    psf_match_hr = psf_hr[np.int((ny_hr - n_p) / 2):np.int((ny_hr + n_p) / 2),
                   np.int((nx_hr - n_p) / 2):np.int((nx_hr + n_p) / 2)]

    psf_match_hr /= np.max(psf_match_hr)
    psf_match_lr /= np.max(psf_match_lr)
    return psf_match_hr, psf_match_lr

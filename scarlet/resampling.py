import numpy as np


def get_to_common_frame(obs, frame_wcs):
    """ Matches an `Observation`'s coordinates to a `Frame`'s wcs

    Parameters
    ----------
    obs: `Observation`
        An observation instance for which we want to know the coordinates in the frame of `frame_wcs`
    frame_wcs: `WCS`
        a wcs that gives the mapping between pixel coordinates and sky coordinates for a given frame
    Returns
    -------
        coord: `array`
            Coordinates of the observations's pixels in the frame of the provided wcs
    """
    c, ny, nx = obs.frame.shape
    #Positions of the observation's pixels
    y, x = np.indices((ny, nx))
    y = y.flatten()
    x = x.flatten()

    # Ra-Dec positions of the observation's pixels
    if np.size(obs.frame.wcs.array_shape) == 2:
        ra, dec = obs.frame.wcs.all_pix2world(x, y, 0, ra_dec_order=True)
    elif np.size(obs.frame.wcs.array_shape) == 3:
        ra, dec = obs.frame.wcs.all_pix2world(x, y, 0, 0, ra_dec_order=True)

    # Positions of Observation's pixel in the frame of the wcs
    if np.size(frame_wcs.array_shape) == 2:
        X, Y = frame_wcs.all_world2pix(ra, dec, 0, ra_dec_order=True)
    elif np.size(frame_wcs.array_shape) == 3:
        X, Y, _ = frame_wcs.all_world2pix(ra, dec, 0, 0, ra_dec_order=True)

    coord = (Y,X)
    return coord






def match_patches(shape_hr, shape_lr, wcs_hr, wcs_lr, isrot = True, coverage  = 'union'):
    """Matches datasets at different resolutions


    Finds the region of intersection between two datasets and creates a mask for the region as well as the pixel coordinates
    for the dataset pixels inside the intersection.

    Parameters
    ----------
    shape_hr, shape_lr: tuples
        shapes of the two datasets
    wcs_hr, wcs_lr: WCS objects
        WCS of the Low and High resolution fields respectively
    coverage: string
        returns the coordinates in the intersection or union of both frames if set to 'intersection' or 'union' respectively

    Returns
    -------
    coordlr_over_lr: array
        coordinates of the matching pixels at low resolution in the low resolution frame.
    coordlr_over_hr: array
        coordinates of the matching pixels at low resolution in the high resolution frame.
    coordhr_hr: array
        coordinates of the high resolution pixels in the intersection. Necessary for psf matching
    """
    assert coverage in [
        "intersection",
        "union",
    ], "coverage should be either intersection or union."



    B_hr, ny_hr, nx_hr = shape_hr

    B_lr, Ny_lr, Nx_lr = shape_lr

    assert wcs_hr != None
    assert wcs_lr != None

    y_hr, x_hr = np.array(range(ny_hr)), np.array(range(nx_hr))

    # Capital letters are for coordinates of low-resolution pixels
    if isrot:

        # Coordinates of all low resolution pixels. All are needed if frames are rotated.
        Y_lr, X_lr = np.indices((Ny_lr, Nx_lr))

        X_lr = X_lr.flatten()
        Y_lr = Y_lr.flatten()

    else:
        Y_lr, X_lr = np.array(range(Ny_lr)), np.array(range(Nx_lr))

    # Corresponding angular positions
    # of low resolution pixels
    if np.size(wcs_lr.array_shape) == 2:
        ra_lr, dec_lr = wcs_lr.all_pix2world(X_lr, Y_lr, 0, ra_dec_order=True)
    elif np.size(wcs_lr.array_shape) == 3:
        ra_lr, dec_lr = wcs_lr.all_pix2world(X_lr, Y_lr, 0, 0, ra_dec_order=True)
    # of high resolution pixels
    if np.size(wcs_hr.array_shape) == 2:
        ra_hr, dec_hr = wcs_hr.all_pix2world(x_hr, y_hr, 0, ra_dec_order=True)
    elif np.size(wcs_hr.array_shape) == 3:
        ra_hr, dec_hr, _ = wcs_hr.all_pix2world(x_hr, y_hr, 0, 0, ra_dec_order=True)

    # Coordinates of the low resolution pixels in the high resolution frame
    if np.size(wcs_hr.array_shape) == 2:
        X_hr, Y_hr = wcs_hr.all_world2pix(ra_lr, dec_lr, 0, ra_dec_order=True)
    elif np.size(wcs_hr.array_shape) == 3:
        X_hr, Y_hr, _ = wcs_hr.all_world2pix(ra_lr, dec_lr, 0, 0, ra_dec_order=True)

    # Coordinates of the high resolution pixels in the low resolution frame
    if np.size(wcs_lr.array_shape) == 2:
        x_lr, y_lr = wcs_lr.all_world2pix(ra_hr, dec_hr, 0, ra_dec_order=True)
    elif np.size(wcs_lr.array_shape) == 3:
        x_lr, y_lr, _ = wcs_lr.all_world2pix(ra_hr, dec_hr, 0, 0, ra_dec_order=True)

    # mask of low resolution pixels at high resolution in the intersection:
    over_lr = (X_hr >= 0) * (X_hr < nx_hr + 1) * (Y_hr >= 0) * (Y_hr < ny_hr + 1)
    # mask of high resolution pixels at low resolution in the intersection (needed for psf matching)
    over_hr = (x_lr >= 0) * (x_lr < Nx_lr + 1) * (y_lr >= 0) * (y_lr < Ny_lr + 1)

    # pixels of the high resolution frame in the intersection in high resolution frame (needed for PSF only)
    coordhr_hr = (y_hr[(over_hr == 1)], x_hr[(over_hr == 1)])


    class SourceInitError(Exception):
        """
        Datasets do not match, no intersection found. Check the coordinates of the observations or the WCS.
        """

        pass

    if np.sum(over_lr) == 0:
        raise SourceInitError

    if coverage is "intersection":
        # Coordinates of low resolution pixels in the intersection at low resolution:
        ylr_lr = Y_lr[(over_lr == 1)]
        xlr_lr = X_lr[(over_lr == 1)]
        coordlr_lr = (ylr_lr, xlr_lr)
        # Coordinates of low resolution pixels in the intersection at high resolution:
        ylr_hr = Y_hr[(over_lr == 1)]
        xlr_hr = X_hr[(over_lr == 1)]

        coordlr_hr = (ylr_hr, xlr_hr)

    elif coverage is "union":

        # Coordinates of low resolution pixels at low resolution:
        coordlr_lr = (Y_lr, X_lr)

        # Coordinates of low resolution pixels at high resolution:
        coordlr_hr = (Y_hr, X_hr)

    return coordlr_lr, coordlr_hr, coordhr_hr

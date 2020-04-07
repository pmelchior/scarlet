import numpy as np

def _pix2radec(coord, wcs):
    """Converts coordinates from pixels to Ra-Dec given a wcs
    """
    y,x = coord
    if np.size(wcs.array_shape) == 2:
        ra, dec = wcs.all_pix2world(x, y, 0, ra_dec_order=True)
    elif np.size(wcs.array_shape) == 3:
        ra, dec = wcs.all_pix2world(x, y, 0, 0, ra_dec_order=True)
    return (ra, dec)

def _radec2pix(coord, wcs):
    """Converts coordinates from Ra-Dec to pixels given a wcs
    """
    ra, dec = coord
    # Positions of coords  in the frame of the obs
    if np.size(wcs.array_shape) == 2:
        X, Y = wcs.all_world2pix(ra, dec, 0, ra_dec_order=True)
    elif np.size(wcs.array_shape) == 3:
        X, Y, _ = wcs.all_world2pix(ra, dec, 0, 0, ra_dec_order=True)
    return (Y, X)

def convert_coordinates(coord, origin_wcs, target_wcs):
    """Converts coordinates from one reference frame to another
    Parameters
    ----------
    coord: `tuple`
        coordinates in the frame of the `origin_wcs` to convert in the frame of the `target_wcs`
    origin_wcs: WCS
        wcs of `coord`
    target_wcs: WCS
        wcs of the frame to which coord is converted

    Returns
    -------
    coord_target: `tuple`
        coordinates at the location of `coord` in the target frame defined by `target_wcs`
    """
    ra, dec = _pix2radec(coord, origin_wcs)
    y,x = _radec2pix((ra, dec), target_wcs)
    return (y, x)

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
        coord_obs_frame: `tuple`
            Coordinates of the observations's pixels in the frame of the provided wcs
    """
    c, ny, nx = obs.frame.shape
    #Positions of the observation's pixels
    y, x = np.indices((ny, nx))
    y = y.flatten()
    x = x.flatten()

    # Positions of Observation's pixel in the frame of the wcs
    Y,X = convert_coordinates((y,x), obs.frame.wcs, frame_wcs)

    coord = (Y,X)
    return coord




def match_patches(obs, frame, isrot = True):
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
    B_lr, Ny_lr, Nx_lr = obs.frame.shape

    wcs_hr = frame.wcs
    wcs_lr = obs.frame.wcs

    assert wcs_hr != None
    assert wcs_lr != None

    # Capital letters are for coordinates of low-resolution pixels
    if isrot:

        # Coordinates of all low resolution pixels. All are needed if frames are rotated.
        Y_lr, X_lr = np.indices((Ny_lr, Nx_lr))

        X_lr = X_lr.flatten()
        Y_lr = Y_lr.flatten()

    else:
        # Handling rectangular scenes. I make the grid square so that wcs can be applied
        N_lr = (Ny_lr != Nx_lr) * np.max([Ny_lr, Nx_lr]) + (Nx_lr == Ny_lr) * Ny_lr
        Y_lr, X_lr = np.array(range(N_lr)), np.array(range(N_lr))

    # Corresponding angular positions
    # Coordinates of the low resolution pixels in the high resolution frame
    Y_hr, X_hr = convert_coordinates((Y_lr, X_lr), wcs_lr, wcs_hr)


    # Coordinates of low resolution pixels in the intersection at low resolution:
    coordlr_lr = (Y_lr, X_lr)
    # Coordinates of low resolution pixels in the intersection at high resolution:
    coordlr_hr = (Y_hr, X_hr)
    return coordlr_lr, coordlr_hr

import numpy as np


def convert_coordinates(coord, origin, target):
    """Converts coordinates from one reference frame to another

    Parameters
    ----------
    coord: `tuple`
        pixel coordinates in the frame of the `origin`
    origin: `~scarlet.Frame`
        origin frame
    target: `~scarlet.Frame`
        target frame

    Returns
    -------
    coord_target: `tuple`
        coordinates at the location of `coord` in the target frame
    """
    pix = np.stack(coord, axis=1)
    ra_dec = origin.get_sky_coord(pix)
    yx = target.get_pixel(ra_dec)
    return yx[:, 0], yx[:, 1]


def get_to_common_frame(origin, target):
    """ Converts all pixels from `origin` to their position in `target`.

    Parameters
    ----------
    origin: `~scarlet.Frame`
        origin frame
    frame: `~scarlet.Frame`
        target frame

    Returns
    -------
        coord_obs_frame: `tuple`
            Pixel coordinates of the observations in the target frame
    """
    c, ny, nx = origin.shape
    # Positions of the observation's pixels
    y, x = np.indices((ny, nx))

    # Positions of Observation's pixel in the frame of the wcs
    Y, X = convert_coordinates((y, x), origin, target)
    return Y, X


def match_patches(obs, frame, isrot=True):
    """Matches datasets at different resolutions


    Finds the region of intersection between two datasets and creates a mask for the region as well as the pixel coordinates
    for the dataset pixels inside the intersection.

    Parameters
    ----------
    obs: `~scarlet.Observation`
        An observation instance for which we want to know the coordinates in the frame of `frame`
    frame: `~scarlet.Frame`
        target frame
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
    assert obs.frame.wcs is not None
    assert frame.wcs is not None

    B_lr, Ny_lr, Nx_lr = obs.frame.shape

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
    Y_hr, X_hr = convert_coordinates((Y_lr, X_lr), obs.frame, frame)

    # Coordinates of low resolution pixels in the intersection at low resolution:
    coordlr_lr = (Y_lr, X_lr)
    # Coordinates of low resolution pixels in the intersection at high resolution:
    coordlr_hr = (Y_hr, X_hr)
    return coordlr_lr, coordlr_hr

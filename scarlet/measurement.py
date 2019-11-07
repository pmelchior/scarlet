import numpy as np


def max_pixel(morph, center=None, window=None):
    """Use the pixel with the maximum flux as the center

    In case there is a nearby bright neighbor, we only update
    the center within the immediate vascinity of the previous center.
    This allows the center to shift over time, but prevents a large,
    likely unphysical update.

    Parameters
    ----------
    window: tuple of slices
        Slices in y and x of the central region to include in the fit.
        If `window` is `None` then only the 3x3 grid of pixels centered
        on the previous center are used. If it is desired to use the entire
        morphology just set `window=(slice(None), slice(None))`.
    """
    if center is None:
        shape = morph.shape
        center = (shape[0]//2, shape[1]//2)
    cy, cx = np.int(center[0]), np.int(center[1])

    if window is None:
        ymax = morph.shape[0] - 1
        xmax = morph.shape[1] - 1
        window = (slice(np.max([cy-2, 0]), np.min([cy+3, ymax])),
                  slice(np.max([cx-2, 0]), np.max([cx+3, xmax])))

    _morph = morph[window]
    yx0 = np.array([window[0].start, window[1].start])
    return tuple(np.unravel_index(np.argmax(_morph), _morph.shape) + yx0)


def psf_weighted_centroid(morph, psf, pixel_center):
    """Calculate the fraction position of a component

    Since our models should be relatively noise free it should be possible
    to estimate the fractional pixel offset of a `Componet.morph` by
    calculating the center of flux (centroid).
    This method uses the frame PSF, if available, otherwise a narrow gaussian
    is chosen such that that the flux is weighted to give more strength to
    pixels closer to the `pixel_center`.
    """
    cy, cx = pixel_center

    # Determine the width of the psf
    psf_shape_y, psf_shape_x = psf.shape
    # These are all the same value because psf are square and odd, but using
    # multiple variable names to clarify things
    x_rad = y_rad = psf_peak_y = psf_peak_x = psf_shape_x//2

    # Determine the overlapping coordinates of the morphology image and the psf
    # this is needed if the psf image, centered at the peak pixel location goes
    # off the edge of the psf array
    y_morph_ext = np.arange(morph.shape[0])
    x_morph_ext = np.arange(morph.shape[1])

    # calculate the offset in the morph frame such that the center pixel aligns
    # with the psf coordindate frame, where the psf peak is in the center
    y_morph_ext = y_morph_ext - cy
    x_morph_ext = x_morph_ext - cx

    # Compare the two end points, and take whichever is the smaller radius
    # use that to select the entries that are equal to or below that radius
    # Find the minimum between the end points (if the psf goes off the edge,
    # and the radius of the psf
    rad_endpoints_y = np.min((abs(y_morph_ext[0]), abs(y_morph_ext[-1]), y_rad))
    rad_endpoints_x = np.min((abs(x_morph_ext[0]), abs(x_morph_ext[-1]), x_rad))
    trimmed_y_range = y_morph_ext[abs(y_morph_ext) <= rad_endpoints_y]
    trimmed_x_range = x_morph_ext[abs(x_morph_ext) <= rad_endpoints_x]

    psf_y_range = trimmed_y_range + psf_peak_y
    psf_x_range = trimmed_x_range + psf_peak_x

    morph_y_range = trimmed_y_range + cy
    morph_x_range = trimmed_x_range + cx

    # Use these arrays to get views of the morphology and psf
    morph_view = morph[morph_y_range][:, morph_x_range]
    psf_view = psf[psf_y_range][:, psf_x_range]

    morph_view_weighted = morph_view*psf_view
    morph_view_weighted_sum = np.sum(morph_view_weighted)
    # build the indices to use the the centroid calculation
    indy, indx = np.indices(psf_view.shape)
    first_moment_y = np.sum(indy*morph_view_weighted) / morph_view_weighted_sum
    first_moment_x = np.sum(indx*morph_view_weighted) / morph_view_weighted_sum
    # build the offset to go from psf_view frame to psf frame to morph frame
    # aka move the peak back by the radius of the psf width adusted for the
    # minimum point in the view
    offset = (morph_y_range[0], morph_x_range[0])

    whole_pixel_center = np.round((first_moment_y, first_moment_x))
    dy, dx = whole_pixel_center - (first_moment_y, first_moment_x)
    morph_pixel_center = tuple((whole_pixel_center + offset).astype(int))
    return morph_pixel_center, (dy, dx)


def threshold(morph):
    """Find the threshold value for a given morphology
    """
    _morph = morph[morph > 0]
    _bins = 50
    # Decrease the bin size for sources with a small number of pixels
    if _morph.size < 500:
        _bins = max(np.int(_morph.size/10), 1)
        if _bins == 1:
            return 0, _bins
    hist, bins = np.histogram(np.log10(_morph).reshape(-1), _bins)
    cutoff = np.where(hist == 0)[0]
    # If all of the pixels are used there is no need to threshold
    if len(cutoff) == 0:
        return 0, _bins
    return 10**bins[cutoff[-1]], _bins

import numpy as np
from scipy.special import erfc


from ..bbox import overlapped_slices, Box
from ..initialization import get_minimal_boxsize


def insert_image(image_box, sub_box, sub_image, fill=0, dtype=None):
    """Insert an image given both an image box and sub-image box

    Parameters
    ----------
    image_box: `scarlet.bbox.Box`
        The bounding box that will contain the full image.
    sub_box: `scarlet.bbox.Box`
        The bounding box of the sub-image that is inserted into the full image.
    sub_image: `numpy.ndarray`
        The image that will be inserted.
    dtype: `numpy.dtype`
        The dtype of the resulting image.
    fill: `float`
        The fill value of the image for pixels outside of `sub_image`.

    Results
    -------
    image: `numpy.ndarray`
        An image of `fill` with the pixels in `sub_box` replaced
        with `sub_image`.
    """
    if dtype is None:
        dtype = sub_image.dtype
    if fill != 0:
        image = np.full(image_box.shape, fill, dtype=dtype)
    else:
        image = np.zeros(image_box.shape, dtype=dtype)
    slices = overlapped_slices(image_box, sub_box)
    image[slices[0]] = sub_image[slices[1]]
    return image

def project_morph_to_center(morph, center, bbox, fullbox, boxsize=None):
    """Project an uncentered morphology into a box that is centered on it

    Since most astrophysical sources are roughly symmetric,
    assuming that there will be an equal area of flux on opposing
    sides of the center is true in most cases.
    Projecting the morphology to the center of a square box
    makes it easier to update the flux without resizing with
    a minimal waste of memory.

    Parameters
    ----------
    morph: `numpy.ndarray`
        The 2D morphology that is to be centered.
    center: `list` of `int`
        The center pixel of `morph` in `fullbox`.
    bbox: `scarlet.bbox.Box`
        The bounding box of `morph`.
    `fullbox`: `scarlet.bbox.Box`
        The bounding box of the full `image` in which
        `center` describes the center of the source.
    `boxsize`: int
        The size of the centered morphology.
        If `boxsize` is `None` then the minimal box needed to
        contain the centered morphology with an odd number of
        pixels in x and y (so that there is a center) is used.

    Returns
    -------
    centered: `numpy.ndarray`
        The centered morphology.
    centered_box: `scarlet.bbox.Box`
        The bounding box that contains the centered morphology
        in coordinates of the `fullbaox`.
    """
    # find fitting bbox
    if bbox.contains(center):
        size = 2 * max(
            (
                center[0] - bbox.start[-2],
                bbox.stop[0] - center[-2],
                center[1] - bbox.start[-1],
                bbox.stop[1] - center[-1],
            )
        )
    else:
        size = 0

    # define new box and cut morphology accordingly
    if boxsize is None:
        boxsize = get_minimal_boxsize(size)

    bottom = center[0] - boxsize // 2
    top = center[0] + boxsize // 2 + 1
    left = center[1] - boxsize // 2
    right = center[1] + boxsize // 2 + 1
    centered_box = Box.from_bounds((bottom, top), (left, right))

    centered = np.zeros(centered_box.shape, dtype=morph.dtype)
    slices = overlapped_slices(centered_box, fullbox)
    centered[slices[0]] = morph[slices[1]]

    return centered, centered_box


def integrated_gaussian(X, sigma):
    """A Gaussian function evaluated at `X`

    Parameters
    ----------
    X: `numpy.ndarray`
        The coordinates to evaluate the integrated Gaussian.
    sigma: `float`
        The standard deviation of the Gaussian.

    Returns
    -------
    gaussian: `numpy.ndarray`
        A Gaussian function integrated over `X`
    """
    sqrt2 = np.sqrt(2)
    lhs = erfc((0.5 - X) / (sqrt2 * sigma))
    rhs = erfc((2 * X + 1) / (2 * sqrt2 * sigma))
    return np.sqrt(np.pi / 2) * sigma * (1-lhs + 1-rhs)


def integrated_circular_gaussian(X=None, Y=None, sigma=0.8):
    """Create a circular Gaussian that is integrated over pixels

    This is typically used for the model PSF,
    working well with the default parameters.

    Parameters
    ----------
    X, Y: `numpy.ndarray`
        The x,y-coordinates to evaluate the integrated Gaussian.
        If `X` and `Y` are `None` then they will both be given the
        default value `numpy.arange(-7, 8)`, resulting in a
        `15x15` centered image.
    sigma: `float`
        The standard deviation of the Gaussian.

    Returns
    -------
    image: `numpy.ndarray`
        A Gaussian function integrated over `X` and `Y`.
    """
    if X is None:
        if Y is None:
            X = np.arange(-7, 8)
            Y = X
        else:
            raise Exception(
                f"Either X and Y must be specified, or neither must be specified, got X={X} and Y={Y}")
    result = integrated_gaussian(X, sigma)[None, :] * integrated_gaussian(Y, sigma)[:, None]
    return result/np.sum(result)


def get_circle_mask(diameter, dtype=np.float64):
    """Get a boolean image of a circle

    Parameters
    ----------
    diameter: `int`
        The diameter of the circle and width
        of the image.
    dtype: `numpy.dtype`
        The `dtype` of the image.

    Returns
    -------
    circle: `numpy.ndarray`
        A boolean array with ones for the
        inside of the circle and zeros
        outside of the circle.
    """
    c = (diameter-1) / 2
    # The center of the circle and its radius are
    # off by half a pixel for circles with
    # even numbered diameter
    if diameter % 2 == 0:
        r = (diameter)/2
    else:
        r = c
    X = np.arange(diameter)
    X, Y = np.meshgrid(X,X)
    R = np.sqrt((X-c)**2 + (Y-c)**2)

    circle = np.ones((diameter, diameter), dtype=dtype)
    circle[R>r] = 0
    return circle

from functools import partial

import numpy as np
from proxmin.operators import prox_plus, prox_hard, prox_soft

from . import interpolation
from . import operator
from .bbox import trim
from .cache import Cache


def _find_yx(coeffs1, coeffs2, signs, center, err=1e-15):
    """Find the fractional pixels shifts dx and dy

    Given the location of two symmetric pairs of pixels, find the correct
    dx,dy to cause each symmetric pair to have the same value.

    This function will return `None` if neither of the solutions are valid
    """
    cy, cx = center
    # Equation for pixel pair one
    a, b, c, d = coeffs1
    # Equation for pixel pair 2
    A, B, C, D = coeffs2
    sign_y, sign_x = signs
    # Combine the coefficients for each pair to solve for
    # alpha y**2 + beta * y + gamma = 0
    alpha = a*C - A*c
    beta = a*D + b*C - A*d - B*c
    gamma = b*D - B*d

    # The solution must be real
    if beta**2 < 4*alpha*gamma:
        return None

    sqrt = np.sqrt(beta**2 - 4*alpha*gamma)

    y1 = 0.5*(-beta + sqrt)/alpha
    y2 = 0.5*(-beta - sqrt)/alpha

    x1 = -(C*y1+D)/(A*y1+B)
    x2 = -(C*y2+D)/(A*y2+B)

    def check(y, x):
        """Check whether or not the solution is valid
        """
        if abs(y) > 1 or abs(x) > 1:
            return False
        if (np.sign(y) != sign_y and abs(y) > err) or (np.sign(x) != sign_x and abs(y) > err):
            return False
        return True

    if check(y1, x1):
        return cy-y1, cx-x1
    elif check(y2, x2):
        return cy-y2, cx-x2
    # Neither of the solutions were valid
    return None


class RecenteringError(Exception):
    pass


def _linear_fit_center(morph, pixel_center, references=4):
    """Fit the center using bilinear interpolation

    While this interpolation has larger errors than
    a Lanczos kernel or polynomial spline, the solution
    for dx and dy can be determined analytically, making
    this a very fast algorithm.

    The solution is basically
        a1*dx*dy + b1*dx + c1*dy + d1 = a2*dx*dy + b2*dx + c2*dy + d1
    where a1, b1, c1, d1 are the coefficients of one pixels and
    a2, b2, c2, d2 are the coefficients of it's symmetric partner.
    So if a = a1-a2, b=b1-2, etc. then the equation for each pair
    of pixels is
        a*dx*dy + b*dx + c*dy + d = 0,
    and with two pairs of pixels (for example the pixels to the left,
    right, bottom, and top of the central pixel) it is possible to find
    dx and dy analytically. Unfortunately this equation is quadratic, so
    there are two possible solutions. A further complication is that the
    coefficents are different depending on whether dx and dy are positive
    or negative, so we have to compute at most 8 different solutions.
    Of the 8 possible solutions, there will only be one with `abs(dx)<1`,
    `abs(dy)<1`, and the sign of `dx` and `dy` that is the same as the solution.

    Note: in theory we should be able to write down a similar equation for the
    cubic and quintic equations, but hopefully the bilinear solution is a good
    enough solution to avoid the trouble.
    """
    def pxpy(x):
        """dx>=0, dy>=0
        """
        a = x[4] - x[3] - x[1] + x[0]
        b = -x[4] + x[3]
        c = -x[4] + x[1]
        d = x[4]
        return np.array([a, b, c, d]), (1, 1)

    def mxpy(x):
        """dx<0, dy>=0
        """
        a = -x[4] + x[5] + x[1] - x[2]
        b = x[4] - x[5]
        c = -x[4] + x[1]
        d = x[4]
        return np.array([a, b, c, d]), (1, -1)

    def pxmy(x):
        """dx>=0, dy<0
        """
        a = -x[4] + x[3] + x[7] - x[6]
        b = -x[4] + x[3]
        c = x[4] - x[7]
        d = x[4]
        return np.array([a, b, c, d]), (-1, 1)

    def mxmy(x):
        """dx<0, dy<0
        """
        a = x[4] - x[5] - x[7] + x[8]
        b = x[4] - x[5]
        c = x[4] - x[7]
        d = x[4]
        return np.array([a, b, c, d]), (-1, -1)

    def fit_coeffs(points, func):
        coeffs = []
        for point in points:
            py, px = point
            block = morph[cy+py-1:cy+py+2, cx+px-1: cx+px+2].reshape(-1)
            coeff, signs = func(block)
            coeffs.append(coeff)
        coeffs01 = coeffs[0] - coeffs[1]
        coeffs23 = coeffs[2] - coeffs[3]
        return _find_yx(coeffs01, coeffs23, signs, pixel_center)

    cy, cx = pixel_center
    sets = (
        ((0, -1), (0, 1),  # left, right
         (-1, 0), (1, 0)),  # bottom, top
        ((-1, -1), (1, 1),
         (-1, 1), (1, -1)),
        ((0, -2), (0, 2),
         (-2, 0), (2, 0)),
        ((-2, -2), (2, 2),
         (-2, 2), (2, -2)),
    )
    # Get the relevent vectors for each pixel
    # Check all four possible signs for x and y
    my_func = None
    points = sets[0]
    results = []
    weights = []
    for func in [pxpy, mxpy, pxmy, mxmy]:
        result = fit_coeffs(points, func)
        if result is not None:
            weights.append(np.sum(np.array(points)**2))
            results.append(result)
            my_func = func
            break
    if my_func is not None:
        for points in sets[1:references]:
            result = fit_coeffs(points, my_func)
            if result is not None:
                weights.append(np.sum(np.array(points)**2))
                results.append(result)
        results = np.array(results)
        weights = 1/np.array(weights)
        result = np.sum(results * weights[:, None], axis=0)/weights.sum()
        return tuple(result)

    # If none of the solutions are correct raise an error
    # (that could possibly be caught and disable positon updates
    # for the gien source).
    raise RecenteringError("scarlet failed to properly update the center of the source")


def symmetric_fit_center(component, update_interval=5, window=None):
    """Fit the center of an object based on symmetry

    This algorithm interpolates the morphology until the pixels on the
    left and right of the central pixel are equal and the top and
    bottom are equal.
    """
    # Only update at specified interval
    it = component._parent.it
    if (it-1) % update_interval:
        return component
    # Update the center pixel
    _fit_pixel_center(component.morph, component.pixel_center, window)
    center = _linear_fit_center(component.morph, component.pixel_center)
    component.float_center = center
    center_int = np.array((np.round(center[0]), np.round(center[1]))).astype(int)
    component.center_int = center_int
    dy, dx = center[0]-center_int[0], center[1]-center_int[1]
    if dy < 0:
        dy = 1+dy
    if dx < 0:
        dx = 1+dx
    component.shift = (-dy, -dx)
    return component


def _fit_pixel_center(morph, center, window=None):
    cy, cx = center
    if window is None:
        window = slice(cy-2, cy+3), slice(cx-2, cx+3)
    _morph = morph[window]
    yx0 = np.array([window[0].start, window[1].start])
    return tuple(np.unravel_index(np.argmax(_morph), _morph.shape) + yx0)


def fit_pixel_center(component, window=None):
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
    component.pixel_center = _fit_pixel_center(component.morph, component.pixel_center, window)
    return component


def positive_sed(component):
    """Make the SED non-negative
    """
    prox_plus(component.sed, component.step_sed)
    return component


def positive_morph(component):
    """Make the morphology non-negative
    """
    prox_plus(component.morph, component.step_morph)
    return component


def positive(component):
    """Make both the SED and morpholgy non-negative
    """
    prox_plus(component.sed, component.step_sed)
    prox_plus(component.morph, component.step_morph)
    return component


def normalized(component, type='morph_max'):
    """Normalize SED or morphology

    In order to break degeneracies, either the SED or
    morphology should have some form of normalization.
    For consistency this should normally be the same for
    all components in a blend.

    For `type='sed'` the sed matrix is normalized to
    sum to unity.

    For `type='morph'` the morphology matrix is normalized
    to sum to unity.

    For `type='morph_max'` the morphology matrix is
    normalized so that its maximum value is one.
    """
    assert type.lower() in ['sed', 'morph', 'morph_max']
    t = type.lower()

    if t == 'sed':
        norm = component.sed.sum()
        component.sed[:] = component.sed / norm
        component.morph[:] = component.morph * norm
    elif t == 'morph':
        norm = component.morph.sum()
        component.sed[:] = component.sed * norm
        component.morph[:] = component.morph / norm
    elif t == 'morph_max':
        norm = component.morph.max()
        component.sed[:] = component.sed * norm
        component.morph[:] = component.morph / norm
    else:
        raise ValueError("Unrecognized normalization '{0}'".format(type))


def sparse_l0(component, thresh):
    """L0 norm (sparsity) on morphology
    """
    prox_hard(component.morph, component.step_morph, thresh)
    return component


def sparse_l1(component, thresh):
    """L1 norm (sparsity) on morphology
    """
    prox_soft(component.morph, component.step_morph, thresh)
    return component


def _threshold(morph):
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


def threshold(component):
    """Set a cutoff threshold for pixels below the noise

    Use the log histogram of pixel values to determine when the
    source is fitting noise. This function works well to prevent
    faint sources from growing large footprints but for large
    diffuse galaxies with a wide range of pixel values this
    does not work as well.
    """
    thresh, _bins = _threshold(component.morph)
    component.morph[component.morph < thresh] = 0
    bbox = trim(component.morph)
    if not hasattr(component, "bboxes"):
        component.bboxes = {}
    component.bboxes["thresh"] = bbox
    return component


def monotonic(component, pixel_center, use_nearest=False, thresh=0, exact=False, bbox=None):
    """Make morphology monotonically decrease from the center

    Parameters
    ----------
    exact: `bool`
        Whether to use the (very slow) exact monotonicity proximal operator
        (`exact` is `True`) or a projection operator to the monotonic space.

    See `~scarlet.operator.prox_monotonic`
    for a description of the other parameters.
    """
    if bbox is not None:
        # Only apply monotonicity to the pixels inside the bounding box
        morph = component.morph[bbox.slices]
        shape = morph.shape
        if shape[0] <= 1 or shape[1] <= 1:
            return component
        center = pixel_center[0]-bbox.bottom, pixel_center[1]-bbox.left
    else:
        morph = component.morph
        shape = component.shape[-2:]
        center = pixel_center
    morph = morph.copy()

    prox_name = "update.monotonic"
    key = (shape, center, use_nearest, thresh, exact)
    # The creation of this operator is expensive,
    # so load it from memory if possible.
    try:
        prox = Cache.check(prox_name, key)
    except KeyError:
        if not exact:
            prox = operator.prox_strict_monotonic(shape, use_nearest=use_nearest,
                                                  thresh=thresh, center=center)
        else:
            # Exact monotonicy still needs to be tested and implmented with v0.5.
            raise NotImplementedError("Exact monotonicity is not currently supported")
            # cone method for monotonicity: exact but VERY slow
            G = operator.getRadialMonotonicOp(shape, useNearest=use_nearest).L.toarray()
            prox = partial(operator.prox_cone, G=G)
        Cache.set(prox_name, key, prox)

    step_size = component.step_morph
    prox(morph, step_size)
    if bbox is not None:
        component.morph[bbox.slices] = morph
    else:
        component.morph[:] = morph
    return component


def translation(component, direction=1, kernel=interpolation.lanczos, padding=3):
    """Shift the morphology by a given amount
    """
    dy, dx = component.shift
    dy *= direction
    dx *= direction
    _kernel, _, _ = interpolation.get_separable_kernel(dy, dx, kernel=kernel)
    component.morph[:] = interpolation.fft_resample(component.morph, dy, dx)
    return component


def symmetric(component, strength=1, use_prox=True, kernel=interpolation.lanczos, padding=3):
    """Make the source symmetric about its center

    See `~scarlet.operator.prox_uncentered_symmetry`
    for a description of the parameters.
    """
    step_size = component.step_morph
    center = component.pixel_center
    operator.prox_uncentered_symmetry(component.morph, step_size, center, strength, use_prox)
    return component

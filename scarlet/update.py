from functools import partial

import numpy as np
from proxmin.operators import prox_plus, prox_hard, prox_soft

from . import interpolation
from . import operator
from .cache import Cache


def _quintic_recenter_loss(params, image, center):
    """Loss function for recentering based on quintic interpolation
    """
    y0, x0 = center
    dy, dx = params
    window = np.arange(-3, 4)

    Ly, _ = interpolation.quintic_spline(-dy)
    Lx, _ = interpolation.quintic_spline(-dx)
    kernel = np.outer(Ly, Lx)

    ywin = window + y0
    xwin = window + x0

    left = image[ywin[0]: ywin[-1]+1, xwin[0]-1:xwin[-1]] * kernel
    right = image[ywin[0]: ywin[-1]+1, xwin[0]+1:xwin[-1]+2] * kernel
    bottom = image[ywin[0]-1: ywin[-1], xwin[0]:xwin[-1]+1] * kernel
    top = image[ywin[0]+1: ywin[-1]+2, xwin[0]:xwin[-1]+1] * kernel
    return (left.sum()-right.sum())**2, (top.sum()-bottom.sum())**2


def symmetric_fit_center(component, loss_func=_quintic_recenter_loss):
    """Fit the center of an object based on symmetry

    This algorithm interpolates the morphology until the pixels on the
    left and right of the central pixel are equal and the top and
    bottom are equal.

    Parameters
    ----------
    component: `scarlet.component.Component`
        The component to make symmetric

    loss_func: `function`
        Function to use to calculate the loss.
        This is basically a combination of an interpolation of the four
        nearest neighbors to the central pixel and a comparison of
        their values

    Returns
    -------
    component: `scarlet.component.Component`
        The same component with its center udpated.
    """
    from scipy.optimize import fsolve

    # Set the pixel center to the maximum pixel value of the morphology
    fit_pixel_center(component)
    center = component.pixel_center
    # Initally assume there is no fractional shift
    dyx = [0, 0]
    dyx = fsolve(loss_func, dyx, (component.morph, center))
    component.float_center = (center[0] + dyx[0], center[1] + dyx[1])
    return component


def fit_pixel_center(component):
    """Use the pixel with the maximum flux as the center
    """
    component.pixel_center = np.unravel_index(np.argmax(component.morph), component.morph.shape)
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


def monotonic(component, pixel_center, use_nearest=False, thresh=0, exact=False):
    """Make morphology monotonically decrease from the center

    Parameters
    ----------
    exact: `bool`
        Whether to use the (very slow) exact monotonicity proximal operator
        (`exact` is `True`) or a projection operator to the monotonic space.

    See `~scarlet.operator.prox_monotonic`
    for a description of the other parameters.
    """
    prox_name = "update.monotonic"
    shape = component.shape[-2:]
    key = (shape, pixel_center, use_nearest, thresh, exact)
    # The creation of this operator is expensive,
    # so load it from memory if possible.
    try:
        prox = Cache.check(prox_name, key)
    except KeyError:
        if not exact:
            prox = operator.prox_strict_monotonic(shape, use_nearest=use_nearest,
                                                  thresh=thresh, center=pixel_center)
        else:
            # Exact monotonicy still needs to be tested and implmented with v0.5.
            raise NotImplementedError("Exact monotonicity is not currently supported")
            # cone method for monotonicity: exact but VERY slow
            G = operator.getRadialMonotonicOp(shape, useNearest=use_nearest).L.toarray()
            prox = partial(operator.prox_cone, G=G)
        Cache.set(prox_name, key, prox)

    step_size = component.step_morph
    prox(component.morph, step_size)
    return component


def symmetric(component, center, strength=1, use_prox=True):
    """Make the source symmetric about its center

    See `~scarlet.operator.prox_uncentered_symmetry`
    for a description of the parameters.
    """
    step_size = component.step_morph
    operator.prox_uncentered_symmetry(component.morph, step_size, center, strength, use_prox)
    return component

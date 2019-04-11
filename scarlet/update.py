from enum import Enum
from functools import partial

import numpy as np
from proxmin.operators import prox_plus, prox_hard, prox_soft

from . import resample
from . import operator
from .cache import Cache


class Normalization(Enum):
    """Type of normalization to use for A and S

    Due to degeneracies in the product AS it is common to
    normalize one of the two matrices to unity. This
    enumerator is used to define which normalization is
    used to break this degeneracy.

    Attributes
    ----------
    A: 1
        Normalize the A (color) matrix to unity
    S: 2
        Normalize the S (morphology) matrix to unity
    Smax: 3
        Normalize S so that the maximum (peak) value is unity
    """
    A = 1
    S = 2
    Smax = 3


def quintic_recenter_loss(params, image, center):
    y0, x0 = center
    dy, dx = params
    window = np.arange(-3, 4)

    Ly, _ = resample.quintic_spline(-dy)
    Lx, _ = resample.quintic_spline(-dx)
    kernel = np.outer(Ly, Lx)

    ywin = window + y0
    xwin = window + x0

    left = image[ywin[0]: ywin[-1]+1, xwin[0]-1:xwin[-1]] * kernel
    right = image[ywin[0]: ywin[-1]+1, xwin[0]+1:xwin[-1]+2] * kernel
    bottom = image[ywin[0]-1: ywin[-1], xwin[0]:xwin[-1]+1] * kernel
    top = image[ywin[0]+1: ywin[-1]+2, xwin[0]:xwin[-1]+1] * kernel
    return (left.sum()-right.sum())**2, (top.sum()-bottom.sum())**2


def symmetric_fit_center(component, func=quintic_recenter_loss):
    from scipy.optimize import fsolve

    center = component.pixel_center
    dyx = [0, 0]
    component.coord = fsolve(func, dyx, (component.morph.detach().numpy(), center))
    return component


def fit_pixel_center(component):
    component.pixel_center = np.unravel_index(np.argmax(component.morph), component.morph.shape)
    return component


def positive_sed(component):
    prox_plus(component.sed.data)
    return component


def positive_morph(component):
    prox_plus(component.morph.data)
    return component


def positive(component):
    prox_plus(component.sed.data, component.L_sed)
    prox_plus(component.morph.data, component.L_morph)
    return component


def normalize(component, normalization=Normalization.Smax):
    if normalization == Normalization.A:
        operator.prox_unity(component.sed.data, component.L_sed)
    elif normalization == Normalization.S:
        operator.prox_unity(component.morph.data, component.L_morph)
    else:
        operator.prox_max(component.morph.data, component.L_morph)


def l0_norm(component, thresh):
    prox_hard(component.morph.data, 1/component.L_morph, thresh)
    return component


def l1_norm(component, thresh):
    prox_soft(component.morph.data, 1/component.L_morph, thresh)
    return component


def monotonic(component, center, use_nearest=False, thresh=0, exact=False):
    prox_name = "update.monotonic"
    shape = component.shape[-2:]
    key = (shape, center, use_nearest, thresh, exact)
    try:
        prox = Cache.check(prox_name, key)
    except KeyError:
        if not exact:
            prox = operator.prox_strict_monotonic(shape, use_nearest=use_nearest,
                                                  thresh=thresh, center=center)
        else:
            raise NotImplementedError("Exact monotonicity is not currently supported")
            # cone method for monotonicity: exact but VERY slow
            G = operator.getRadialMonotonicOp(shape, useNearest=use_nearest).L.toarray()
            prox = partial(operator.prox_cone, G=G)
        Cache.set(prox_name, key, prox)

    step_size = 1/component.L_morph
    prox(component.morph.data, step_size)
    return component


def symmetric(component, center, strength=1, use_prox=True):
    step_size = 1/component.L_morph
    operator.prox_uncentered_symmetry(component.morph.data, step_size, strength, use_prox, center)
    return component

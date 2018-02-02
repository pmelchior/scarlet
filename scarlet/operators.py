from __future__ import print_function, division
import logging
from functools import partial

import numpy as np
import proxmin

logger = logging.getLogger("scarlet.operators")

def _prox_strict_monotonic(X, step, ref_idx, dist_idx, thresh=0):
    """Force an intensity profile to be monotonic
    """
    from . import operators_pybind11
    operators_pybind11.prox_monotonic(X, step, ref_idx, dist_idx, thresh)
    return X

def _prox_weighted_monotonic(X, step, weights, didx, offsets, thresh=0):
    from . import operators_pybind11
    operators_pybind11.prox_weighted_monotonic(X, step, weights, offsets, didx, thresh)
    return X

def sort_by_radius(shape):
    """Sort indices distance from the center

    Given a shape, calculate the distance of each
    pixel from the center and return the indices
    of each pixel, sorted by radial distance from
    the center.

    Parameters
    ----------
    shape: tuple
        Shape (y,x) of the source frame.

    Returns
    -------
    didx: `~numpy.array`
        Indices of elements in an image with shape `shape`,
        sorted by distance from the center.
    ref_idx: `~numpy.array`
        Indices of the element in the image closest to the
        center for each pixel in didx.
    """
    # Get the center pixels
    cx = (shape[1]-1) >> 1
    cy = (shape[0]-1) >> 1
    # Calculate the distance between each pixel and the peak
    x = np.arange(shape[1])
    y = np.arange(shape[0])
    X,Y = np.meshgrid(x,y)
    X = X - cx
    Y = Y - cy
    distance = np.sqrt(X**2+Y**2)
    # Get the indices of the pixels sorted by distance from the peak
    didx = np.argsort(distance.flatten())
    return didx

def prox_strict_monotonic(shape, use_nearest=False, thresh=0):
    """Build the prox_monotonic operator
    """
    from . import transformations

    height, width = shape
    if not height % 2 or not width % 2:
        err = "Shape must have an odd width and height, received shape {0}".format(shape)
        raise ValueError(err)
    didx = sort_by_radius(shape)

    if use_nearest:
        from scipy import sparse
        monotonicOp = transformations.getRadialMonotonicOp(shape, useNearest=True)
        x_idx, ref_idx = sparse.find(monotonicOp==1)[:2]
        ref_idx = ref_idx[np.argsort(x_idx)]
        result = partial(_prox_strict_monotonic, ref_idx=ref_idx.tolist(),
                         dist_idx=didx.tolist(), thresh=thresh)
    else:
        coords = [(-1,-1), (-1,0), (-1, 1), (0,-1), (0,1), (1, -1), (1,0), (1,1)]
        offsets = np.array([width*y+x for y,x in coords])
        weights = transformations.getRadialMonotonicWeights(shape, useNearest=False)
        result = partial(_prox_weighted_monotonic, weights=weights, didx=didx[1:], offsets=offsets, thresh=thresh)
    return result

def prox_cone(X, step, G=None):
    """Exact projection of components of X onto cone defined by Gx >= 0"""
    k, n = X.shape
    for i in range(k):
        Y = X[i]

        # Creating set of half-space defining vectors
        Vs = []
        for j in range(0, n):
            add = G[j]
            Vs.append(add)
        Q = find_Q(Vs, n)

        # Finding and using relevant dimensions until a point on the cone is found
        for j in range(n):
            index = find_relevant_dim(Y, Q, Vs)
            if index != -1:
                Y, Q, Vs = use_relevant_dim(Y, Q, Vs, index)
            else:
                break
        X[i] = Y
    return X

def prox_center_on(X, step, tiny=1e-10):
    # make sure that the center pixel as at least some amount of flux
    # otherwise centering will go off rails
    center_pix = X.size // 2 # only works for odd pixel numbers in x and y
    X[center_pix] = max(X[center_pix], tiny)
    return X

def proj(A,B):
    """Returns the projection of A onto the hyper-plane defined by B"""
    return A - (A*B).sum()*B/(B**2).sum()

def proj_dist(A,B):
    """Returns length of projection of A onto B"""
    return (A*B).sum()/(B**2).sum()**0.5

def use_relevant_dim(Y, Q, Vs, index):
    """Uses relevant dimension to reduce problem dimensionality (projects everything onto the
    new hyperplane"""
    projector = Vs[index]
    del Vs[index]
    Y = proj(Y, projector)
    Q = proj(Y, projector)
    for i in range(len(Vs)):
        Vs[i] = proj(Vs[i], projector)
    return Y, Q, Vs

def find_relevant_dim(Y, Q, Vs):
    """Finds a dimension relevant to the problem by 'raycasting' from Y to Q"""
    max_t = 0
    index = -1
    for i in range(len(Vs)):
        Y_p = proj_dist(Y, Vs[i])
        Q_p = proj_dist(Q, Vs[i])
        if Y_p < 0:
            t = -Y_p/(Q_p - Y_p)
        else:
            t = -2
        if t > max_t:
            max_t = t
            index = i
    return index

def find_Q(Vs, n):
    """Finds a Q that is within the solution space that can act as an appropriate target
    (could be rigorously constructed later)"""
    res = np.zeros(n)
    res[int((n-1)/2)] = n
    return res

def strict_monotonicity(images, peaks=None, components=None, l0_thresh=None, l1_thresh=None, constraints="m"):
    """Use monotonicity as a strict proximal operator
    """
    import proxmin

    if components is None:
        component_count = len(peaks)
    else:
        component_count = np.sum([len(c) for c in components])
    B, N, M = images.shape

    if l0_thresh is None and l1_thresh is None:
        prox_S = proxmin.operators.prox_plus
    else:
        # L0 has preference
        if l0_thresh is not None:
            if l1_thresh is not None:
                logger.warn("l1_thresh ignored in favor of l0_thresh")
            prox_S = partial(proxmin.operators.prox_hard, thresh=l0_thresh)
        else:
            prox_S = partial(proxmin.operators.prox_soft_plus, thresh=l1_thresh)
    if isinstance(constraints, str):
        if constraints!="m":
            raise ValueError("Monotonicity 'm' is the only allowed strict constraint")
        seeks = [True]*component_count
    else:
        seeks = [constraints[k]=="m" for k in range(component_count)]
    prox_S = build_prox_monotonic(shape=(N,M), seeks=seeks, prox_chain=prox_S)
    return prox_S

def project_disk_sed_mean(bulge_sed, disk_sed):
    """Project the disk SED onto the space where it is bluer

    For the majority of observed galaxies, it appears that
    the difference between the bulge and the disk SEDs is
    roughly monotonic, making the disk bluer.

    This projection operator projects colors that are redder
    than other colors onto the average SED difference for
    that wavelength. This is a more accurate SED than
    `project_disk_sed` but is more likely to create
    discontinuities in the evaluation of A, and should
    probably be avoided. It is being kept for now to record
    its effect.
    """
    new_sed = disk_sed.copy()
    diff = bulge_sed - disk_sed
    slope = (diff[-1]-diff[0])/(len(bulge_sed)-1)
    for s in range(1, len(diff)-1):
        if diff[s]<diff[s-1]:
            new_sed[s] = bulge_sed[s] - (slope*s + diff[0])
            diff[s] = bulge_sed[s] - new_sed[s]
    return new_sed

def project_disk_sed(bulge_sed, disk_sed):
    """Project the disk SED onto the space where it is bluer

    For the majority of observed galaxies, it appears that
    the difference between the bulge and the disk SEDs is
    roughly monotonic, making the disk bluer.

    This projection operator projects colors that are redder onto
    the same difference in color as the previous wavelength,
    similar to the way monotonicity works for the morphological
    `S` matrix of the model.

    While a single iteration of this model is unlikely to yield
    results that are as good as those in `project_disk_sed_mean`,
    after many iterations it is expected to converge to a better value.
    """
    new_sed = disk_sed.copy()
    diff = bulge_sed - disk_sed
    for s in range(1, len(diff)-1):
        if diff[s]<diff[s-1]:
            new_sed[s] = new_sed[s] + diff[s-1]
            diff[s] = diff[s-1]
    return new_sed

def proximal_disk_sed(X, step, peaks, algorithm=project_disk_sed_mean):
    """Ensure that each disk SED is bluer than the bulge SED
    """
    for peak in peaks.peaks:
        if "disk" in peak.components and "bulge" in peak.components:
            bulge_k = peak["bulge"].index
            disk_k = peak["disk"].index
            X[:,disk_k] = algorithm(X[:,bulge_k], X[:,disk_k])
    X = proxmin.operators.prox_unity_plus(X, step, axis=0)
    return X

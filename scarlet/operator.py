from functools import partial

import autograd.numpy as np
from proxmin.operators import prox_unity_plus
from proxmin.utils import MatrixAdapter

from .cache import Cache


import logging
logger = logging.getLogger("scarlet.operator")


def prox_max_unity(X, step):
    """Normalize X so that it's max value is unity."""
    norm = X.max()
    X[:] = X/norm
    return X


def _prox_strict_monotonic(X, step, ref_idx, dist_idx, thresh=0):
    """Force an intensity profile to be monotonic based on nearest neighbor
    """
    from . import operators_pybind11
    operators_pybind11.prox_monotonic(X.reshape(-1), step, ref_idx, dist_idx, thresh)
    return X


def _prox_weighted_monotonic(X, step, weights, didx, offsets, thresh=0):
    """Force an intensity profile to be monotonic based on weighting neighbors
    """
    from . import operators_pybind11
    operators_pybind11.prox_weighted_monotonic(X.reshape(-1), step, weights, offsets, didx, thresh)
    return X


def sort_by_radius(shape, center=None):
    """Sort indices distance from the center

    Given a shape, calculate the distance of each
    pixel from the center and return the indices
    of each pixel, sorted by radial distance from
    the center, which need not be in the center
    of the image.

    Parameters
    ----------
    shape: `tuple`
        Shape (y,x) of the source frame.

    center: array-like
        Location of the center pixel.

    Returns
    -------
    didx: `~numpy.array`
        Indices of elements in an image with shape `shape`,
        sorted by distance from the center.
    """
    # Get the center pixels
    if center is None:
        cx = (shape[1]-1) >> 1
        cy = (shape[0]-1) >> 1
    else:
        cy, cx = int(center[0]), int(center[1])
    # Calculate the distance between each pixel and the peak
    x = np.arange(shape[1])
    y = np.arange(shape[0])
    X, Y = np.meshgrid(x, y)
    X = X - cx
    Y = Y - cy
    distance = np.sqrt(X**2+Y**2)
    # Get the indices of the pixels sorted by distance from the peak
    didx = np.argsort(distance.flatten())
    return didx


def prox_strict_monotonic(shape, use_nearest=False, thresh=0, center=None):
    """Build the prox_monotonic operator

    Parameters
    ----------
    use_nearest: `bool`
        Whether to use the nearest pixel to the center for comparison
        (`use_nearest=True`) or use a weighted combination of all
        neighbors closer to the central pixel (`use_nearest=False`).
    thresh: `float`
        Forced gradient. A `thresh` of zero will allow a pixel to be the
        same value as its reference pixels, while a `thresh` of one
        will force the pixel to zero.
    center: tuple
        Location of the central (highest-value) pixel.

    Returns
    -------
    result: `function`
        The monotonicity function.
    """
    height, width = shape
    didx = sort_by_radius(shape, center)

    if use_nearest:
        from scipy import sparse
        if thresh != 0:
            # thresh and nearest neighbors are not compatible, since this thresholds the
            # central pixel and eventually sets the entire array to zero
            raise ValueError("Thresholding does not work with nearest neighbor monotonicity")
        monotonicOp = getRadialMonotonicOp(shape, useNearest=True)
        x_idx, ref_idx = sparse.find(monotonicOp.L == 1)[:2]
        ref_idx = ref_idx[np.argsort(x_idx)]
        result = partial(_prox_strict_monotonic, ref_idx=ref_idx.tolist(),
                         dist_idx=didx.tolist(), thresh=thresh)
    else:
        coords = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        offsets = np.array([width*y+x for y, x in coords])
        weights = getRadialMonotonicWeights(shape, useNearest=False, center=center)
        result = partial(_prox_weighted_monotonic, weights=weights,
                         didx=didx[1:], offsets=offsets, thresh=thresh)
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
    """Ensure that the central pixel has positive flux

    Make sure that the center pixel as at least some amount of flux
    otherwise centering will go off rails
    """
    cy = X.shape[0] // 2
    cx = X.shape[1] // 2
    X[cy, cx] = max(X[cy, cx], tiny)
    return X


def prox_sed_on(X, step, tiny=1e-10):
    """Ensure that the SED has some flux.

    This is used when S is normalized and A is
    not to prevent a source from having no flux,
    which is known to break the centering algorithm.
    Once we put in a check to ensure that the difference
    image has a dipole this operator will be rendered unecessary
    """
    if np.all(X <= 0):
        X[:] = tiny
    return X


def uncentered_operator(X, func, center=None, fill=None, **kwargs):
    """Only apply the operator on a centered patch

    In some cases, for example symmetry, an operator might not make
    sense outside of a centered box. This operator only updates
    the portion of `X` inside the centered region.

    Parameters
    ----------
    X: array
        The parameter to update.
    func: `function`
        The function (or operator) to apply to `X`.
    center: tuple
        The location of the center of the sub-region to
        apply `func` to `X`.
    `fill`: `float`
        The value to fill the region outside of centered
        `sub-region`, for example `0`. If `fill` is `None`
        then only the subregion is updated and the rest of
        `X` remains unchanged.
    """
    if center is None:
        py, px = np.unravel_index(np.argmax(X), X.shape)
    else:
        py, px = center
    cy, cx = np.array(X.shape) // 2

    if py == cy and px == cx:
        return func(X, **kwargs)

    dy = int(2*(py-cy))
    dx = int(2*(px-cx))
    if not X.shape[0] % 2:
        dy += 1
    if not X.shape[1] % 2:
        dx += 1
    if dx < 0:
        xslice = slice(None, dx)
    else:
        xslice = slice(dx, None)
    if dy < 0:
        yslice = slice(None, dy)
    else:
        yslice = slice(dy, None)

    if fill is not None:
        _X = np.ones(X.shape, X.dtype) * fill
        _X[yslice, xslice] = func(X[yslice, xslice], **kwargs)
        X[:] = _X
    else:
        X[yslice, xslice] = func(X[yslice, xslice], **kwargs)

    return X


def prox_sdss_symmetry(X, step):
    """SDSS/HSC symmetry operator

    This function uses the *minimum* of the two
    symmetric pixels in the update.
    """
    Xs = np.fliplr(np.flipud(X))
    X[:] = np.min([X, Xs], axis=0)
    return X


def prox_soft_symmetry(X, step, strength=1):
    """Soft version of symmetry
    Using a `strength` that varies from 0 to 1,
    with 0 meaning no symmetry enforced at all and
    1  being completely symmetric, the user can customize
    the level of symmetry required for a component
    """
    Xs = np.fliplr(np.flipud(X))
    X[:] = 0.5 * strength * (X+Xs) + (1-strength) * X
    return X


def prox_kspace_symmetry(X, step, shift=None, padding=10):
    """Symmetry in Fourier Space

    This algorithm by Nate Lust uses the fact that throwing
    away the imaginary part in Fourier space leaves a symmetric
    soution in real space. So `X` is transformed to Fourier space,
    shifted by the fractional amount `shift=(dy, dx)`,
    the imaginary part is discarded, shited back to its original position,
    then transformed back to real space.
    """
    # Record the morph shape
    shape = X.shape
    dy, dx = shift
    padding = np.max(X.shape) + padding // 2
    edges = ((padding, padding), (padding, padding))
    corner = (padding, padding)
    zeroMask = X <= 0
    X = np.pad(X, edges, 'constant')

    freq_x = np.fft.fftfreq(X.shape[1])
    freq_y = np.fft.fftfreq(X.shape[0])

    # Transform to k space
    X_fft = np.fft.fftn(np.fft.ifftshift(X))

    # Shift the signal to recenter it, negative because math is opposite from
    # pixel direction
    shifter = np.outer(np.exp(-1j*2*np.pi*freq_y*-(dy)),
                       np.exp(-1j*2*np.pi*freq_x*-(dx)))
    inv_shifter = np.outer(np.exp(-1j*2*np.pi*freq_y*(dy)),
                           np.exp(-1j*2*np.pi*freq_x*(dx)))
    result_fft = X_fft*shifter

    # symmeterize
    result_fft = result_fft.real

    # Shift back
    result_fft = result_fft*inv_shifter

    # Transform to real space
    result = np.fft.fftshift(np.fft.ifftn(result_fft))
    # Return the unpadded transform
    result = np.real(result[corner[0]:corner[0]+shape[0], corner[1]:corner[1]+shape[1]])
    result[zeroMask] = 0
    assert result.shape == shape
    return result


def prox_uncentered_symmetry(X, step, center=None, algorithm="kspace", fill=None, shift=None, strength=.5):
    """Symmetry with off-center peak

    Symmetrize X for all pixels with a symmetric partner.

    Parameters
    ----------
    X: array
        The parameter to update.
    step: `int`
        Step size of the gradient step.
    center: tuple of `int`
        The center pixel coordinates to apply the symmetry operator.
    algorithm: `string`
        The algorithm to use for symmetry.
        * If `algorithm = "kspace" then `X` is shifted by `shift` and
          symmetry is performed in kspace. This is the only symmetry algorithm
          in scarlet that works for fractional pixel shifts.
        * If `algorithm = "sdss" then the SDSS symmetry is used,
          namely the source is made symmetric around the `center` pixel
          by taking the minimum of each pixel and its symmetric partner.
          This is the algorithm used when initializing an `ExtendedSource`
          because it keeps the morphologies small, but during optimization
          the penalty is much stronger than the gradient
          and often leads to vanishing sources.
        * If `algorithm = "soft" then soft symmetry is used,
          meaning `X` will be allowed to differ from symmetry by the fraction
          `strength` from a perfectly symmetric solution. It is advised against
          using this algorithm because it does not work in general for sources
          shifted by a fractional amount, however it is used internally if
          a source is centered perfectly on a pixel.
    fill: `float`
        The value to fill the region that cannot be made symmetric.
        When `fill` is `None` then the region of `X` that is not symmetric
        is not constrained.
    strength: `float`
        The amount that symmetry is enforced. If `strength=0` then no
        symmetry is enforced, while `strength=1` enforces strict symmetry
        (ie. the mean of the two symmetric pixels is used for both of them).
        This parameter is only used when `algorithm = "soft"`.

    Returns
    -------
    result: `function`
        The update function based on the specified parameters.
    """
    if algorithm == "kspace" and (shift is None or np.all(shift == 0)):
        algorithm = "soft"
        strength = 1
    if algorithm == "kspace":
        return uncentered_operator(X, prox_kspace_symmetry, center, shift=shift, step=step, fill=fill)
    if algorithm == "sdss":
        return uncentered_operator(X, prox_sdss_symmetry, center, step=step, fill=fill)
    if algorithm == "soft" or algorithm == "kspace" and shift is None:
        # If there is no shift then the symmetry is exact and we can just use
        # the soft symmetry algorithm
        return uncentered_operator(X, prox_soft_symmetry, center, step=step, strength=strength, fill=fill)

    msg = "algorithm must be one of 'soft', 'sdss', 'kspace', recieved '{0}''"
    raise ValueError(msg.format(algorithm))


def proj(A, B):
    """Returns the projection of A onto the hyper-plane defined by B"""
    return A - (A*B).sum()*B/(B**2).sum()


def proj_dist(A, B):
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
        if diff[s] < diff[s-1]:
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
        if diff[s] < diff[s-1]:
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
            X[:, disk_k] = algorithm(X[:, bulge_k], X[:, disk_k])
    X = prox_unity_plus(X, step, axis=0)
    return X


def getOffsets(width, coords=None):
    """Get the offset and slices for a sparse band diagonal array
    For an operator that interacts with its neighbors we want a band diagonal matrix,
    where each row describes the 8 pixels that are neighbors for the reference pixel
    (the diagonal). Regardless of the operator, these 8 bands are always the same,
    so we make a utility function that returns the offsets (passed to scipy.sparse.diags).
    See `diagonalizeArray` for more on the slices and format of the array used to create
    NxN operators that act on a data vector.
    """
    # Use the neighboring pixels by default
    if coords is None:
        coords = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    offsets = [width*y+x for y, x in coords]
    slices = [slice(None, s) if s < 0 else slice(s, None) for s in offsets]
    slicesInv = [slice(-s, None) if s < 0 else slice(None, -s) for s in offsets]
    return offsets, slices, slicesInv


def diagonalizeArray(arr, shape=None, dtype=np.float64):
    """Convert an array to a matrix that compares each pixel to its neighbors
    Given an array with length N, create an 8xN array, where each row will be a
    diagonal in a diagonalized array. Each column in this matrix is a row in the larger
    NxN matrix used for an operator, except that this 2D array only contains the values
    used to create the bands in the band diagonal matrix.
    Because the off-diagonal bands have less than N elements, ``getOffsets`` is used to
    create a mask that will set the elements of the array that are outside of the matrix to zero.
    ``arr`` is the vector to diagonalize, for example the distance from each pixel to the peak,
    or the angle of the vector to the peak.
    ``shape`` is the shape of the original image.
    """
    if shape is None:
        height, width = arr.shape
        data = arr.flatten()
    elif len(arr.shape) == 1:
        height, width = shape
        data = np.copy(arr)
    else:
        raise ValueError("Expected either a 2D array or a 1D array and a shape")
    size = width * height

    # We hard code 8 rows, since each row corresponds to a neighbor
    # of each pixel.
    diagonals = np.zeros((8, size), dtype=dtype)
    mask = np.ones((8, size), dtype=bool)
    offsets, slices, slicesInv = getOffsets(width)
    for n, s in enumerate(slices):
        diagonals[n][slicesInv[n]] = data[s]
        mask[n][slicesInv[n]] = 0

    # Create a mask to hide false neighbors for pixels on the edge
    # (for example, a pixel on the left edge should not be connected to the
    # pixel to its immediate left in the flattened vector, since that pixel
    # is actual the far right pixel on the row above it).
    mask[0][np.arange(1, height)*width] = 1
    mask[2][np.arange(height)*width-1] = 1
    mask[3][np.arange(1, height)*width] = 1
    mask[4][np.arange(1, height)*width-1] = 1
    mask[5][np.arange(height)*width] = 1
    mask[7][np.arange(1, height-1)*width-1] = 1

    return diagonals, mask


def diagonalsToSparse(diagonals, shape, dtype=np.float64):
    """Convert a diagonalized array into a sparse diagonal matrix
    ``diagonalizeArray`` creates an 8xN array representing the bands that describe the
    interactions of a pixel with its neighbors. This function takes that 8xN array and converts
    it into a sparse diagonal matrix.
    See `diagonalizeArray` for the details of the 8xN array.
    """
    import scipy.sparse
    height, width = shape
    offsets, slices, slicesInv = getOffsets(width)
    diags = [diag[slicesInv[n]] for n, diag in enumerate(diagonals)]
    diagonalArr = scipy.sparse.diags(diags, offsets, dtype=dtype)
    return diagonalArr


def getRadialMonotonicWeights(shape, useNearest=True, minGradient=1, center=None):
    """Create the weights used for the Radial Monotonicity Operator
    This version of the radial monotonicity operator selects all of the pixels closer to the peak
    for each pixel and weights their flux based on their alignment with a vector from the pixel
    to the peak. In order to quickly create this using sparse matrices, its construction is a bit opaque.
    """
    if center is None:
        center = ((shape[0]-1) // 2, (shape[1]-1) // 2)
    name = "RadialMonotonicWeights"


    key = tuple(shape) + tuple(center) + (useNearest, minGradient)
    try:

        cosNorm = Cache.check(name, key)
    except KeyError:

        # Center on the center pixel
        py, px = int(center[0]), int(center[1])
        # Calculate the distance between each pixel and the peak
        x = np.arange(shape[1])
        y = np.arange(shape[0])
        X, Y = np.meshgrid(x, y)
        X = X - px
        Y = Y - py
        distance = np.sqrt(X**2+Y**2)

        # Find each pixels neighbors further from the peak and mark them as invalid
        # (to be removed later)
        distArr, mask = diagonalizeArray(distance, dtype=np.float64)
        relativeDist = (distance.flatten()[:, None]-distArr.T).T
        invalidPix = relativeDist <= 0

        # Calculate the angle between each pixel and the x axis, relative to the peak position
        # (also avoid dividing by zero and set the tan(infinity) pixel values to pi/2 manually)
        inf = X == 0
        tX = X.copy()
        tX[inf] = 1
        angles = np.arctan2(-Y, -tX)
        angles[inf & (Y != 0)] = 0.5*np.pi*np.sign(angles[inf & (Y != 0)])

        # Calcualte the angle between each pixel and it's neighbors
        xArr, m = diagonalizeArray(X)
        yArr, m = diagonalizeArray(Y)
        dx = (xArr.T-X.flatten()[:, None]).T
        dy = (yArr.T-Y.flatten()[:, None]).T
        # Avoid dividing by zero and set the tan(infinity) pixel values to pi/2 manually
        inf = dx == 0
        dx[inf] = 1
        relativeAngles = np.arctan2(dy, dx)
        relativeAngles[inf & (dy != 0)] = 0.5*np.pi*np.sign(relativeAngles[inf & (dy != 0)])

        # Find the difference between each pixels angle with the peak
        # and the relative angles to its neighbors, and take the
        # cos to find its neighbors weight
        dAngles = (angles.flatten()[:, None]-relativeAngles.T).T
        cosWeight = np.cos(dAngles)
        # Mask edge pixels, array elements outside the operator (for offdiagonal bands with < N elements),
        # and neighbors further from the peak than the reference pixel
        cosWeight[invalidPix] = 0
        cosWeight[mask] = 0

        if useNearest:
            # Only use a single pixel most in line with peak
            cosNorm = np.zeros_like(cosWeight)
            columnIndices = np.arange(cosWeight.shape[1])
            maxIndices = np.argmax(cosWeight, axis=0)
            indices = maxIndices*cosNorm.shape[1]+columnIndices
            indices = np.unravel_index(indices, cosNorm.shape)
            cosNorm[indices] = minGradient
            # Remove the reference for the peak pixel
            cosNorm[:, px+py*shape[1]] = 0
        else:
            # Normalize the cos weights for each pixel
            normalize = np.sum(cosWeight, axis=0)
            normalize[normalize == 0] = 1
            cosNorm = (cosWeight.T/normalize[:, None]).T
            cosNorm[mask] = 0

        Cache.set(name, key, cosNorm)

    return cosNorm


def getRadialMonotonicOp(shape, useNearest=True, minGradient=1, subtract=True):
    """Create an operator to constrain radial monotonicity
    This version of the radial monotonicity operator selects all of the pixels closer to the peak
    for each pixel and weights their flux based on their alignment with a vector from the pixel
    to the peak. In order to quickly create this using sparse matrices, its construction is a bit opaque.
    """
    import scipy.sparse

    name = "RadialMonotonic"
    key = tuple(shape) + (useNearest, minGradient, subtract)
    try:
        monotonic = Cache.check(name, key)
    except KeyError:

        cosNorm = getRadialMonotonicWeights(shape, useNearest=useNearest, minGradient=1)
        cosArr = diagonalsToSparse(cosNorm, shape)

        # The identity with the peak pixel removed represents the reference pixels
        # Center on the center pixel
        px = int(shape[1]/2)
        py = int(shape[0]/2)
        # Calculate the distance between each pixel and the peak
        size = shape[0]*shape[1]
        diagonal = np.ones(size)
        diagonal[px+py*shape[1]] = -1
        if subtract:
            monotonic = cosArr-scipy.sparse.diags(diagonal, offsets=0)
        else:
            monotonic = cosArr
        monotonic = MatrixAdapter(monotonic.tocoo(), axis=1)
        monotonic.spectral_norm
        Cache.set(name, key, monotonic)

    return monotonic

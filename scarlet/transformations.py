from __future__ import print_function, division

import numpy as np
import scipy.sparse

class GammaOp():
    """Combination of Linear (x,y) Transformation and PSF Convolution

    Since the translation operators and PSF convolution operators both act
    on the deconvolved, centered S matrix, we can instead think of the translation
    operators translating the PSF convolution kernel, making a single transformation
    Gamma = Ty.P.Tx, where Tx,Ty are the translation operators and P is the PSF
    convolution operator.
    """
    def __init__(self, shape, psf=None, offset_int=None):
        """Constructor

        Parameters
        ----------
        shape: tuple
            Shape of the Source frame
        psf: array-like, default=`None`
            PSF image in either a single band (used for all images)
            or an array/list of images with a PSF image for each band.
        offset_int: int, default=`None`
            The translation matrices only care about fractional offsets.
            Integer offsets are handled in a different way, but are no longer necessary
            because we are now creating frames around sources.
            TODO: remove this option
        """
        if offset_int is None:
            offset_int = (0,0)
        self.psf = psf
        self._cache = {}

    def _make_matrices(self, shape, offset_int):
        """Build Tx, Ty, P, Gamma

        To save processing time we separate the diagonal component
        (which is just an identity matrix) and the off-diagonal
        component (which is just a band diagonal matrix that is all zeros).

        Parameters
        ----------
        shape: tuple
            Shape of the `~scarlet.Source` frame
        offset_int: int, default=`None`
            Integer offset of the translation. See `__init__`.

        Returns
        -------
        result: tuple
            tx, tx_plus, tx_minus, ty, ty_plus, ty_minus, P
        tx, ty: `~scipy.sparse` array
            Sparse arrays that contain the diagonal components of the Tx, Ty matrices
        tx_plus, ty_plus: `~scipy.sparse` array
            Sparse arrays that contain the off diagonal components of the Tx, Ty matrices.

        """
        self.B, height, width = shape
        tx = scipy.sparse.diags([1.], offsets=[offset_int[1]], shape=(width, width))
        tx_minus = scipy.sparse.diags([-1.,1.], offsets=[offset_int[1],offset_int[1]+1], shape=(width, width))
        tx_plus = scipy.sparse.diags([1.,-1.],offsets=[offset_int[1],offset_int[1]-1], shape=(width, width))
        tx = scipy.sparse.block_diag([tx]*height)
        tx_plus = scipy.sparse.block_diag([tx_plus]*height)
        tx_minus = scipy.sparse.block_diag([tx_minus]*height)

        size = height*width
        ty = scipy.sparse.diags([1], offsets=[offset_int[0]*width], shape=(size, size), dtype=np.float64)
        ty_minus = scipy.sparse.diags([-1., 1.], offsets=[offset_int[0]*width, (offset_int[0]+1)*width],
                                      shape=(size, size))
        ty_plus = scipy.sparse.diags([1., -1.], offsets=[offset_int[0]*width, (offset_int[0]-1)*width],
                                     shape=(size, size))

        P = self._adapt_PSF(shape[1:])
        return tx,tx_plus,tx_minus,ty,ty_plus,ty_minus,P

    def __call__(self, pos, shape, offset_int=None):
        """Get the operators to translate source

        Parameters
        ----------
        pos: array-like
            (dy,dx) Fractional position in the x and y directions to shift the source.
        shape: tuple
            Shape of the `~scarlet.Source` frame
        offset_int: int, default=`None`
            Integer offset of the translation. See `__init__`.

        Returns
        -------
        Gamma: list of `~scipy.sparse` arrays
            Sparse Gamma array for each band, where Gamma=Ty.P.Tx.
        """
        dy, dx = pos
        if offset_int is None:
            offset_int = (0,0)
        key = tuple(shape[1:]) + tuple(offset_int)

        try:
            tx,tx_plus,tx_minus,ty,ty_plus,ty_minus,P = self._cache[key]
        except KeyError:
            self._cache[key] = self._make_matrices(shape, offset_int)
            tx, tx_plus, tx_minus, ty, ty_plus, ty_minus, P = self._cache[key]

        # Create Tx
        if dx<0:
            dtx = tx_minus
        else:
            dtx = tx_plus
        # linear interpolation between centers and offset by one pixel
        Tx = tx - dx*dtx
        # Create Ty
        if dy<0:
            dty = ty_minus
        else:
            dty = ty_plus
        Ty = ty - dy*dty
        # return Tx, Ty

        if P is None:
            return Ty.dot(Tx)
        if hasattr(P, 'shape'):
            _gamma = P.dot(Ty.dot(Tx))
            # simplifies things later on: PSF always comes with B Gamma operators
            return [_gamma] * self.B
        return [Pb.dot(Ty.dot(Tx)) for Pb in P]

    def _adapt_PSF(self, shape):
        """Create multiband PSF operator (if necessary)

        `~scipy.sparse` only works for 2D matrices. Since we have a
        3rd dimension, the band, we need an adapter to keep track of the
        PSF operator in different bands.

        Parameters
        ----------
        shape: tuple
            Shape of the `~scarlet.Source` frame.

        Returns
        -------
        P: `~scipy.sparse` array or list of `~scipy.sparse` arrays
            PSF convolution operator.
        """
        if self.psf is None:
            return None

        if hasattr(self.psf, 'shape'): # single matrix
            return getPSFOp(self.psf, shape)

        P = []
        for b in range(len(self.psf)):
            P.append(getPSFOp(self.psf[b], shape))
        return P


def getZeroOp(shape):
    size = shape[0]*shape[1]
    # matrix with ones on diagonal shifted by k, here out of matrix: all zeros
    return scipy.sparse.eye(size,k=size)

def getIdentityOp(shape):
    size = shape[0]*shape[1]
    return scipy.sparse.identity(size)

def getSymmetryOp(shape):
    """Create a linear operator to symmetrize an image

    Given the ``shape`` of an image, create a linear operator that
    acts on the flattened image to return its symmetric version.
    """
    size = shape[0]*shape[1]
    idx = np.arange(shape[0]*shape[1])
    sidx = idx[::-1]
    symmetryOp = scipy.sparse.identity(size)
    symmetryOp -= scipy.sparse.coo_matrix((np.ones(size),(idx, sidx)), shape=(size,size))
    return symmetryOp

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
        coords = [(-1,-1), (-1,0), (-1, 1), (0,-1), (0,1), (1, -1), (1,0), (1,1)]
    offsets = [width*y+x for y,x in coords]
    slices = [slice(None, s) if s<0 else slice(s, None) for s in offsets]
    slicesInv = [slice(-s, None) if s<0 else slice(None, -s) for s in offsets]
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
    elif len(arr.shape)==1:
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
    mask[0][np.arange(1,height)*width] = 1
    mask[2][np.arange(height)*width-1] = 1
    mask[3][np.arange(1,height)*width] = 1
    mask[4][np.arange(1,height)*width-1] = 1
    mask[5][np.arange(height)*width] = 1
    mask[7][np.arange(1,height-1)*width-1] = 1

    return diagonals, mask

def diagonalsToSparse(diagonals, shape, dtype=np.float64):
    """Convert a diagonalized array into a sparse diagonal matrix

    ``diagonalizeArray`` creates an 8xN array representing the bands that describe the
    interactions of a pixel with its neighbors. This function takes that 8xN array and converts
    it into a sparse diagonal matrix.

    See `diagonalizeArray` for the details of the 8xN array.
    """
    height, width = shape
    offsets, slices, slicesInv = getOffsets(width)
    diags = [diag[slicesInv[n]] for n, diag in enumerate(diagonals)]
    diagonalArr = scipy.sparse.diags(diags, offsets, dtype=dtype)
    return diagonalArr

def getRadialMonotonicWeights(shape, useNearest=True, minGradient=1):
    """Create the weights used for the Radial Monotonicity Operator

    This version of the radial monotonicity operator selects all of the pixels closer to the peak
    for each pixel and weights their flux based on their alignment with a vector from the pixel
    to the peak. In order to quickly create this using sparse matrices, its construction is a bit opaque.
    """
    # Center on the center pixel
    px = int(shape[1]/2)
    py = int(shape[0]/2)
    # Calculate the distance between each pixel and the peak
    size = shape[0]*shape[1]
    x = np.arange(shape[1])
    y = np.arange(shape[0])
    X,Y = np.meshgrid(x,y)
    X = X - px
    Y = Y - py
    distance = np.sqrt(X**2+Y**2)

    # Find each pixels neighbors further from the peak and mark them as invalid
    # (to be removed later)
    distArr, mask = diagonalizeArray(distance, dtype=np.float64)
    relativeDist = (distance.flatten()[:,None]-distArr.T).T
    invalidPix = relativeDist<=0

    # Calculate the angle between each pixel and the x axis, relative to the peak position
    # (also avoid dividing by zero and set the tan(infinity) pixel values to pi/2 manually)
    inf = X==0
    tX = X.copy()
    tX[inf] = 1
    angles = np.arctan2(-Y,-tX)
    angles[inf&(Y!=0)] = 0.5*np.pi*np.sign(angles[inf&(Y!=0)])

    # Calcualte the angle between each pixel and it's neighbors
    xArr, m = diagonalizeArray(X)
    yArr, m = diagonalizeArray(Y)
    dx = (xArr.T-X.flatten()[:, None]).T
    dy = (yArr.T-Y.flatten()[:, None]).T
    # Avoid dividing by zero and set the tan(infinity) pixel values to pi/2 manually
    inf = dx==0
    dx[inf] = 1
    relativeAngles = np.arctan2(dy,dx)
    relativeAngles[inf&(dy!=0)] = 0.5*np.pi*np.sign(relativeAngles[inf&(dy!=0)])

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
        columnIndices =  np.arange(cosWeight.shape[1])
        maxIndices = np.argmax(cosWeight, axis=0)
        indices = maxIndices*cosNorm.shape[1]+columnIndices
        indices = np.unravel_index(indices, cosNorm.shape)
        cosNorm[indices] = minGradient
        # Remove the reference for the peak pixel
        cosNorm[:,px+py*shape[1]] = 0
    else:
        # Normalize the cos weights for each pixel
        normalize = np.sum(cosWeight, axis=0)
        normalize[normalize==0] = 1
        cosNorm = (cosWeight.T/normalize[:,None]).T
        cosNorm[mask] = 0
    return cosNorm

def getRadialMonotonicOp(shape, useNearest=True, minGradient=1, subtract=True):
    """Create an operator to constrain radial monotonicity

    This version of the radial monotonicity operator selects all of the pixels closer to the peak
    for each pixel and weights their flux based on their alignment with a vector from the pixel
    to the peak. In order to quickly create this using sparse matrices, its construction is a bit opaque.
    """
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

    return monotonic.tocoo()

def getPSFOp(psf, imgShape):
    """Create an operator to convolve intensities with the PSF

    Given a psf image ``psf`` and the shape of the blended image ``imgShape``,
    make a banded matrix out of all non-zero pixels in ``psfImg`` that acts as
    the PSF operator.
    """
    height, width = imgShape
    size = width * height

    # Calculate the coordinates of the pixels in the psf image above the threshold
    indices = np.where(psf != 0)
    indices = np.dstack(indices)[0]
    # assume all PSF images have odd dimensions and are centered!
    cy, cx = psf.shape[0]//2, psf.shape[1]//2
    coords = indices-np.array([cy,cx])

    # Create the PSF Operator
    offsets, slices, slicesInv = getOffsets(width, coords)
    psfDiags = [psf[y,x] for y,x in indices]
    psfOp = scipy.sparse.diags(psfDiags, offsets, shape=(size, size), dtype=np.float64)
    psfOp = psfOp.tolil()

    # Remove entries for pixels on the left or right edges
    cxRange = np.unique([cx for cy,cx in coords])
    for h in range(height):
        for y,x in coords:
            # Left edge
            if x<0 and width*(h+y)+x>=0 and h+y<=height:
                psfOp[width*h, width*(h+y)+x] = 0

                # Pixels closer to the left edge
                # than the radius of the psf
                for x_ in cxRange[cxRange<0]:
                    if (x<x_ and
                        width*h-x_>=0 and
                        width*(h+y)+x-x_>=0 and
                        h+y<=height
                    ):
                        psfOp[width*h-x_, width*(h+y)+x-x_] = 0

            # Right edge
            if x>0 and width*(h+1)-1>=0 and width*(h+y+1)+x-1>=0 and h+y<=height and width*(h+1+y)+x-1<size:
                psfOp[width*(h+1)-1, width*(h+y+1)+x-1] = 0

                for x_ in cxRange[cxRange>0]:
                    # Near right edge
                    if (x>x_ and
                        width*(h+1)-x_-1>=0 and
                        width*(h+y+1)+x-x_-1>=0 and
                        h+y<=height and
                        width*(h+1+y)+x-x_-1<size
                    ):
                        psfOp[width*(h+1)-x_-1, width*(h+y+1)+x-x_-1] = 0

    # Return the transpose, which correctly convolves the data with the PSF
    return psfOp.T.tocoo()

# ring-shaped masks around the peak
def getRingMask(im_shape, peak, outer, inner=0, flatten=False):
    height,width = im_shape
    x,y = np.meshgrid(np.arange(width), np.arange(height))
    r = np.sqrt((x-peak[1])**2 + (y-peak[0])**2)
    mask = (r < inner) | (r >= outer)
    if flatten:
        return mask.flatten()
    return mask

# odd-integer downsampling
def downsample(S, oversampling, mask=None):
    assert isinstance(oversampling, (int, long))
    if oversampling <= 1:
        return S
    else:
        height,width = S.shape
        height /= oversampling
        width /= oversampling
        Sd = np.zeros((height, width), dtype=S.dtype)
        if mask is None:
            S_ = S
        else:
            S_ = S*(~mask)
        # TODO: can we avoid the double loop?
        for h in range(height):
            for w in range(width):
                Sd[h,w] = S_[h*oversampling:(h+1)*oversampling,
                             w*oversampling:(w+1)*oversampling].sum() / oversampling**2
        return Sd

# construct spin-wave decomposition operator for given list of spin numbers m
# radial behavior can be specified as appropriate
def getSpinOp(ms, shape, thickness=4, peak=None, oversampling=21, radial_fct=lambda r:1./np.maximum(1,r)):
    """ Spin decomposition operator.

    The operator maps onto a basis function of R(r) exp(i m phi), where phi is
    the polar angle wrt to the peak (or, if None, the center of the image).

    The decomposition is performed in a set of concentric rings of fixed thickness.

    ms is a list of integers that indicate the requested spin numbers.
    thickness is the radial separation between the inner and outer ring radius.
    peak is (an optional) offset of the object from the image center.
    oversampling determine the higher-resolution grid for the in-pixel
    integration; it must be odd.
    radial_fct is the radial part of the spin basis function.

    """
    assert oversampling % 2 == 1
    assert hasattr(ms, '__iter__')

    height,width = shape
    assert height % 2 == 0 and width % 2 == 0

    if peak is None:
        peak = [height/2, width/2]
    x,y = np.meshgrid(np.arange(width*oversampling), np.arange(height*oversampling))
    x = x * 1./oversampling - peak[1]
    y = y * 1./oversampling - peak[0]
    # proper treatment of over & downsampling: center pixel location
    if oversampling > 1:
        x -= 0.5 - 0.5/oversampling
        y -= 0.5 - 0.5/oversampling
    # convert to polar
    r = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y,x)

    # define series of radial ring-shaped masks (oversampled as well)
    r_limit = (np.min([peak[0], height-peak[0], peak[1], width-peak[1]]) - 1)*oversampling
    base = thickness*oversampling
    lims = [(base*(i+1), base*i) for i in range(r_limit/base)]
    mask_peak = ((peak[0]+0.5)*oversampling - 0.5, (peak[1]+0.5)*oversampling - 0.5)
    masks = [getRingMask(r.shape, mask_peak, outer, inner) for outer, inner in lims]

    Ss = []
    for i in range(len(ms)):
        m = ms[i]
        spin = radial_fct(r) * np.exp(1j*m*phi)
        for j in range(len(masks)):
            mask = masks[j]
            S = downsample(spin, oversampling, mask=mask).flatten()
            Ss.append(S)

    # TODO: make Ss sparse and split real and imaginary part
    return np.array(Ss)

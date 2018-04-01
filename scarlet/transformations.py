from __future__ import print_function, division
import warnings

import numpy as np
import scipy.sparse

# global cache to hold all transformation matrices, except for GammaOp
cache = {}

def check_cache(name, key):
    global cache
    try:
        cache[name]
    except KeyError:
        cache[name] = {}

    return cache[name][key]

def get_filter_slices(coords):
    """Convert a list of relative coordinates to slices

    A `LinearFilter` is defined by an image of weights
    and a list of relative coordinates from the current pixels
    for each weight.
    This method converts those coordinates into slices that are
    used to apply the filter.
    """
    slices = []
    inv_slices = []
    for cy, cx in coords:
        _slice = [slice(None),slice(None)]
        _inv_slice = [slice(None),slice(None)]
        if cy>0:
            _slice[0] = slice(cy,None)
            _inv_slice[0] = slice(None,-cy)
        elif cy<0:
            _slice[0] = slice(None,cy)
            _inv_slice[0] = slice(-cy, None)
        if cx>0:
            _slice[1] = slice(cx,None)
            _inv_slice[1] = slice(None,-cx)
        elif cx<0:
            _slice[1] = slice(None,cx)
            _inv_slice[1] = slice(-cx, None)
        slices.append(_slice)
        inv_slices.append(_inv_slice)
    return slices, inv_slices

def apply_filter(X, weights, slices, inv_slices):
    """Apply a filter to a 2D image X

    Parameters
    ----------
    X: 2D numpy array
        The image to apply the filter to
    weights: 1D array
        Weights corresponding to each slice in `slices`
    slices: list of `slice` objects
        Slices in the new `X` to store the filtered X
    inv_slices: list of `slice` objects
        Slices of `X` to apply each weight
    
    Returns
    -------
    new_X: 2D numpy array
        The result of applying the filter to `X`
    """
    assert len(slices) == len(inv_slices) == len(weights)
    result = np.zeros(X.shape, dtype=X.dtype)
    for n, weight in enumerate(weights):
        result[slices[n]] += weight * X[inv_slices[n]]
    return result

class LinearFilter:
    """A filter that can be applied to an image

    This acts like a sparse diagonal matrix that applies an
    image of weights to a 2D matrix.
    """
    def __init__(self, img, coords):
        """Initialize the Filter

        Parameters
        ----------
        img: 2D or 1D array-like
            Weights to apply to the filter
        coords: 2D or 1D array-like
            Relative coordinates from the current pixel
            for each weight in `img`
            (so [0,0] is the current pixel).
        """
        self.img = np.array(img)
        self._coords = np.array(coords)
        self.coords = self._coords
        self.slices, self.inv_slices = get_filter_slices(self.coords.reshape(-1, 2))
    
    @property
    def T(self):
        """Transpose the filter
        """
        return LinearFilter(self.img, -self._coords)

    def dot(self, X):
        """Apply the filter to an image

        Parameters
        ----------
        X: 2D numpy array or `LinearFilter` or `LinearFilterChain`
            Array to apply the filter to, or chain of filters to
            prepend this filter to.
        
        Returns
        -------
        result: 2D numpy array or `LinearFilterChain`
            If `X` is an array, this is the result of the
            filter applied to `X`.
            If `X` is not an image but is another filter
            (or chain of filters) then a new `LinearFilterChain` is
            returned with this one prepended.
        """
        if isinstance(X, LinearFilter):
            return LinearFilterChain([self, X])
        elif isinstance(X, LinearFilterChain):
            X.filters.insert(0,self)
            return X
        else:
            return apply_filter(X, self.img.reshape(-1), self.slices, self.inv_slices)

class LinearFilterChain:
    """Chain of `LinearFilter` objects

    Because `LinearFilter` objects are not really arrays,
    this class keeps track of the order of a series of filters.
    """
    def __init__(self, filters):
        """Initialize the class

        Parameters
        ----------
        filters: list
            List of `LinearFilter` objects, in the order from
            left to right. So the last element in `filters` is
            applied to an image first.
        """
        self.filters = filters
    
    @property
    def T(self):
        """Transpose the list of filters

        Reverse the order and transpose each `LinearFilter` in
        `self.filters`.
        """
        return LinearFilterChain([f.T for f in self.filters[::-1]])

    def dot(self, X):
        """Apply the filters

        Apply the filters in reverse order,
        starting with the last element, to the
        image `X`.

        Parameters
        ----------
        X: 2D numpy array
            Image to apply the filters to.
        
        Returns
        -------
        result: 2D numpy array or `LinearFilterChain`
            If `X` is an array, this will be the result of
            applying all of the filters in `X`.
            Otherwise this returns a new `LinearFilterChain`
            that appends `X`.
        """
        if isinstance(X, LinearFilter):
            self.filters.append(X)
        elif isinstance(X, LinearFilterChain):
            for f in X.filters:
                self.filters.append(f)
        else:
            _filters = self.filters[::-1]
            result = X
            for f in _filters:
                result = f.dot(result)
            return result
        return self

class LinearTranslation(LinearFilter):
    """Linear translation in x and y
    """
    def __init__(self, dy=0, dx=0):
        """Initialize the filter

        Parameters
        ----------
        dy: float
            Fractional amount (from 0 to 1) to
            shift the image in the y-direction
        dx: float
            Fractional amount (from 0 to 1) to
            shift the image in the x-direction
        """
        self.set_transform(dy, dx)

    def set_transform(self, dy=0, dx=0):
        """Create the image and coords for the transform

        Parameters
        ----------
        dy: float
            Fractional amount (from 0 to 1) to
            shift the image in the y-direction
        dx: float
            Fractional amount (from 0 to 1) to
            shift the image in the x-direction
        """
        sign_x = np.sign(dx)
        sign_y = np.sign(dy)
        dx = np.abs(dx)
        dy = np.abs(dy)
        ddx = 1-dx
        ddy = 1-dy
        img = np.array([ddx*ddy, ddy*dx, ddx*dy, dx*dy])
        coords = np.array([[0,0], [0,sign_x], [sign_y,0], [sign_y,sign_x]], dtype=int)
        super(LinearTranslation,self).__init__(img, coords)

class Gamma:
    """Combination of Linear (x,y) Transformation and PSF Convolution

    Since the translation operators and PSF convolution operators both act
    on the de-convolved, centered S matrix, we can instead think of the translation
    operators translating the PSF convolution kernel, making a single transformation
    Gamma = Ty.P.Tx, where Tx,Ty are the translation operators and P is the PSF
    convolution operator.
    """
    def __init__(self, psfs=None, center=None, dy=0, dx=0):
        """Constructor

        Parameters
        ----------
        psfs: array-like, default=`None`
            PSF image in either a single band (used for all images)
            or an array/list of images with a PSF image for each band.
            If `psfs` is `None` then no PSF convolution is performed, but
            a number of bands `B` must be specified.
        center: integer array-like, default=`None`
            Center of the PSF. If `center` is `None` and a set of `psfs` is given,
            the central pixel of `psfs[0]` is used.
        dy: float
            Fractional shift in the y direction
        dx: float
            Fractional shift in the x direction
        """
        self.psfs = psfs

        # Create the PSF filter for each band
        if psfs is not None:
            self._update_psf(psfs, center)
            self.B = len(psfs)
        else:
            self.psfFilters = None
            self.B = None
        # Create the transformation matrices
        self._update_translation(dy, dx)
    
    def _update_psf(self, psfs, center=None):
        """Update the psf convolution filter
        """
        if center is None:
            center = [psfs[0].shape[0]//2, psfs[0].shape[1]//2]
        self.center = center
        self.psfFilters = []
        x = np.arange(psfs[0].shape[1])
        y = np.arange(psfs[0].shape[0])
        x,y = np.meshgrid(x,y)
        x -= center[1]
        y -= center[0]
        coords = np.dstack([y,x])
        for psf in psfs:
            self.psfFilters.append(LinearFilter(psf, coords))

    def _update_translation(self, dy=0, dx=0):
        """Update the translation filter
        """
        self.dx = dx
        self.dy = dy
        self.translation = LinearTranslation(dy, dx)

    def update(self, psfs=None, center=None, dx=None, dy=None):
        """Update the psf convolution filter and/or the translations

        See `self.__init__` for parameter descriptions
        """
        if psfs is not None:
            self._update_psf(psfs, center)
        if dx is not None or dy is not None:
            if dx is None:
                dx = self.dx
            if dy is None:
                dy = self.dy
            self._update_translation(dy, dx)

    def __call__(self, dyx=None):
        """Build a Gamma "Matrix"

        Combine the translation and PSF convolution into
        a single class that acts like a linear operator.

        Parameters
        ----------
        dyx: 2D array-like, default=`None`
            Fractional shift in position in `[dy,dx]`.
            If `dyx` is `None`, then the already built
            translation matrix is used.
        """
        if dyx is None or (dyx[0] == self.dy and dyx[1] == self.dx):
            translation = self.translation
        else:
            translation = LinearTranslation(*dyx)
        if self.psfFilters is None:
            gamma = translation
        else:
            gamma = []
            for b in range(self.B):
                gamma.append(LinearFilterChain([translation, self.psfFilters[b]]))
        return gamma

class LinearOperator:
    """Mock a linear operator

    Because scarlet uses 2D images for morphologies,
    the morphology they operate on must be flattened,
    so this class mocks a linear operator by applying it's
    functions to a flattened variable.
    """
    def __init__(self, L):
        """Initialize the class
        """
        while isinstance(L, LinearOperator):
            L = L.L
        self.L = L

    def dot(self, X):
        """Take the dot product with a 2D X
        """
        if isinstance(X, np.ndarray):
            return self.L.dot(X.reshape(-1)).reshape(X.shape)
        else:
            return LinearOperator(self.L.dot(X))

    @property
    def T(self):
        """Return a transposed version of the linear operator
        """
        return LinearOperator(self.L.T)

    def spectral_norm(self):
        """Spectral norm of the operator
        """
        from scipy.sparse import issparse
        LTL = self.L.T.dot(self.L)
        if issparse(self.L):
            if min(self.L.shape) <= 2:
                L2 = np.real(np.linalg.eigvals(LTL.toarray()).max())
            else:
                import scipy.sparse.linalg
                L2 = np.real(scipy.sparse.linalg.eigs(LTL, k=1, return_eigenvectors=False)[0])
        else:
            import IPython; IPython.embed()
            L2 = np.real(np.linalg.eigvals(LTL).max())
        return L2

    def __len__(self):
        return len(self.L)

    @property
    def shape(self):
        return self.L.shape

    @property
    def size(self):
        return self.L.size
    
    @property
    def ndim(self):
        print(self.L.ndim)
        return self.L.ndim

    def __sub__(self, op):
        return LinearOperator(self.L - op)

    def __rsub__(self, op):
        return LinearOperator(op - self.L)

    def __add__(self, op):
        return LinearOperator(self.L + op)

    def __radd__(self, op):
        return LinearOperator(op + self.L)

    def __mul__(self, op):
        return LinearOperator(self.L * op)

    def __rmul__(self, op):
        return LinearOperator(op * self.L)

    def __div__(self, op):
        return LinearOperator(self.L / op)

    def __rdiv__(self, op):
        return LinearOperator(self.L / op)
    
    def reshape(self, shape):
        return LinearOperator(self.L.reshape(shape))

    def __array_prepare__(self, *args):
        print("preparing")
        return self.L.__array_prepare__(*args)

    def __getattr__(self, attr):
        if attr not in self.__dict__.keys:
            return getattr(self.L, attr)

def getPSFOp(psf, imgShape):
    """Create an operator to convolve intensities with the PSF

    Given a psf image ``psf`` and the shape of the blended image ``imgShape``,
    make a banded matrix out of all non-zero pixels in ``psfImg`` that acts as
    the PSF operator.
    """

    warnings.warn("The 'psfOp' is deprecated, use 'LinearFilter' instead")
    name = "PSF"
    key = tuple(imgShape)
    try:
        psfOp = check_cache(name, key)

    except KeyError:
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
        psfOp = LinearOperator(psfOp.T.tocoo())

        global cache
        cache[name][key] = psfOp

    return psfOp

def getZeroOp(shape):
    size = shape[0]*shape[1]
    name = "Zero"
    key = tuple(shape)
    try:
        L = check_cache(name, key)
    except KeyError:
        # matrix with ones on diagonal shifted by k, here out of matrix: all zeros
        L = LinearOperator(scipy.sparse.eye(size, k=size))
        global cache
        cache[name][key] = L
    return L

def getIdentityOp(shape):
    size = shape[0]*shape[1]
    name = "Id"
    key = tuple(shape)
    try:
        L = check_cache(name, key)
    except KeyError:
        # matrix with ones on diagonal shifted by k, here out of matrix: all zeros
        L = LinearOperator(scipy.sparse.identity(size))
        global cache
        cache[name][key] = L
    return L

def getSymmetryOp(shape):
    """Create a linear operator to symmetrize an image

    Given the ``shape`` of an image, create a linear operator that
    acts on the flattened image to return its symmetric version.
    """
    size = shape[0]*shape[1]
    name = "Symm"
    key = tuple(shape)
    try:
        symmetryOp = check_cache(name, key)
    except KeyError:
        idx = np.arange(shape[0]*shape[1])
        sidx = idx[::-1]
        symmetryOp = getIdentityOp(shape)
        symmetryOp -= scipy.sparse.coo_matrix((np.ones(size),(idx, sidx)), shape=(size,size))
        symmetryOp = LinearOperator(symmetryOp)
        global cache
        cache[name][key] = symmetryOp
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

    name = "RadialMonotonicWeights"
    key = tuple(shape) + (useNearest, minGradient)
    try:
        cosNorm = check_cache(name, key)
    except KeyError:

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

        global cache
        cache[name][key] = cosNorm

    return cosNorm

def getRadialMonotonicOp(shape, useNearest=True, minGradient=1, subtract=True):
    """Create an operator to constrain radial monotonicity

    This version of the radial monotonicity operator selects all of the pixels closer to the peak
    for each pixel and weights their flux based on their alignment with a vector from the pixel
    to the peak. In order to quickly create this using sparse matrices, its construction is a bit opaque.
    """

    name = "RadialMonotonic"
    key = tuple(shape) + (useNearest, minGradient, subtract)
    try:
        monotonic = check_cache(name, key)
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
        monotonic = LinearOperator(monotonic.tocoo())

        global cache
        cache[name][key] = monotonic

    return monotonic


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

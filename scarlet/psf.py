import autograd.numpy as np
from functools import partial
from .interpolation import apply_2D_trapezoid_rule

def moffat(y, x, alpha=4.7, beta=1.5, shape=None):
    """Symmetric 2D Moffat function

    .. math::

        (1+\frac{(x-x0)^2+(y-y0)^2}{\alpha^2})^{-\beta}

    Parameters
    ----------
    y: float
        Vertical coordinate of the center
    x: float
        Horizontal coordinate of the center
    alpha: float
        Core width
    beta: float
        Power-law index
    shape: tuple
        Shape of the resulting array, typically `(C, Height, Width)`
        Note `C=None` is expected for model PSFs

    Returns
    -------
    result: array
        A 2D circular gaussian sampled at the coordinates `(y_i, x_j)`
        for all i and j in `shape`.
    """
    X = np.arange(shape[1])
    Y = np.arange(shape[2])
    X, Y = np.meshgrid(X, Y)
    return ((1+((X-x)**2+(Y-y)**2)/alpha**2)**-beta).reshape(shape)

def gaussian(y, x, sigma=1, shape=None):
    """Circular Gaussian Function

    Parameters
    ----------
    y: float
        Vertical coordinate of the center
    x: float
        Horizontal coordinate of the center
    sigma: float
        Standard deviation of the gaussian
    shape: tuple
        Shape of the resulting array, typically `(C, Height, Width)`
        Note `C=None` is expected for model PSFs

    Returns
    -------
    result: array
        A 2D circular gaussian sampled at the coordinates `(y_i, x_j)`
        for all i and j in `shape`.
    """
    Y = np.arange(shape[1])
    X = np.arange(shape[2])
    return (np.exp(-(y-Y)**2/(2*sigma**2))[:,None] * np.exp(-(x-X)**2/(2*sigma**2))).reshape(shape)


class PSF:
    def __init__(self, X):
        if hasattr(X, 'shape'):
            self._image = X.copy()
            self.normalize()
            self._func = None
            self.shape = X.shape


        elif hasattr(X, '__call__'):
            self._image = None
            self._func = X
            self.shape = None
        else:
            msg = "A PSF must be initialized with either an image or function"
            raise ValueError(msg)

    def __call__(self, y, x, shape=None):
        if shape is None:
            shape = self.shape
        if self._func is not None:
            # TODO: connect to oversampling
            return self._func(y, x, shape=shape)
        return None

    @property
    def image(self):
        if self._image is None:
            y, x = self.shape[1] // 2, self.shape[2] // 2
            self._image = self.__call__(y, x)
            self.normalize()
        return self._image

    def normalize(self):
        sums = self._image.sum(axis=(1, 2))
        self._image /= sums[:, None, None]
        return self

    def update_dtype(self, dtype):
        if self.image.dtype != dtype:
            self._image = self._image.astype(dtype)
        return self
        

def generate_psf_image(func, shape, subsamples=10, normalize=True, **kwargs):
    """Generate a PSF image based on a function and shape

    This function uses a subdivided pixel scale to sample `func` at
    higher frequencies and uses the trapezoid rule to estimate the integral
    in each subsampled region. The subsampled pixel are then summed to
    give the estimated integration values for each pixel at the scale requested
    by `shape`.

    Parameters
    ----------
    shape: tuple
        Expected shape of the output image.
    subsamples: int
        Number of pixels to subdivide each output pixel for
        a more accurate integration.
    normalize: bool
        Whether or not to normalize the PSF.
    kwargs: keyword arguments
        Keyword arguments to pass to `func`.

    Returns
    -------
    result: array
        Integrated image generated
    """
    ry, rx = np.array(shape) // 2

    y = np.linspace(-ry, ry, shape[0])
    x = np.linspace(-rx, rx, shape[1])
    f = partial(func, **kwargs)
    result = apply_2D_trapezoid_rule(y, x, f, subsamples)
    if normalize:
        result /= result.sum()
    return result

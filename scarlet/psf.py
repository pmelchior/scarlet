import autograd.numpy as np
import autograd.scipy as scipy

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
    # TODO: has no pixel-integration formula
    return ((1+((X-x)**2+(Y-y)**2)/alpha**2)**-beta)[None,:,:]

def gaussian(y, x, sigma=1, integrate=True, shape=None):
    """Circular Gaussian Function

    Parameters
    ----------
    y: float
        Vertical coordinate of the center
    x: float
        Horizontal coordinate of the center
    sigma: float
        Standard deviation of the gaussian
    integrate: bool
        Whether pixel integration is performed
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
    if not integrate:
        f = lambda X: np.exp(-X**2/(2*sigma**2))
    else:
        sqrt2 = np.sqrt(2)
        f = lambda x: np.sqrt(np.pi/2) * sigma * (scipy.special.erf((0.5 - x)/(sqrt2 * sigma)) + scipy.special.erf((2*x + 1)/(2*sqrt2*sigma)))

    return (f(Y-y)[:,None] * f(X-x)[None,:])[None,:,:]


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
        assert shape is not None, "Set PSF.shape first"

        if self._func is not None:
            return self._func(y, x, shape=shape)
        return None

    @property
    def image(self):
        if self._image is None:
            assert self.shape is not None, "Set PSF.shape first"
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

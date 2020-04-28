import autograd.numpy as np
import autograd.scipy as scipy
from .bbox import Box
from .component import Component
from .parameter import Parameter
from .fft import Fourier
from .observation import Observation


def moffat(y, x, alpha=4.7, beta=1.5, bbox=None):
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
    bbox: Box
        Bounding box over which to evaluate the function

    Returns
    -------
    result: array
        A 2D circular gaussian sampled at the coordinates `(y_i, x_j)`
        for all i and j in `shape`.
    """
    Y = np.arange(bbox.shape[1]) + bbox.origin[1]
    X = np.arange(bbox.shape[2]) + bbox.origin[2]
    X, Y = np.meshgrid(X, Y)
    # TODO: has no pixel-integration formula
    return ((1 + ((X - x) ** 2 + (Y - y) ** 2) / alpha ** 2) ** -beta)[None, :, :]


def gaussian(y, x, sigma=1, integrate=True, bbox=None):
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
    bbox: Box
        Bounding box over which to evaluate the function

    Returns
    -------
    result: array
        A 2D circular gaussian sampled at the coordinates `(y_i, x_j)`
        for all i and j in `shape`.
    """
    Y = np.arange(bbox.shape[1]) + bbox.origin[1]
    X = np.arange(bbox.shape[2]) + bbox.origin[2]

    def f(X):
        if not integrate:
            return np.exp(-(X ** 2) / (2 * sigma ** 2))
        else:
            sqrt2 = np.sqrt(2)
            return (
                np.sqrt(np.pi / 2)
                * sigma
                * (
                    scipy.special.erf((0.5 - X) / (sqrt2 * sigma))
                    + scipy.special.erf((2 * X + 1) / (2 * sqrt2 * sigma))
                )
            )

    return (f(Y - y)[:, None] * f(X - x)[None, :])[None, :, :]


class PSF:
    """Class to represent PSFs

    Parameters
    ----------
    X: array-like or method
        If `X` is an array, it represent an image of the PSF in every band.
        If `X` is a callable method, it describes a function that can generate a PSF when given coordinates (y, x).
    shape: tuple
        Shape of the 2D image to generate an PSF image for. Only used if `X` is a method.
    """

    def __init__(self, X, shape=None):
        if hasattr(X, "shape"):
            self._image = X.copy()
            self.normalize()
            self._func = None
            self.shape = X.shape
        elif hasattr(X, "__call__"):
            assert shape is not None, "Functional PSFs must set shape argument"
            self._image = None
            self._func = X
            self.shape = shape
        else:
            msg = "A PSF must be initialized with either an image or function"
            raise ValueError(msg)

    def __call__(self, y, x, bbox=None):
        """Generate analytic PSF image at coordinate (y,x)

        Parameters
        ----------
        y: float
            Vertical model frame coordinates for the center of PSF image
        x: float
            Horizontal model frame coordinates for the center of PSF image
        bbox: `~scarlet.Box`
            Bounding Box for the PSF image in model coordinates

        Returns
        -------
        image: array-like
            Centered image of the PSF at given coordinate.
        """
        if bbox is None:
            bbox = Box(self.shape)
        if self._func is not None:
            return self._func(y, x, bbox=bbox)
        return None

    @property
    def image(self):
        """Centered image of the PSF
        """
        if self._image is None:
            assert self.shape is not None, "Set PSF.shape first"
            y, x = self.shape[1] // 2, self.shape[2] // 2
            self._image = self.__call__(y, x)
            self.normalize()
        return self._image

    def normalize(self):
        """Normalize to PSF image in every band to unity
        """
        sums = self._image.sum(axis=(1, 2))
        self._image /= sums[:, None, None]
        return self

    def update_dtype(self, dtype):
        """Update data type of `image` to `dtype`
        """
        if self.image.dtype != dtype:
            self._image = self._image.astype(dtype)
        return self


class PSFDiffKernel(Component):
    """PSF Difference Kernel Source

    "Source" used to model the difference kernel to match one
    PSF to another. Typically this should be modeling the
    deconvolution kernel to convolve an image from an
    observed PSF into a model PSF, as the difference
    from model PSF to observed PSF is calculated much
    more quickly using a DFT in `~scarlet.Observation.match`.
    """
    def __init__(
        self,
        frame,
        initial,
        band,
        step=1e-2
    ):
        """Initialize the Model

        Parameters
        ----------
        frame: `~scarlet.Frame`
            The spectral and spatial characteristics of this component.
        initial: `numpy.array`
            Initial guess for the kernel.
        step: `double`
            Step size for the kernel parameter.
        """
        kernel = np.zeros(initial.shape, dtype=initial.dtype)
        kernel[band] = initial[band]
        kernel = Parameter(
            kernel,
            name="kernel",
            step=step,
        )
        super().__init__(frame, kernel)

    def get_model(self, *parameters):
        kernel = self.kernel
        if len(parameters) == 1:
            kernel = parameters[0]
        elif len(parameters) > 1:
            raise ValueError("PsfDiffKernel only takes a single parameter")
        return kernel

    @property
    def kernel(self):
        """Return the contents of the kernel parameter
        """
        return self._parameters[0]._data


class PsfObservation(Observation):
    def match(self, psfs):
        """Implement the observed PSFs as the difference kernel

        This is different than `~scarlet.Observation.match`,
        where the difference kernel that matches the model
        PSF to the observed PSF is calculated and stored.
        For a `PsfObservation` the input PSF, which
        should be the observed PSF (see below) is stored
        as the "difference kernel" while the actual
        deconvolution (difference) kernel is calculated
        by the model.

        Parameters
        ----------
        psfs: `~numpy.array`
            The input PSF that is being convolved.
            For deconvolution this is the observed PSF,
            since the observed PSF is convolved with the
            deconvolution kernel to match the model PSF.
            For matching a model PSF to a wider observed
            PSF use the `Observation` class,
            which calculates the difference kernel much
            more quickly using a DFT.

        Returns
        -------
        self: `~scarlet.PsfObservation`
            Return this object to allow for chaining.
        """
        self.bbox = Box(self.frame.shape)
        self.slices_for_images = self.bbox.slices_for(self.frame.shape)
        self.slices_for_model = self.bbox.slices_for(self.frame.shape)
        self._diff_kernels = Fourier(psfs)
        return self

    def get_loss(self, model):
        """get_loss

        We override `scarlet.Observation.get_loss` since the lognorm
        is the dominant term, which interferes with calculating
        relative error.
        """
        model_ = self.render(model)
        images_ = self.images[self.slices_for_images]
        weights_ = self.weights[self.slices_for_images]
        return np.sum(weights_ * (model_ - images_) ** 2) / 2


class DeconvolvedObservation(Observation):
    """Deconvolved image using a deconvolution kernel

    This is the same as `~scarlet.Observation` except that the matching
    algorithm uses a precalculated deconvolution kernel and the
    images are deconolved using that kernel. A separate class exists
    to make it less likely that the user executes this operation
    more than once, which will have adverse affects on the
    `images` property.
    """
    def match(self, model_frame, kernel):
        if hasattr(self, "matched") and self.matched:
            msg = ("Matching has already been executed. "
                   "Matching multiple times can have unexpected results.")
            raise RuntimeError(msg)
        super().match(model_frame, kernel)
        self.images = self.render(self.images)
        self.matched = True
        return self

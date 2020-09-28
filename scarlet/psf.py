import autograd.numpy as np
import autograd.scipy as scipy
from .bbox import Box
from .model import Model, abstractmethod
from .parameter import Parameter
from .fft import Fourier, shift


class PSF(Model):
    @abstractmethod
    def get_model(self, *parameter, offset=None):
        """Get the PSF realization

        Parameters
        ----------
        parameters: tuple of optimimzation parameters
        offset: 2D tuple or ``~scarlet.Parameter`
            Optional subpixel offset of the model, in units of frame pixels

        Returns
        -------
        result: array
            A centered PSF model defined by its parameters, shifted by `offset`
        """
        pass

    def prepare_param(self, X, name):
        if isinstance(X, Parameter):
            assert X.name == name
        else:
            if np.isscalar(X):
                X = (X,)
            X = Parameter(np.array(X, dtype=np.float), name=name, fixed=True)
        return X


def normalize(image):
    """Normalize to PSF image in every band to unity
    """
    sums = image.sum(axis=(1, 2))
    if isinstance(image, Parameter):
        image._data /= sums[:, None, None]
    else:
        image /= sums[:, None, None]
    return image


class GaussianPSF(PSF):
    """Circular Gaussian Function

    Parameters
    ----------
    sigma: float, array, or `~scarlet.Parameter`
        Standard deviation of the Gaussian in frame pixels
        If the width is to be optimized, provide a full defined `Parameter`.
    integrate: bool
        Whether pixel integration is performed
    boxsize: Box
        Size of bounding box over which to evaluate the function, in frame pixels
    """

    def __init__(self, sigma, integrate=True, boxsize=None):

        sigma = self.prepare_param(sigma, "sigma")
        self.integrate = integrate

        if boxsize is None:
            boxsize = int(np.ceil(10 * np.max(sigma)))
        if boxsize % 2 == 0:
            boxsize += 1

        shape = (len(sigma), boxsize, boxsize)
        origin = (0, -(boxsize // 2), -(boxsize // 2))
        self.bbox = Box(shape, origin=origin)

        super().__init__(sigma)

    def get_model(self, *parameters, offset=None):

        sigma = self.get_parameter(0, *parameters)
        Y = np.arange(self.bbox.shape[1]) + self.bbox.origin[1]
        X = np.arange(self.bbox.shape[2]) + self.bbox.origin[2]
        if offset is None:
            offset = (0, 0)

        psfs = np.stack(
            (
                self._f(Y - offset[0], s)[:, None] * self._f(X - offset[1], s)[None, :]
                for s in sigma
            ),
            axis=0,
        )
        # use image integration instead of analytic for consistency with other PSFs
        return normalize(psfs)

    def _f(self, X, sigma):
        if not self.integrate:
            return np.exp(-(X ** 2) / (2 * sigma ** 2))
        else:
            sqrt2 = np.sqrt(2)
            return (
                np.sqrt(np.pi / 2)
                * sigma
                * (
                    1
                    - scipy.special.erfc((0.5 - X) / (sqrt2 * sigma))
                    + 1
                    - scipy.special.erfc((2 * X + 1) / (2 * sqrt2 * sigma))
                )
            )


class MoffatPSF(PSF):
    """Symmetric 2D Moffat function

    .. math::

        (1+\frac{(x-x0)^2+(y-y0)^2}{\alpha^2})^{-\beta}

    Parameters
    ----------
    alpha: float
        Core width, in frame pixels
    beta: float
        Power-law index
    integrate: bool
        Whether pixel integration is performed. Not implemented!
    boxsize: Box
        Size of bounding box over which to evaluate the function, in frame pixels
    """

    def __init__(self, alpha=4.7, beta=1.5, integrate=False, boxsize=None):

        alpha = self.prepare_param(alpha, "alpha")
        beta = self.prepare_param(beta, "beta")
        assert len(alpha) == len(beta)
        assert integrate is False, "In-pixel integration not implemented (yet)!"

        if boxsize is None:
            boxsize = int(np.ceil(5 * np.max(alpha)))
        if boxsize % 2 == 0:
            boxsize += 1

        shape = (len(alpha), boxsize, boxsize)
        origin = (0, -(boxsize // 2), -(boxsize // 2))
        self.bbox = Box(shape, origin=origin)

        super().__init__(alpha, beta)

    def get_model(self, *parameters, offset=None):

        alpha = self.get_parameter(0, *parameters)
        beta = self.get_parameter(1, *parameters)
        Y = np.arange(self.bbox.shape[1]) + self.bbox.origin[1]
        X = np.arange(self.bbox.shape[2]) + self.bbox.origin[2]
        if offset is None:
            offset = (0, 0)

        psfs = np.stack(
            (self._f(Y - offset[0], X - offset[1], a, b) for a, b in zip(alpha, beta)),
            axis=0,
        )
        return normalize(psfs)

    def _f(self, Y, X, a, b):
        # TODO: has no pixel-integration formula
        return ((1 + ((X - x) ** 2 + (Y - y) ** 2) / a ** 2) ** -b)[None, :, :]


class ImagePSF(PSF):
    """Image PSF

    Creates a PSF model from a image of the centered PSF.

    Parameters
    ----------
    image: 2D or 3D array
    """

    def __init__(self, image):

        if len(image.shape) == 2:
            shape = image.shape
            image = image.reshape(1, *shape)

        image = normalize(image)
        image = self.prepare_param(image, "image")
        super().__init__(image)

        origin = (0, -(image.shape[1] // 2), -(image.shape[2] // 2))
        self.bbox = Box(image.shape, origin=origin)

    def get_model(self, *parameters, offset=None):
        image = self.get_parameter(0, *parameters).copy()

        if offset is not None:
            image = shift(image, offset, return_Fourier=False)

        return image

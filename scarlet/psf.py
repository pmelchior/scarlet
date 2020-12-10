import autograd.numpy as np
import autograd.scipy as scipy
from .bbox import Box
from .model import Model, abstractmethod
from .parameter import Parameter
from .fft import Fourier, shift


def normalize(image):
    """Normalize to PSF image in every band to unity
    """
    sums = image.sum(axis=(1, 2))
    if isinstance(image, Parameter):
        image._data /= sums[:, None, None]
    else:
        image /= sums[:, None, None]
    return image


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


class FunctionPSF(PSF):
    """Base class for PSFs with functional forms.

    Parameters
    ----------
    parameters: `~scarlet.Parameter` or list thereof
        Optimization parameters. Can be fixed.
    integrate: bool
        Whether pixel integration is performed
    boxsize: Box
        Size of bounding box over which to evaluate the function, in frame pixels
    """

    def __init__(self, *parameters, integrate=True, boxsize=None):

        super().__init__(*parameters)

        self.integrate = integrate

        if boxsize is None:
            boxsize = 15
        if boxsize % 2 == 0:
            boxsize += 1

        # length of 0 parameter gives number of channels
        p0 = self.get_parameter(0, *parameters)
        shape = (len(p0), boxsize, boxsize)
        origin = (0, -(boxsize // 2), -(boxsize // 2))
        self.bbox = Box(shape, origin=origin)

        self._Y = np.arange(self.bbox.shape[-2]) + self.bbox.origin[-2]
        self._X = np.arange(self.bbox.shape[-1]) + self.bbox.origin[-1]

        # same across all bands
        self.is_same = np.all(p0 == p0[0])
        self._d = self.bbox.D - 2

    def expand_dims(self, model):
        return np.expand_dims(model, axis=tuple(range(self._d)))



class GaussianPSF(FunctionPSF):
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

        if boxsize is None:
            boxsize = int(np.ceil(10 * np.max(sigma)))

        super().__init__(sigma, integrate=integrate, boxsize=boxsize)

    def get_model(self, *parameters, offset=None):

        sigma = self.get_parameter(0, *parameters)

        if offset is None:
            offset = (0, 0)

        if self.is_same:
            s = sigma[0]
            psfs = self.expand_dims(
                self._f(self._Y - offset[0], s)[:, None]
                * self._f(self._X - offset[1], s)[None, :]
            )
        else:
            psfs = np.stack(
                (
                    self._f(self._Y - offset[0], s)[:, None]
                    * self._f(self._X - offset[1], s)[None, :]
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


class MoffatPSF(FunctionPSF):
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

        super().__init__(alpha, beta, integrate=integrate, boxsize=boxsize)

    def get_model(self, *parameters, offset=None):

        alpha = self.get_parameter(0, *parameters)
        beta = self.get_parameter(1, *parameters)

        if offset is None:
            offset = (0, 0)

        if self.is_same:
            a, b = alpha[0], beta[0]
            psfs = self.expand_dims(
                self._f(self._Y - offset[0], self._X - offset[1], a, b)
            )
        else:
            psfs = np.stack(
                (
                    self._f(Y - offset[0], X - offset[1], a, b)
                    for a, b in zip(alpha, beta)
                ),
                axis=0,
            )
        # use image integration instead of analytic for consistency with other PSFs
        return normalize(psfs)

    def _f(self, Y, X, a, b):
        # TODO: has no pixel-integration formula
        return (1 + (X[None, :] ** 2 + Y[:, None] ** 2) / a ** 2) ** -b


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

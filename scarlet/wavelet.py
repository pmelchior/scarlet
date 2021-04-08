import autograd.numpy as np
from . import fft
from .cache import Cache
from scipy.stats import median_absolute_deviation as mad

# Filter for the scarlet transform. Here bspline
h = np.array([1.0 / 16, 1.0 / 4, 3.0 / 8, 1.0 / 4, 1.0 / 16])


class Starlet(object):
    """ A class used to create the Wavelet transform of a cube of images from the 'a trou' algorithm.

        The transform is performed by convolving the image by a seed starlet: the transform of an all-zero
        image with its central pixel set to one. This requires 2-fold padding of the image and an odd pad
        shape. The fft of the seed starlet is cached so that it can be reused in the transform of other
        images that have the same shape.
    """

    def __init__(self, image=None, lvl=None, coefficients=None, direct=True):
        """ Initialise the Starlet object

        Paramters
        image: numpy ndarray
            image to transform
        lvl: int
            number of starlet levels to use in the decomposition
        starlet: array
            Starlet transform of an array
        direct: bool
            if set to True, uses direct wavelet transform with the a trou algorithm.
            if set to False, the transform is performed by convolving the image by the wavelet transform of a dirac.
        """
        self.seed = None
        # Transform method
        self._direct = direct
        if coefficients is None:
            if image is None:
                raise InputError(
                    "At least an image or a set of coefficients should be provided"
                )
            else:
                # Original shape of the image
                self._image_shape = image.shape
                # Padding shape for the starlet transform
                if lvl is None:
                    self._lvl = get_starlet_shape(image.shape)
                else:
                    self._lvl = lvl
                if len(image.shape) == 2:
                    image = image[np.newaxis, :, :]

        else:
            if len(np.shape(coefficients)) == 3:
                coefficients = coefficients[np.newaxis, :, :, :]
            self._image_shape = [coefficients.shape[0], *coefficients.shape[-2:]]
            self._lvl = coefficients.shape[1]
            if image is not None:
                raise InputError(
                    "Ambiguous initialisation: \
                    Starlet objects should be instanciated either with an image of a set of coefficients, not both"
                )

        self._image = image
        self._coeffs = coefficients
        self._starlet_shape = [self._lvl, *self._image_shape[-2:]]
        if self.seed is None:
            self.seed = mk_starlet(self._starlet_shape)
        self._norm = np.sqrt(np.sum(self.seed ** 2, axis=(-2, -1)))

    @property
    def image(self):
        """The real space image"""
        rec = []
        for star in self._coeffs:
            rec.append(iuwt(star))
        self._image = np.array(rec)

        return self._image

    @image.setter
    def image(self, image):
        """Updates the coefficients if the image is changed"""
        if len(image.shape) == 2:
            self._image = image[np.newaxis, :, :]
        else:
            self._image = image
        if self._direct == True:
            self._coeffs = self.direct_transform()
        else:
            self._coeffs = self.transform()

    @property
    def norm(self):
        """The norm of the seed wavelet in each wavelet level (not in coarse wavelet)"""
        return self._norm

    @property
    def coefficients(self):
        """Starlet coefficients"""
        if self._direct == True:
            self._coeffs = self.direct_transform()
        else:
            self._coeffs = self.transform()
        return self._coeffs

    @coefficients.setter
    def coefficients(self, coeffs):
        """Updates the image if the coefficients are changed"""
        if len(np.shape(coeffs)) == 3:
            coeffs = coeffs[np.newaxis, :, :, :]
        self._coeffs = coeffs
        rec = []
        for star in self._coeffs:
            rec.append(iuwt(star))
        self._image = np.array(rec)

    @property
    def shape(self):
        """The shape of the real space image"""
        return self._image.shape

    @property
    def scales(self):
        """Number of starlet scales"""
        return self._lvl

    def transform(self):
        """ Performs the wavelet transform of an image by convolution with the seed wavelet

         Seed wavelets are the transform of a dirac in starlets when computed for a given shape,
         the seed is cached to be reused for images with the same shape.
         The transform is applied to `self._image`

        Returns
        -------
        starlet: numpy ndarray
            the starlet transform of the Starlet object's image
        """
        try:
            # Check if the starlet seed exists
            seed_fft = Cache.check("Starlet", tuple(self._starlet_shape))
        except KeyError:
            # make a starlet seed
            self.seed = mk_starlet(self._starlet_shape)
            # Take its fft
            seed_fft = fft.Fourier(self.seed)
            seed_fft.fft(self._starlet_shape[-2:], (-2, -1))
            # Cache the fft
            Cache.set("Starlet", tuple(self._starlet_shape), seed_fft)
        coefficients = []
        for im in self._image:
            coefficients.append(
                fft.convolve(
                    seed_fft, fft.Fourier(im[np.newaxis, :, :]), axes=(-2, -1)
                ).image
            )
        return np.array(coefficients)

    def direct_transform(self):
        """ Computes the direct starlet transform of the starlet's image

        Returns
        -------
        starlet: numpy ndarray
            the starlet transform of the Starlet object's image
        """
        return mk_starlet(self._starlet_shape, self._image)

    def __len__(self):
        return len(self._image)

    def filter(self, niter=20, k=5):
        """ Applies wavelet iterative filtering to denoise the image

        Parameters
        ----------
        niter: int
            number of iterations
        k: float
            threshold in units of noise levels below which coefficients are thresholded
        lvl: int
            Number of wavelet scale to use in the decomposition

        Results
        -------
        filtered: array
            the image of filtered images
        """
        if self._coeffs is None:
            self.coefficients
        if self._image is None:
            self.image()
        sigma = k * mad_wavelet(self._image)[:, None] * self.norm[None, :]

        filtered = 0
        image = self._image
        wavelet = self._coeffs
        support = np.where(
            np.abs(wavelet[:, :-1, :, :])
            < sigma[:, :-1, None, None] * np.ones_like(wavelet[:, :-1, :, :])
        )
        for i in range(niter):
            R = image - filtered
            R_coeff = Starlet(R)
            R_coeff.coefficients[support] = 0
            filtered += R_coeff.image
            filtered[filtered < 0] = 0
        self.image = filtered
        return filtered


def bspline_convolve(image, scale):
    """Convolve an image with a bpsline at a given scale.

    This uses the spline
    `h1D = np.array([1.0 / 16, 1.0 / 4, 3.0 / 8, 1.0 / 4, 1.0 / 16])`
    from Starck et al. 2011.

    Parameters
    ----------
    image: 2D array
        The image or wavelet coefficients to convolve.
    scale: int
        The wavelet scale for the convolution. This sets the
        spacing between adjacent pixels with the spline.

    """
    # Filter for the scarlet transform. Here bspline
    h1D = np.array([1.0 / 16, 1.0 / 4, 3.0 / 8, 1.0 / 4, 1.0 / 16])
    j = scale

    slice0 = slice(None, -2**(j+1))
    slice1 = slice(None, -2**j)
    slice3 = slice(2**j, None)
    slice4 = slice(2**(j+1), None)
    # row
    col = image * h1D[2]
    col[slice4] += image[slice0] * h1D[0]
    col[slice3] += image[slice1] * h1D[1]
    col[slice1] += image[slice3] * h1D[3]
    col[slice0] += image[slice4] * h1D[4]

    # column
    result = col * h1D[2]
    result[:, slice4] += col[:, slice0] * h1D[0]
    result[:, slice3] += col[:, slice1] * h1D[1]
    result[:, slice1] += col[:, slice3] * h1D[3]
    result[:, slice0] += col[:, slice4] * h1D[4]
    return result


def get_scales(image_shape, scales=None):
    """Get the number of scales to use in the starlet transform.

    Parameters
    ----------
    image_shape: tuple
        The 2D shape of the image that is being transformed
    scales: int
        The number of scale to transform with starlets.
        The total dimension of the starlet will have
        `scales+1` dimensions, since it will also hold
        the image at all scales higher than `scales`.
    """
    # Number of levels for the Starlet decomposition
    max_scale = np.int(np.log2(np.min(image_shape[-2:]))) - 1
    if (scales is None) or scales > max_scale:
        scales = max_scale
    return int(scales)


def starlet_transform(image, scales=None, generation=2, convolve2D=None):
    """Perform a scarlet transform, or 2nd gen starlet transform.

    Parameters
    ----------
    image: 2D array
        The image to transform into starlet coefficients.
    generation: int
        The generation of the transform.
        This must be `1` or `2`.
    convolve2D: function
        The filter function to use to convolve the image
        with starlets in 2D.

    Returns
    -------
    starlet: array with dimension (scales+1, Ny, Nx)
        The starlet dictionary for the input `image`.
    """
    assert len(image.shape) == 2, "Image should be 2D"
    assert generation in (1, 2)

    scales = get_scales(image.shape, scales)
    c = image
    if convolve2D is None:
        convolve2D = bspline_convolve

    ## wavelet set of coefficients.
    starlet = np.zeros((scales + 1,) + image.shape)
    for j in range(scales):
        gen1 = convolve2D(c, j)

        if generation == 2:
            gen2 = convolve2D(gen1, j)
            starlet[j] = c - gen2
        else:
            starlet[j] = c - gen1

        c = gen1

    starlet[-1] = c
    return starlet


def starlet_reconstruction(starlets, convolve2D=None):
    """Reconstruct an image from a dictionary of starlets

    Parameters
    ----------
    starlets: array with dimension (scales+1, Ny, Nx)
        The starlet dictionary used to reconstruct the image.
    convolve2D: function
        The filter function to use to convolve the image
        with starlets in 2D.

    Returns
    -------
    image: 2D array
        The image reconstructed from the input `starlet`.
    """
    if convolve2D is None:
        convolve2D = bspline_convolve
    scales = len(starlets) - 1

    c = starlets[-1]
    for i in range(1, scales + 1):
        j = scales - i
        cj = convolve2D(c, j)
        c = cj + starlets[j]
    return c


def get_multiresolution_support(image, starlets, sigma, K=3, epsilon=1e-1, max_iter=20, image_type="ground"):
    """Calculate the multi-resolution support for a dictionary of starlet coefficients

    This is different for ground and space based telescopes.
    For space-based telescopes the procedure in Starck and Murtagh 1998
    iteratively calculates the multi-resolution support.
    For ground based images, where the PSF is much wider and there are no
    pixels with no signal at all scales, we use a modified method that
    estimates support at each scale independently.

    Parameters
    ----------
    image: 2D array
        The image to transform into starlet coefficients.
    starlets: array with dimension (scales+1, Ny, Nx)
        The starlet dictionary used to reconstruct `image`.
    sigma: float
        The standard deviation of the `image`.
    K: float
        The multiple of `sigma` to use to calculate significance.
        Coefficients `w` where `|w| > K*sigma_j`, where `sigma_j` is
        standard deviation at the jth scale, are considered significant.
    epsilon: float
        The convergence criteria of the algorithm.
        Once `|new_sigma_j - sigma_j|/new_sigma_j < epsilon` the
        algorithm has completed.
    max_iter: int
        Maximum number of iterations to fit `sigma_j` at each scale.
    image_type: str
        The type of image that is being used.
        This should be "ground" for ground based images with wide PSFs or
        "space" for images from space-based telescopes with a narrow PSF.

    Returns
    -------
    M: array of `int`
        Mask with significant coefficients in `starlets` set to `True`.
    """
    assert image_type in ("ground", "space")

    if image_type == "space":
        # Calculate sigma_je, the standard deviation at
        # each scale due to gaussian noise
        shape = (get_scales(image.shape),) + image.shape
        noise_img = np.random.normal(size=image.shape)
        noise_starlet = starlet_transform(shape, noise_img, generation=1)
        sigma_je = np.zeros((len(noise_starlet),))
        for j, star in enumerate(noise_starlet):
            sigma_je[j] = np.std(star)
        noise = image - starlets[-1]

        last_sigma_i = sigma
        for it in range(max_iter):
            M = (np.abs(starlets) > K * sigma * sigma_je[:, None, None])
            S = np.sum(M, axis=0) == 0
            sigma_i = np.std(noise * S)
            if np.abs(sigma_i-last_sigma_i)/sigma_i < epsilon:
                break
            last_sigma_i = sigma_i
    else:
        # Sigma to use for significance at each scale
        # Initially we use the input `sigma`
        sigma_j = np.ones((len(starlets),), dtype=image.dtype) * sigma
        last_sigma_j = sigma_j
        for it in range(max_iter):
            M = (np.abs(starlets) > K * sigma_j[:, None, None])
            # Take the standard deviation of the current insignificant coeffs at each scale
            S = ~M
            sigma_j = np.std(starlets * S.astype(int), axis=(1, 2))
            # At lower scales all of the pixels may be significant,
            # so sigma is effectively zero. To avoid infinities we
            # only check the scales with non-zero sigma
            cut = sigma_j > 0
            if np.all(np.abs(sigma_j[cut] - last_sigma_j[cut]) / sigma_j[cut] < epsilon):
                break

            last_sigma_j = sigma_j
    return M.astype(int)


class InputError(Exception):
    """Exception raised for errors in the input.

    Attributes:
        expression -- input expression in which the error occurred
        message -- explanation of the error
    """

    def __init__(self, message):
        self.message = message


def mad_wavelet(image):
    """ image: Median absolute deviation of the first wavelet scale.
    (WARNING: sorry to disapoint, this is not a wavelet for mad scientists)

    Parameters
    ----------
    image: array
        An image or cube of images
    Returns
    -------
    mad: array
        median absolute deviation for each image in the cube
    """
    sigma = mad(Starlet(image, lvl=2).coefficients[:, 0, ...], axis=(-2, -1))
    return sigma

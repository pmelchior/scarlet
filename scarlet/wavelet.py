import autograd.numpy as np
from . import fft
from . import Cache
from scipy.stats import median_absolute_deviation as mad

# Filter for the scarlet transform. Here bspline
h = np.array([1. / 16, 1. / 4, 3. / 8, 1. / 4, 1. / 16])

class Starlet(object):
    """ A class used to create the Wavelet transform of a cube of images from the 'a trou' algorithm.

        The transform is performed by convolving the image by a seed starlet: the transform of an all-zero
        image with its central pixel set to one. This requires 2-fold padding of the image and an odd pad
        shape. The fft of the seed starlet is cached so that it can be reused in the transform of other
        images that have the same shape.
    """
    def __init__(self, image = None, lvl = None, coefficients = None, direct = True):
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
                raise InputError('At least an image or a set of coefficients should be provided')
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
                raise InputError("Ambiguous initialisation: \
                    Starlet objects should be instanciated either with an image of a set of coefficients, not both")

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
            #Check if the starlet seed exists
            seed_fft = Cache.check('Starlet', tuple(self._starlet_shape))
        except KeyError:
            # make a starlet seed
            self.seed = mk_starlet(self._starlet_shape)
            # Take its fft
            seed_fft = fft.Fourier(self.seed)
            seed_fft.fft(self._starlet_shape[-2:], (-2,-1))
            # Cache the fft
            Cache.set('Starlet', tuple(self._starlet_shape), seed_fft)
        coefficients = []
        for im in self._image:
            coefficients.append(fft.convolve(seed_fft, fft.Fourier(im[np.newaxis, :, :]), axes = (-2,-1)).image)
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


def get_starlet_shape(shape, lvl = None):
    """ Get the pad shape for a starlet transform
    """
    #Number of levels for the Starlet decomposition
    lvl_max = np.int(np.log2(np.min(shape[-2:])))
    if (lvl is None) or lvl > lvl_max:
        lvl = lvl_max
    return lvl

def mk_starlet(shape, image = None):
    """ Creates a starlet for a given 2d shape.

    Parameters
    ----------
    shape: tuple
        2D shape of the desired shapelet
    lvl: int
        number of shapelet levels to compute. If None, lvl is set to the log2 of the number of pixels on a side.
        if lvl is higher than this number lvl will be set to it.

    Returns
    -------
    starlet: Fourier object
        the starlet transform of a Dirac fonction as the `image` of a Fourier object

    """
    lvl, n1, n2 = shape[-3:]

    # Filter size
    n = np.size(h)
    if image is None:
        c = np.zeros((n1,n2))
        c[int(n1/2), int(n2/2)] = 1
    else:
        if len(image.shape) > 2:
            wave = []
            for im in image:
                wave.append(mk_starlet(shape, im))
            return np.array(wave)
        else:
            c = image
    c = fft.Fourier(c)
    ## wavelet set of coefficients.
    wave = np.zeros([lvl, n1, n2])
    for i in np.arange(lvl - 1):
        newh = np.zeros((n + (n - 1) * (2 ** i - 1), 1))
        newh[0::2 ** i, 0] = h
        newhT = fft.Fourier(newh.T)
        newh = fft.Fourier(newh)

        # Calculates c(j+1)
        # Line convolution
        cnew = fft.convolve(c, newh, axes=[0])

        # Column convolution
        cnew = fft.convolve(cnew, newhT, axes=[1])

        ###### hoh for g; Column convolution
        hc = fft.convolve(cnew, newh, axes=[0])

        # hoh for g; Line convolution
        hc = fft.convolve(hc, newhT, axes=[1])

        # wj+1 = cj-hcj+1
        wave[i, :, :] = c.image - hc.image

        c = cnew

    wave[-1, :, :] = c.image
    return wave



def iuwt(starlet):

    """ Inverse starlet transform

    Parameters
    ----------
    starlet: Shapelet object
        Starlet to be inverted

    Returns
    -------
    cJ: array
        a 2D image that corresponds to the inverse transform of stralet.
    """
    lvl, n1, n2 = np.shape(starlet)
    n = np.size(h)
    # Coarse scale
    cJ = fft.Fourier(starlet[-1, :, :])
    for i in np.arange(1, lvl):
        newh = np.zeros((n + (n - 1) * (2 ** (lvl - i - 1) - 1), 1))
        newh[0::2 ** (lvl - i - 1), 0] = h
        newhT = fft.Fourier(newh.T)
        newh = fft.Fourier(newh)

        # Line convolution
        cnew = fft.convolve(cJ, newh, axes=[0])
        # Column convolution
        cnew = fft.convolve(cnew, newhT, axes=[1])

        cJ = fft.Fourier(cnew.image + starlet[lvl - 1 - i, :, :])

    return np.reshape(cJ.image, (n1, n2))


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
        median absolute deviation each image in the cube
    """
    sigma = mad(Starlet(image, lvl = 2).coefficients[:,0,...], axis = (-2,-1))
    return sigma
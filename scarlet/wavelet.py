import autograd.numpy as np
from . import fft
from scipy import fftpack
import scipy.ndimage.filters as sc

# Filter for the scarlet transform. Here bspline
h = np.array([1. / 16, 1. / 4, 3. / 8, 1. / 4, 1. / 16])

class Starlet(object):
    """ A class used to create the Wavelet transform of an array
    """
    def __init__(self, image, transform = None):
        """ Initialise the Starlet object

        Paramters
        image: array
            image to transform
        transform: array
            Starlet transform of an array
        """
        if transform is None:
            self._transform = {}
        else:
            self._transform = transform
        self.image = image

    @property
    def image(self):
        """The real space image"""
        return self._image

    @property
    def shape(self):
        """The shape of the real space image"""
        return self.image.shape

    @staticmethod
    def from_transform(transform, shape = None):
        """ Creates a Starlet object from its transform and uses the inverse transform to create the image.

        """

    def transform(self):
        """

        """


def mk_starlet(shape, lvl = None):
    """ Creates a starlet for a given 2d shape.

    Parameters
    ----------
    shape: tuple
        2D shape of the desired shapelet
    lvl: int
        number of shapelet levels to compute. If None, lvl is set to the log2 of the number of pixels on a side.
        if lvl is higher than this number lvl will be set to it.

    Returns:
    starlet: array
        the starlet transform of a Dirac fonction

    """
    #Number of levels for the Starlet decomposition
    lvl_max = np.int(np.log2(np.min(shape)))
    if (lvl is None) or lvl > lvl_max:
        lvl = lvl_max

    # FFT shape (2-fold padding)
    fft_shape = [0,0]
    for i in range(len(shape)):
        fft_shape[i] = fftpack.helper.next_fast_len(2*shape[i])
        while fft_shape[i] % 2 == 0:
            fft_shape[i] = fftpack.helper.next_fast_len(fft_shape[i]+1)
    n1, n2 = fft_shape

    # Filter size
    n = np.size(h)

    c = np.zeros((n1,n2))
    c[int(n1/2), int(n2/2)] = 1
    ## wavelet set of coefficients.
    wave = np.zeros([lvl, n1, n2])

    for i in np.arange(lvl - 1):
        newh = np.zeros(n + i * (n - 1))
        newh[0::i + 1] = h

        # Calculates c(j+1)
        # Line convolution
        cnew = sc.convolve1d(c, newh, axis=0, mode='nearest')

        # Column convolution
        cnew = sc.convolve1d(cnew, newh, axis=1, mode='nearest')

        ###### hoh for g; Column convolution
        hc = sc.convolve1d(cnew, newh, axis=0, mode='nearest')

        # hoh for g; Line convolution
        hc = sc.convolve1d(hc, newh, axis=1, mode='nearest')

        # wj+1 = cj-hcj+1
        wave[i, :, :] = c - hc

        c = cnew

    wave[-1, :, :] = c

    return wave


def iuwt(starlet, shape = None):
    """ Inverse starlet transform

    Parameters
    ----------
    starlet: Shapelet object
        Savelet to be inverted
    shape: tuple
        Original shape of the image to reconstruct. If set to None, the shape of the last two axes of starlet are used
    """
    # Shape of the image to reconstruct
    if shape is None:
        shape = np.shape(starlet)[-2:]
    lvl = np.shape(starlet)[0]

    # Shape of the filter
    n = np.size(h)

    # Reconstruction starts from the las scale
    cJ = np.copy(starlet[- 1, :, :])
    for i in range(lvl):

        newh = np.zeros(n + i * (n - 1))
        newh[0::i + 1] = h

        # Line convolution
        cnew = sc.convolve1d(cJ, newh, axis=0, mode='nearest')
        # Column convolution
        cnew = sc.convolve1d(cnew, newh, axis=1, mode='nearest')

        cJ = cnew + starlet[lvl - i - 1, :, :]

    return fft._centered(cJ, shape)


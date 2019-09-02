import operator

import numpy as np
from scipy import fftpack


def _centered(arr, newshape):
    """Return the center newshape portion of the array.

    This function is used by `fft_convolve` to remove
    the zero padded region of the convolution.

    Note: If the array shape is odd and the target is even,
    the center of `arr` is shifted to the center-right
    pixel position.
    This is slightly different than the scipy implementation,
    which uses the center-left pixel for the array center.
    The reason for the difference is that we have
    adopted the convention of `np.fft.fftshift` in order
    to make sure that changing back and forth from
    fft standard order (0 frequency and position is
    in the bottom left) to 0 position in the center.
    """
    newshape = np.asarray(newshape)
    currshape = np.array(arr.shape)

    if not np.all(newshape <= currshape):
        msg = "arr must be larger than newshape in both dimensions, received {0}, and {1}"
        raise ValueError(msg.format(arr.shape, newshape))

    startind = (currshape - newshape+1) // 2
    endind = startind + newshape
    myslice = [slice(startind[k], endind[k]) for k in range(len(endind))]

    return arr[tuple(myslice)]


def _pad(arr, newshape, axes=None):
    """Pad an array to fit into newshape

    Pad `arr` with zeros to fit into newshape,
    which uses the `np.fft.fftshift` convention of moving
    the center pixel of `arr` (if `arr.shape` is odd) to
    the center-right pixel in an even shaped `newshape`.
    """
    if axes is None:
        newshape = np.asarray(newshape)
        currshape = np.array(arr.shape)
        dS = newshape - currshape
        startind = (dS+1) // 2
        endind = dS - startind
        pad_width = list(zip(startind, endind))
    else:
        # only pad the axes that will be transformed
        pad_width = [(0, 0) for axis in arr.shape]
        for a, axis in enumerate(axes):
            dS = newshape[a] - arr.shape[axis]
            startind = (dS+1) // 2
            endind = dS - startind
            pad_width[axis] = (startind, endind)
    return np.pad(arr, pad_width, mode="constant")


def _get_fft_axes(spectral_axis=0):
    """Return the axes used for the fft

    Given the dimension that contains spectral information,
    return the `axes` required by `np.fft.fftn` to only
    Fourier Transform the spatial dimensions.
    """
    if spectral_axis is not None:
        axes = [0, 1, 2]
        axes.remove(spectral_axis)
    else:
        axes = None
    return axes


def _get_fft_shape(img1, img2, padding=3, spectral_axis=0):
    """Return the fast fft shapes for each spatial axis

    Given the axis that contains the spectral information
    (`spectral_axis`), calculate the fast fft shape
    for each remaining dimension.
    """
    shape1 = np.asarray(img1.shape)
    shape2 = np.asarray(img2.shape)
    # Make sure the shapes are the same size
    if len(shape1) != len(shape2):
        msg = "img1 and img2 must have the same number of dimensions, but got {0} and {1}"
        raise ValueError(msg.format(len(shape1, len(shape2))))
    # Set the combined shape based on the total dimensions
    print("spectral_axis", spectral_axis)
    if spectral_axis == 0:
        print("spectral_axis 0")
        shape = shape1[1:] + shape2[1:]
    elif spectral_axis == 2:
        shape = shape1[:-1] + shape2[:-1]
    elif spectral_axis is None:
        shape = shape1 + shape2
    else:
        raise ValueError("spectral_axis should be 0, 2 or None, got {0}".format(spectral_axis))
    print("shape is", shape)
    shape += padding
    # Use the next fastest shape in each dimension
    shape = [fftpack.helper.next_fast_len(s) for s in shape]
    # autograd.numpy.fft does not currently work
    # if the last dimension is odd
    while shape[-1] % 2 != 0:
        shape[-1] += 1
        shape[-1] = fftpack.helper.next_fast_len(shape[-1])
    return shape


class Fourier(object):
    """An array that stores its Fourier Transform

    The `Fourier` class is used for images that will make
    use of their Fourier Transform multiple times.
    In order to prevent numerical artifacts the same image
    convolved with different images might require different
    padding, so the FFT for each different shape is stored
    in a dictionary.
    """
    def __init__(self, image, image_fft=None, spectral_axis=0):
        """Initialize the object

        Parameters
        ----------
        image: array
            The real space image.
        image_fft: dict
            A dictionary of {shape: fft_value} for which each different
            shape has a precalculated FFT.
        spectral_axis: int
            The dimension of the array that contains spectral information.
            If `spectral_axis` is `None` then there is no spectral axis.
        """
        if image_fft is None:
            self._fft = {}
        else:
            self._fft = image_fft
        self._image = image
        self._spectral_axis = spectral_axis

    @staticmethod
    def from_fft(image_fft, fft_shape, image_shape, spectral_axis=0):
        """Generate a new Fourier object from an FFT dictionary

        If the fft of an image has been generated but not its
        real space image (for example when creating a convolution kernel),
        this method can be called to create a new `Fourier` instance
        from the k-space representation.

        Parameters
        ----------
        image_fft: array
            The FFT of the image.
        fft_shape: tuple
            Shape of the image used to generate the FFT.
            This will be different than `image_fft.shape` if
            any of the dimensions are odd, since `np.fft.rfft`
            requires an even number of dimensions (for symmetry),
            so this tells `np.fft.irfft` how to go from
            complex k-space to real space.
        image_shape: tuple
            The shape of the image *before padding*.
            This will regenerate the image with the extra
            padding stripped.
        spectral_axis: int
            The dimension of the array that contains spectral information.
            If `image` only has 2 dimensions then this argument is ignored.

        Returns
        -------
        result: `Fourier`
            A `Fourier` object generated from the FFT.
        """
        axes = _get_fft_axes(spectral_axis)
        image = np.fft.irfftn(image_fft, fft_shape, axes=axes)
        # Shift the center of the image from the bottom left to the center
        image = np.fft.fftshift(image)
        # Trim the image to remove the padding added
        # to reduce fft artifacts
        image = _centered(image, image_shape)
        return Fourier(image, {tuple(fft_shape): image_fft}, spectral_axis)

    @property
    def image(self):
        """The real space image"""
        return self._image

    @property
    def spectral_axis(self):
        """The dimension of the image containing spectral information"""
        return self._spectral_axis

    @property
    def shape(self):
        """The shape of the real space image"""
        return self.image.shape

    def fft(self, fft_shape):
        """The FFT of an image for a given `fft_shape`
        """
        fft_shape = tuple(fft_shape)
        # If this is the first time calling `fft` for this shape,
        # generate the FFT.
        if fft_shape not in self._fft:
            axes = _get_fft_axes(self.spectral_axis)
            print("axes:", axes)
            image = _pad(self.image, fft_shape, axes)
            print("image shape", image.shape)
            self._fft[fft_shape] = np.fft.rfftn(np.fft.ifftshift(image, axes), axes=axes)
        return self._fft[fft_shape]


def _kspace_operation(image1, image2, padding, operator, shape):
    """Combine two images in k-space using a given `operator`"""
    if image1.spectral_axis != image2.spectral_axis:
        msg = "Both images must have the same spectral dimension, received {0} and {1}"
        msg.format(image1.spectral_axis, image2.spectral_axis)
        raise ValueError(msg)
    fft_shape = _get_fft_shape(image1.image, image2.image, padding, image1.spectral_axis)
    convolved_fft = operator(image1.fft(fft_shape), image2.fft(fft_shape))
    convolved = Fourier.from_fft(convolved_fft, fft_shape, shape, image1.spectral_axis)
    return convolved


def match_psfs(psf1, psf2, padding=3):
    """Calculate the difference kernel between two psfs

    Parameters
    ----------
    psf1: `Fourier`
        `Fourier` object represeting the psf and it's FFT.
    psf2: `Fourier`
        `Fourier` object represeting the psf and it's FFT.
    padding: int
        Additional padding to use when generating the FFT
        to supress artifacts.
    """
    if psf1.shape[psf1.spectral_axis] < psf2.shape[psf2.spectral_axis]:
        shape = psf2.shape
    else:
        shape = psf1.shape
    return _kspace_operation(psf1, psf2, padding, operator.truediv, shape)


def convolve(image1, image2, padding=3):
    """Convolve two images

    Parameters
    ----------
    image1: `Fourier`
        `Fourier` object represeting the image and it's FFT.
    image2: `Fourier`
        `Fourier` object represeting the image and it's FFT.
    padding: int
        Additional padding to use when generating the FFT
        to supress artifacts.
    """
    return _kspace_operation(image1, image2, padding, operator.mul, image1.shape)

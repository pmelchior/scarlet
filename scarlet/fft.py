import operator

import autograd.numpy as np
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
        try:
            len(axes)
        except TypeError:
            axes = [axes]
        for a, axis in enumerate(axes):
            dS = newshape[a] - arr.shape[axis]
            startind = (dS+1) // 2
            endind = dS - startind
            pad_width[axis] = (startind, endind)
    return np.pad(arr, pad_width, mode="constant")


def _get_fft_shape(img1, img2, padding=3, axes=None):
    """Return the fast fft shapes for each spatial axis

    Calculate the fast fft shape for each dimension in
    axes.
    """
    shape1 = np.asarray(img1.shape)
    shape2 = np.asarray(img2.shape)
    # Make sure the shapes are the same size
    if len(shape1) != len(shape2):
        msg = "img1 and img2 must have the same number of dimensions, but got {0} and {1}"
        raise ValueError(msg.format(len(shape1, len(shape2))))
    # Set the combined shape based on the total dimensions
    if axes is None:
        shape = shape1 + shape2
    else:
        shape = np.zeros(len(axes))
        try:
            len(axes)
        except TypeError:
            axes = [axes]
        for n, ax in enumerate(axes):
            shape[n] = shape1[ax] + shape2[ax]

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
    def __init__(self, image, image_fft=None, axes=None):
        """Initialize the object

        Parameters
        ----------
        image: array
            The real space image.
        image_fft: dict
            A dictionary of {shape: fft_value} for which each different
            shape has a precalculated FFT.
        axes: int or tuple
            The dimension(s) of the array that will be transformed.
        """
        if image_fft is None:
            self._fft = {}
        else:
            self._fft = image_fft
        self._image = image
        self._axes = axes

    @staticmethod
    def from_fft(image_fft, fft_shape, image_shape, axes=None):
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
        axes: int or tuple
            The dimension(s) of the array that will be transformed.

        Returns
        -------
        result: `Fourier`
            A `Fourier` object generated from the FFT.
        """
        image = np.fft.irfftn(image_fft, fft_shape, axes=axes)
        # Shift the center of the image from the bottom left to the center
        image = np.fft.fftshift(image, axes=axes)
        # Trim the image to remove the padding added
        # to reduce fft artifacts
        image = _centered(image, image_shape)
        return Fourier(image, {tuple(fft_shape): image_fft}, axes)

    @property
    def image(self):
        """The real space image"""
        return self._image

    @property
    def axes(self):
        """The axes that are transormed"""
        return self._axes

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
            image = _pad(self.image, fft_shape, self._axes)
            self._fft[fft_shape] = np.fft.rfftn(np.fft.ifftshift(image, self._axes), axes=self._axes)
        return self._fft[fft_shape]

    def __len__(self):
        return len(self.image)

    def normalize(self):
        """Normalize the image to sum to one
        """
        indices = [slice(None)] * len(self.shape)
        for ax in self._axes:
            indices[ax] = None
        indices = tuple(indices)
        normalization = 1/self._image.sum(axis=self._axes)
        self._image *= normalization[indices]
        for shape, image_fft in self._fft.items():
            self._fft[shape] *= normalization[indices]

    def sum(self, axis=None):
        return self.image.sum(axis)

    def max(self, axis=None):
        return self.image.max(axis=axis)

    def __getitem__(self, index):
        return self.image[index]


def _kspace_operation(image1, image2, padding, operator, shape):
    """Combine two images in k-space using a given `operator`"""
    if image1.axes != image2.axes:
        msg = "Both images must have the same axes, got {0} and {1}".format(image1.axes, image2.axes)
        raise Exception(msg)
    fft_shape = _get_fft_shape(image1.image, image2.image, padding, image1.axes)
    convolved_fft = operator(image1.fft(fft_shape), image2.fft(fft_shape))
    convolved = Fourier.from_fft(convolved_fft, fft_shape, shape, image1.axes)
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
    axes: tuple or None
        Axes that contain the spatial information for the PSFs.
    """
    if psf1.shape[0] < psf2.shape[0]:
        shape = psf2.shape
    else:
        shape = psf1.shape
    return _kspace_operation(psf1, psf2, padding, operator.truediv, shape)


def convolve(image1, image2, padding=3, axes=None):
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

import operator

import autograd.numpy as np
from autograd.extend import primitive, defvjp
from scipy import fftpack
from .interpolation import mk_shifter


def _centered(arr, newshape):
    """Return the center newshape portion of the array.

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
        msg = (
            "arr must be larger than newshape in both dimensions, received {0}, and {1}"
        )
        raise ValueError(msg.format(arr.shape, newshape))

    startind = (currshape - newshape + 1) // 2
    endind = startind + newshape
    myslice = [slice(startind[k], endind[k]) for k in range(len(endind))]

    return arr[tuple(myslice)]


@primitive
def fast_zero_pad(arr, pad_width):
    """Fast version of numpy.pad when `mode="constant"`

    Executing `numpy.pad` with zeros is ~1000 times slower
    because it doesn't make use of the `zeros` method for padding.

    Paramters
    ---------
    arr: array
        The array to pad
    pad_width: tuple
        Number of values padded to the edges of each axis.
        See numpy docs for more.

    Returns
    -------
    result: array
        The array padded with `constant_values`
    """
    newshape = tuple([a + ps[0] + ps[1] for a, ps in zip(arr.shape, pad_width)])

    result = np.zeros(newshape, dtype=arr.dtype)
    slices = tuple(
        [slice(start, s - end) for s, (start, end) in zip(result.shape, pad_width)]
    )
    result[slices] = arr
    return result


def _fast_zero_pad_grad(result, arr, pad_width):
    """Gradient for fast_zero_pad
    """
    slices = tuple(
        [slice(start, s - end) for s, (start, end) in zip(result.shape, pad_width)]
    )
    return lambda grad_chain: grad_chain[slices]


# Register this function in autograd
defvjp(fast_zero_pad, _fast_zero_pad_grad)


def _pad(arr, newshape, axes=None, mode="constant", constant_values=0):
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
        startind = (dS + 1) // 2
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
            startind = (dS + 1) // 2
            endind = dS - startind
            pad_width[axis] = (startind, endind)
    if mode == "constant" and constant_values == 0:
        result = fast_zero_pad(arr, pad_width)
    else:
        result = np.pad(arr, pad_width, mode=mode)
    return result


def _get_fft_shape(im_or_shape1, im_or_shape2, padding=3, axes=None, max=False):
    """Return the fast fft shapes for each spatial axis

    Calculate the fast fft shape for each dimension in
    axes.
    """
    if hasattr(im_or_shape1, "shape"):
        shape1 = np.asarray(im_or_shape1.shape)
    else:
        shape1 = np.asarray(im_or_shape1)
    if hasattr(im_or_shape2, "shape"):
        shape2 = np.asarray(im_or_shape2.shape)
    else:
        shape2 = np.asarray(im_or_shape2)
    # Make sure the shapes are the same size
    if len(shape1) != len(shape2):
        msg = (
            "img1 and img2 must have the same number of dimensions, but got {0} and {1}"
        )
        raise ValueError(msg.format(len(shape1), len(shape2)))
    # Set the combined shape based on the total dimensions
    if axes is None:
        if max:
            shape = np.max([shape1, shape2], axis=1)
        else:
            shape = shape1 + shape2
    else:
        shape = np.zeros(len(axes), dtype="int")
        try:
            len(axes)
        except TypeError:
            axes = [axes]
        for n, ax in enumerate(axes):
            shape[n] = shape1[ax] + shape2[ax]
            if max == True:
                shape[n] = np.max([shape1[ax], shape2[ax]])

    shape += padding
    # Use the next fastest shape in each dimension
    shape = [fftpack.helper.next_fast_len(s) for s in shape]

    # autograd.numpy.fft does not currently work
    # if the last dimension is odd
    while shape[-1] % 2 != 0:
        shape[-1] += 1
        shape[-1] = fftpack.helper.next_fast_len(shape[-1])
    if shape2[-2] % 2 == 0:
        while shape[-2] % 2 != 0:
            shape[-2] += 1
            shape[-2] = fftpack.helper.next_fast_len(shape[-2])

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

    def __init__(self, image, image_fft=None):
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
            "Fast" shape of the image used to generate the FFT.
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
        if axes is None:
            axes = range(len(image_fft))
        all_axes = range(len(image_shape))
        image = np.fft.irfftn(image_fft, fft_shape, axes=axes)
        # Shift the center of the image from the bottom left to the center
        image = np.fft.fftshift(image, axes=axes)
        # Trim the image to remove the padding added
        # to reduce fft artifacts
        image = _centered(image, image_shape)
        key = (tuple(fft_shape), tuple(axes), tuple(all_axes))

        return Fourier(image, {key: image_fft})

    @property
    def image(self):
        """The real space image"""
        return self._image

    @property
    def shape(self):
        """The shape of the real space image"""
        return self._image.shape

    def fft(self, fft_shape, axes):
        """The FFT of an image for a given `fft_shape` along desired `axes`
        """
        try:
            iter(axes)
        except TypeError:
            axes = (axes,)
        all_axes = range(len(self.image.shape))
        fft_key = (tuple(fft_shape), tuple(axes), tuple(all_axes))

        # If this is the first time calling `fft` for this shape,
        # generate the FFT.
        if fft_key not in self._fft:
            if len(fft_shape) != len(axes):
                msg = "fft_shape self.axes must have the same number of dimensions, got {0}, {1}"
                raise ValueError(msg.format(fft_shape, axes))
            image = _pad(self.image, fft_shape, axes)
            self._fft[fft_key] = np.fft.rfftn(np.fft.ifftshift(image, axes), axes=axes)
        return self._fft[fft_key]

    def __len__(self):
        return len(self.image)

    def __getitem__(self, index):
        # Make the index a tuple
        if not hasattr(index, "__getitem__"):
            index = tuple([index])

        # Axes that are removed from the shape of the new object
        removed = np.array(
            [
                n
                for n, idx in enumerate(index)
                if not isinstance(idx, slice) and idx is not None
            ]
        )

        # Create views into the fft transformed values, appropriately adjusting
        # the shapes for the new axes

        fft_kernels = {
            (
                tuple(
                    [s for idx, s in enumerate(key[0]) if key[1][idx] not in removed]
                ),
                tuple(
                    [a for ida, a in enumerate(key[1]) if key[1][ida] not in removed]
                ),
                tuple(
                    [
                        aa
                        for idaa, aa in enumerate(key[2])
                        if key[2][idaa] not in removed
                    ]
                ),
            ): kernel[index]
            for key, kernel in self._fft.items()
        }
        return Fourier(self.image[index], fft_kernels)


def _kspace_operation(image1, image2, padding, op, shape, axes):
    """Combine two images in k-space using a given `operator`

    `image1` and `image2` are required to be `Fourier` objects and
    `op` should be an operator (either `operator.mul` for a convolution
    or `operator.truediv` for deconvolution). `shape` is the shape of the
    output image (`Fourier` instance).
    """
    if len(image1.shape) != len(image2.shape):
        msg = "Both images must have the same number of axes, got {0} and {1}"
        raise Exception(msg.format(len(image1.shape), len(image2.shape)))

    fft_shape = _get_fft_shape(image1.image, image2.image, padding, axes)
    transformed_fft = op(image1.fft(fft_shape, axes), image2.fft(fft_shape, axes))
    # why is shape not image1.shape? images are never padded
    return Fourier.from_fft(transformed_fft, fft_shape, shape, axes)


def match_psfs(psf1, psf2, padding=3, axes=(-2, -1), return_Fourier=True):
    """Calculate the difference kernel to match psf1 to psf2

    Parameters
    ----------
    psf1: array or `Fourier`
        PSF1 either as array or as `Fourier` object
    psf2: array or `Fourier`
        PSF1 either as array or as `Fourier` object
    padding: int
        Additional padding to use when generating the FFT
        to supress artifacts.
    axes: tuple or None
        Axes that contain the spatial information for the PSFs.
    return_Fourier: bool
        Whether to return `Fourier` or array
    """
    if not isinstance(psf1, Fourier):
        psf1 = Fourier(psf1)
    if not isinstance(psf2, Fourier):
        psf2 = Fourier(psf2)

    if psf1.shape[0] < psf2.shape[0]:
        shape = psf2.shape
    else:
        shape = psf1.shape

    diff = _kspace_operation(psf1, psf2, padding, operator.truediv, shape, axes=axes)
    if return_Fourier:
        return diff
    else:
        return np.real(diff.image)


def convolve(image, kernel, padding=3, axes=(-2, -1), return_Fourier=True):
    """Convolve image with a kernel

    Parameters
    ----------
    image: array or `Fourier`
        Image either as array or as `Fourier` object
    kernel: array or `Fourier`
        Convolution kernel either as array or as `Fourier` object
    padding: int
        Additional padding to use when generating the FFT
        to supress artifacts.
    axes: tuple or None
        Axes that contain the spatial information for the PSFs.
    return_Fourier: bool
        Whether to return `Fourier` or array
    """
    if not isinstance(image, Fourier):
        image = Fourier(image)
    if not isinstance(kernel, Fourier):
        kernel = Fourier(kernel)

    convolved = _kspace_operation(
        image, kernel, padding, operator.mul, image.shape, axes=axes
    )
    if return_Fourier:
        return convolved
    else:
        return np.real(convolved.image)


def shift(image, shift, fft_shape=None, axes=(-2, -1), return_Fourier=True):

    if fft_shape is None:
        padding = 10
        fft_shape = _get_fft_shape(image, image, padding=padding, axes=axes)

    shifter_y, shifter_x = mk_shifter(fft_shape)  # is cached!

    if not isinstance(image, Fourier):
        image = Fourier(image)

    image_fft = image.fft(fft_shape, axes)

    # Apply shift in Fourier
    D = len(image.shape)
    shifter = np.exp(shifter_y[:, None] * shift[0]) * np.exp(
        shifter_x[None, :] * shift[1]
    )
    if D > 2:
        expand_dims = tuple(d for d in range(D) if d not in axes and d - D not in axes)
        shifter = np.expand_dims(shifter, axis=expand_dims)

    result_fft = image_fft * shifter

    result = Fourier.from_fft(result_fft, fft_shape, image.shape, axes)

    if return_Fourier:
        return result
    else:
        return np.real(result.image)

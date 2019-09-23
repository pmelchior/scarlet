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
        raise ValueError(msg.format(len(shape1), len(shape2)))
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


def rect_window(freq, freq_limit):
    """Use a rectangular window to filter the signal

    `freq_limit` is the maximum frequency that will pass through the filter.
    """
    window = np.zeros(freq.shape, dtype=freq.dtype)
    window[(abs(freq) < freq_limit)] = 1
    return window


def kaiser_window(freq, alpha=3):
    """Use a Kaiser filtering window
    """
    nN = freq
    return np.i0(np.pi*alpha * np.sqrt(1-(nN)**2)) / np.i0(np.pi*alpha)


def hann_window(freq):
    """Use a Hann filtering window
    """
    nN = freq
    return np.cos(np.pi*nN)**2


def tukey_window(freq, alpha=.5):
    """Use a Tukey filtering window
    """
    nN = np.abs(freq)

    window = np.ones(freq.shape, dtype=freq.dtype)
    omega = 2*nN/alpha
    top = 0.5 * (1+np.cos(np.pi*(omega - 1/alpha + 1)))
    cut = nN > 0.5*(1-alpha)
    window[cut] = top[cut]
    return window


def product_window(func, shape, **kwargs):
    """Implement a 2D window as the product of two 1D windows

    Parameters
    ----------
    func: `Function`
        The filtering function in the x and y directions
    shape: tuple
        The shape of the real-space image.
        This is used to set the frequencies
        for the window.
    kwargs: dict
        Keyword arguments for `func`.

    Returns
    -------
    result: 2D array
        A window that can be used to filter the signal.
    """
    fy = np.fft.fftfreq(shape[0])
    fx = np.fft.fftfreq(shape[1])
    fx, fy = np.meshgrid(fx, fy)
    wy = func(fy, **kwargs)
    wx = func(fx, **kwargs)
    return wy * wx


def symmetric_window(func, shape, **kwargs):
    """Implement a 2D window that is circularly symmetric

    Parameters
    ----------
    func: `Function`
        The filtering function in the x and y directions
    shape: tuple
        The shape of the real-space image.
        This is used to set the frequencies
        for the window.
    kwargs: dict
        Keyword arguments for `func`.

    Returns
    -------
    result: 2D array
        A window that can be used to filter the signal.
    """
    fy = np.fft.fftshift(np.fft.fftfreq(shape[0]))
    fx = np.fft.fftshift(np.fft.fftfreq(shape[1]))
    fx, fy = np.meshgrid(fx, fy)
    r = np.sqrt(fy**2 + fx**2)
    result = func(r, **kwargs)
    # Set the window outside of the x, y window to zero
    # This prevents periodic windows from cycling
    # back up at the diagonal edges
    result[(r > np.abs(fy).max()) | (r > np.abs(fx).max())] = 0
    return result


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
        if axes is None:
            axes = tuple(range(len(self.shape)))
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
            if len(fft_shape) != len(self.axes):
                msg = "fft_shape self.axes must have the same number of dimensions, got {0}, {1}"
                raise ValueError(msg.format(fft_shape, self.axes))
            image = _pad(self.image, fft_shape, self._axes)
            self._fft[fft_shape] = np.fft.rfftn(np.fft.ifftshift(image, self._axes), axes=self._axes)
        return self._fft[fft_shape]

    def __len__(self):
        return len(self.image)

    def normalize(self):
        """Normalize the image to sum to one
        """
        if self._axes is not None:
            indices = [slice(None)] * len(self.shape)
            for ax in self._axes:
                indices[ax] = None
        else:
            indices = [None] * len(self.shape)
        indices = tuple(indices)
        normalization = 1/self._image.sum(axis=self._axes)
        self._image *= normalization[indices]
        for shape, image_fft in self._fft.items():
            self._fft[shape] *= normalization[indices]

    def update_dtype(self, dtype):
        if self.image.dtype != dtype:
            self._image = self._image.astype(dtype)
            for shape in self._fft:
                self._fft[shape] = self._fft[shape].astype(dtype)

    def sum(self, axis=None):
        return self.image.sum(axis)

    def max(self, axis=None):
        return self.image.max(axis=axis)

    def __getitem__(self, index):
        # Make the index a tuple
        if not hasattr(index, "__iter__"):
            index = tuple([index])

        # Axes that are removed from the shape of the new object
        removed = np.array([n for n, idx in enumerate(index)
                            if not isinstance(idx, slice) and idx is not None])
        # Axes that are added to the shape of the new object
        # (with `np.newaxis` or `None`)
        added = np.array([n for n, idx in enumerate(index) if idx is None])

        # Only propagate axes that are sliced or not indexed and
        # decrement them by the number of removed axes smaller than each one
        # and increment them by the number of added axes smaller than
        # each index.
        axes = tuple([ax-np.sum(removed < ax)+np.sum(added <= ax) for ax in self.axes if ax not in removed])

        # Create views into the fft transformed values, appropriately adjusting
        # the shapes for the new axes
        fft_kernels = {
            tuple([s for idx, s in enumerate(shape) if self.axes[idx] not in removed]): kernel[index]
            for shape, kernel in self._fft.items()
        }
        # If all of the remaining axes are used then remove the
        # axes dependence
        if axes == tuple(np.arange(len(self.image[index].shape))):
            axes = None
        return Fourier(self.image[index], fft_kernels, axes=axes)


def _kspace_operation(image1, image2, padding, op, shape, window=None):
    """Combine two images in k-space using a given `operator`

    `image1` and `image2` are required to be `Fourier` objects and
    `op` should be an operator (either `operator.mul` for a convolution
    or `operator.truediv` for deconvolution). `shape` is the shape of the
    output image (`Fourier` instance).
    """
    if image1.axes != image2.axes:
        msg = "Both images must have the same axes, got {0} and {1}".format(image1.axes, image2.axes)
        raise Exception(msg)
    fft_shape = _get_fft_shape(image1.image, image2.image, padding, image1.axes)
    convolved_fft = op(image1.fft(fft_shape), image2.fft(fft_shape))
    if window is not None:
        _fft_shape = tuple([a for idx, a in enumerate(convolved_fft.shape) if idx in image1.axes])
        # Since mathcing uses real FFTs we need to chop the
        # window and reshape it to be the same size as the image
        if window.shape != _fft_shape:
            cx = (window.shape[1]-1) // 2
            window[:, :cx] = 0
            window = _pad(window, _fft_shape)
            window = np.fft.ifftshift(window)
        convolved_fft *= window
    convolved = Fourier.from_fft(convolved_fft, fft_shape, shape, image1.axes)
    return convolved


def match_psfs(psf1, psf2, padding=3, window=None):
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
    return _kspace_operation(psf1, psf2, padding, operator.truediv, shape, window)


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

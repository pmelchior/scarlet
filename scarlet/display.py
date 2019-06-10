import numpy as np


def compute_intensity(image_r, image_g=None, image_b=None):
    """
    Return a naive total intensity from the red, blue, and green intensities.
    Parameters
    ----------
    image_r : `~numpy.ndarray`
        Intensity of image to be mapped to red; or total intensity if
        ``image_g`` and ``image_b`` are None.
    image_g : `~numpy.ndarray`, optional
        Intensity of image to be mapped to green.
    image_b : `~numpy.ndarray`, optional
        Intensity of image to be mapped to blue.
    Returns
    -------
    intensity : `~numpy.ndarray`
        Total intensity from the red, blue and green intensities,
         or ``image_r`` if green and blue images are not provided.
    """
    if image_g is None or image_b is None:
        if not (image_g is None and image_b is None):
            raise ValueError("please specify either a single image "
                             "or red, green, and blue images.")
        return image_r

    intensity = (image_r + image_g + image_b)/3.0

    # Repack into whatever type was passed to us
    return np.asarray(intensity, dtype=image_r.dtype)


class Mapping:
    """
    Baseclass to map red, blue, green intensities into uint8 values.
    Parameters
    ----------
    minimum : float or sequence(3)
        Intensity that should be mapped to black
        (a scalar or array for R, G, B).
    image : `~numpy.ndarray`, optional
        An image used to calculate some parameters of some mappings.
    """

    def __init__(self, minimum=None, image=None):
        self._uint8Max = float(np.iinfo(np.uint8).max)

        try:
            len(minimum)
        except TypeError:
            minimum = 3*[minimum]
        if len(minimum) != 3:
            raise ValueError("please provide 1 or 3 values for minimum.")

        self.minimum = minimum
        self._image = np.asarray(image)

    def make_rgb_image(self, image_r, image_g, image_b):
        """Convert 3 arrays into an 8-bit RGB image

        Parameters
        ----------
        image_r : `~numpy.ndarray`
            Image to map to red.
        image_g : `~numpy.ndarray`
            Image to map to green.
        image_b : `~numpy.ndarray`
            Image to map to blue.
        Returns
        -------
        RGBimage : `~numpy.ndarray`
            RGB (integer, 8-bits per channel) color image as an NxNx3
            numpy array.
        """
        image_r = np.asarray(image_r)
        image_g = np.asarray(image_g)
        image_b = np.asarray(image_b)

        if (image_r.shape != image_g.shape) or (image_g.shape != image_b.shape):
            msg = "The image shapes must match. r: {}, g: {} b: {}"
            raise ValueError(msg.format(image_r.shape, image_g.shape, image_b.shape))

        return np.dstack(self._convert_images_to_uint8(image_r, image_g, image_b)).astype(np.uint8)

    def intensity(self, image_r, image_g, image_b):
        """
        Return the total intensity from the red, blue, and green intensities.
        This is a naive computation, and may be overridden by subclasses.
        Parameters
        ----------
        image_r : `~numpy.ndarray`
            Intensity of image to be mapped to red; or total intensity if
            ``image_g`` and ``image_b`` are None.
        image_g : `~numpy.ndarray`, optional
            Intensity of image to be mapped to green.
        image_b : `~numpy.ndarray`, optional
            Intensity of image to be mapped to blue.
        Returns
        -------
        intensity : `~numpy.ndarray`
            Total intensity from the red, blue and green intensities, or
            ``image_r`` if green and blue images are not provided.
        """
        return compute_intensity(image_r, image_g, image_b)

    def map_intensity_to_uint8(self, I):
        """
        Return an array which, when multiplied by an image, returns that image
        mapped to the range of a uint8, [0, 255] (but not converted to uint8).
        The intensity is assumed to have had minimum subtracted (as that can be
        done per-band).
        Parameters
        ----------
        I : `~numpy.ndarray`
            Intensity to be mapped.
        Returns
        -------
        mapped_I : `~numpy.ndarray`
            ``I`` mapped to uint8
        """
        with np.errstate(invalid='ignore', divide='ignore'):
            return np.clip(I, 0, self._uint8Max)

    def _convert_images_to_uint8(self, image_r, image_g, image_b):
        """Use the mapping to convert images image_r, image_g, and image_b to a triplet of uint8 images"""
        image_r = image_r - self.minimum[0]  # n.b. makes copy
        image_g = image_g - self.minimum[1]
        image_b = image_b - self.minimum[2]

        fac = self.map_intensity_to_uint8(self.intensity(image_r, image_g, image_b))

        image_rgb = [image_r, image_g, image_b]
        for c in image_rgb:
            c *= fac
            with np.errstate(invalid='ignore'):
                c[c < 0] = 0                # individual bands can still be < 0, even if fac isn't

        pixmax = self._uint8Max
        r0, g0, b0 = image_rgb           # copies -- could work row by row to minimise memory usage

        with np.errstate(invalid='ignore', divide='ignore'):  # n.b. np.where can't and doesn't short-circuit
            for i, c in enumerate(image_rgb):
                c = np.where(r0 > g0,
                             np.where(r0 > b0,
                                      np.where(r0 >= pixmax, c*pixmax/r0, c),
                                      np.where(b0 >= pixmax, c*pixmax/b0, c)),
                             np.where(g0 > b0,
                                      np.where(g0 >= pixmax, c*pixmax/g0, c),
                                      np.where(b0 >= pixmax, c*pixmax/b0, c))).astype(np.uint8)
                c[c > pixmax] = pixmax

                image_rgb[i] = c

        return image_rgb

    def __call__(self, images):
        return self.make_rgb_image(*images)


class LinearMapping(Mapping):
    """
    A linear map map of red, blue, green intensities into uint8 values.
    A linear stretch from [minimum, maximum].
    If one or both are omitted use image min and/or max to set them.
    Parameters
    ----------
    minimum : float
        Intensity that should be mapped to black
        (a scalar or array for R, G, B).
    maximum : float
        Intensity that should be mapped to white (a scalar).
    """

    def __init__(self, minimum=None, maximum=None, image=None):
        if minimum is None or maximum is None:
            if image is None:
                raise ValueError("you must provide an image if you don't "
                                 "set both minimum and maximum")
            if minimum is None:
                minimum = image.min()
            if maximum is None:
                maximum = image.max()

        Mapping.__init__(self, minimum=minimum, image=image)
        self.maximum = maximum

        if maximum is None:
            self._range = None
        else:
            if maximum == minimum:
                raise ValueError("minimum and maximum values must not be equal")
            self._range = float(maximum - minimum)

    def map_intensity_to_uint8(self, I):
        with np.errstate(invalid='ignore', divide='ignore'):  # n.b. np.where can't and doesn't short-circuit
            return np.where(I <= 0, 0,
                            np.where(I >= self._range, self._uint8Max/I, self._uint8Max/self._range))


class AsinhMapping(Mapping):
    """ A color preserving mapping for an asinh stretch

    x = asinh(Q (I - minimum)/stretch)/Q
    This reduces to a linear stretch if Q == 0
    See http://adsabs.harvard.edu/abs/2004PASP..116..133L
    Parameters
    ----------
    minimum : float
        Intensity that should be mapped to black
        (a scalar or array for R, G, B).
    stretch : float
        The linear stretch of the image.
    Q : float
        The asinh softening parameter.
    """

    def __init__(self, minimum, stretch, Q=8):
        Mapping.__init__(self, minimum)

        epsilon = 1.0/2**23   # 32bit floating point machine epsilon; sys.float_info.epsilon is 64bit
        if abs(Q) < epsilon:
            Q = 0.1
        else:
            Qmax = 1e10
            if Q > Qmax:
                Q = Qmax

        frac = 0.1                  # gradient estimated using frac*stretch is _slope
        self._slope = frac*self._uint8Max/np.arcsinh(frac*Q)

        self._soften = Q/float(stretch)

    def map_intensity_to_uint8(self, I):
        with np.errstate(invalid='ignore', divide='ignore'):  # n.b. np.where can't and doesn't short-circuit
            return np.where(I <= 0, 0, np.arcsinh(I*self._soften)*self._slope/I)


class LinearPercentileNorm(LinearMapping):
    def __init__(self, img, percentiles=[1, 99]):
        """Create norm that is linear between lower and upper percentile of img
        Parameters
        ----------
        img: array_like
            Image to normalize
        percentile: array_like, default=[1,99]
            Lower and upper percentile to consider. Pixel values below will be
            set to zero, above to saturated.
        """
        assert len(percentiles) == 2
        vmin, vmax = np.percentile(img, percentiles)
        super().__init__(minimum=vmin, maximum=vmax)


class AsinhPercentileNorm(AsinhMapping):
    def __init__(self, img, percentiles=[1, 99]):
        """Create norm that is linear between lower and upper percentile of img
        Parameters
        ----------
        img: array_like
            Image to normalize
        percentile: array_like, default=[1,99]
            Lower and upper percentile to consider. Pixel values below will be
            set to zero, above to saturated.
        """
        assert len(percentiles) == 2
        vmin, vmax = np.percentile(img, percentiles)
        # solution for beta assumes flat spectrum at vmax
        stretch = vmax - vmin
        beta = stretch / np.sinh(1)
        super().__init__(minimum=vmin, stretch=stretch, Q=beta)


def img_to_channel(img, filter_weights=None, fill_value=0):
    """Convert multi-band image cube into 3 RGB channels

    Parameters
    ----------
    img: array_like
        This should be an array with dimensions (bands, height, width).
    filter_weights: array_like
        Linear mapping with dimensions (channels, bands)
    fill_value: float, default=`0`
        Value to use for any masked pixels.

    Returns
    -------
    RGB: numpy array with dtype float
    """
    # expand single img into cube
    assert len(img.shape) in [2, 3]
    if len(img.shape) == 2:
        ny, nx = img.shape
        img_ = img.reshape(1, ny, nx)
    elif len(img.shape) == 3:
        img_ = img
    B = len(img_)
    C = 3  # RGB

    # filterWeights: channel x band
    if filter_weights is None:
        filter_weights = np.zeros((C, B))
        if B == 1:
            filter_weights[0, 0] = filter_weights[1, 0] = filter_weights[2, 0] = 1
        if B == 2:
            filter_weights[0, 1] = 0.667
            filter_weights[1, 1] = 0.333
            filter_weights[1, 0] = 0.333
            filter_weights[2, 0] = 0.667
            filter_weights /= 0.667
        if B == 3:
            filter_weights[0, 2] = 1
            filter_weights[1, 1] = 1
            filter_weights[2, 0] = 1
        if B == 4:
            filter_weights[0, 3] = 1
            filter_weights[0, 2] = 0.333
            filter_weights[1, 2] = 0.667
            filter_weights[1, 1] = 0.667
            filter_weights[2, 1] = 0.333
            filter_weights[2, 0] = 1
            filter_weights /= 1.333
        if B == 5:
            filter_weights[0, 4] = 1
            filter_weights[0, 3] = 0.667
            filter_weights[1, 3] = 0.333
            filter_weights[1, 2] = 1
            filter_weights[1, 1] = 0.333
            filter_weights[2, 1] = 0.667
            filter_weights[2, 0] = 1
            filter_weights /= 1.667
        if B == 6:
            filter_weights[0, 5] = 1
            filter_weights[0, 4] = 0.667
            filter_weights[0, 3] = 0.333
            filter_weights[1, 4] = 0.333
            filter_weights[1, 3] = 0.667
            filter_weights[1, 2] = 0.667
            filter_weights[1, 1] = 0.333
            filter_weights[2, 2] = 0.333
            filter_weights[2, 1] = 0.667
            filter_weights[2, 0] = 1
            filter_weights /= 2
    else:
        assert filter_weights.shape == (3, len(img))

    # map bands onto RGB channels
    _, ny, nx = img_.shape
    rgb = np.dot(filter_weights, img_.reshape(B, -1)).reshape(3, ny, nx)

    if hasattr(rgb, "mask"):
        rgb = rgb.filled(fill_value)

    return rgb


def img_to_rgb(img, filter_weights=None, fill_value=0, norm=None):
    """Convert images to normalized RGB.

    If normalized values are outside of the range [0..255], they will be
    truncated such as to preserve the corresponding color.

    Parameters
    ----------
    img: array_like
        This should be an array with dimensions (bands, height, width).
    filter_weights: array_like
        Linear mapping with dimensions (channels, bands)
    fill_value: float, default=`0`
        Value to use for any masked pixels.
    norm: `scarlet.display.Norm`, default `None`
        Norm to use for mapping in the allowed range [0..255]. If `norm=None`,
        `scarlet.display.LinearPercentileNorm` will be used.

    Returns
    -------
    rgb: numpy array with dimensions (3, height, width) and dtype uint8
    """
    RGB = img_to_channel(img)
    if norm is None:
        norm = LinearMapping(image=RGB)
    rgb = norm(RGB)
    return rgb

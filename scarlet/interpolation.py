import numpy as np
from .cache import Cache
from . import fft


def get_filter_coords(filter_values, center=None):
    """Create filter coordinate grid needed for the apply filter function

    Parameters
    ----------
    filter_values: array
        The 2D array of the filter to apply.
    center: tuple
        The center (y,x) of the filter. If `center` is `None` then
        `filter_values` must have an odd number of rows and columns
        and the center will be set to the center of `filter_values`.

    Returns
    -------
    coords: array
        The coordinates of the pixels in `filter_values`,
        where the coordinates of the `center` pixel are `(0,0)`.
    """
    if len(filter_values.shape) != 2:
        raise ValueError("`filter_values` must be 2D")
    if center is None:
        if filter_values.shape[0] % 2 == 0 or filter_values.shape[1] % 2 == 0:
            msg = """Ambiguous center of the `filter_values` array,
                     you must use a `filter_values` array
                     with an odd number of rows and columns or
                     calculate `coords` on your own."""
            raise ValueError(msg)
        center = [filter_values.shape[0]//2, filter_values.shape[1]//2]
    x = np.arange(filter_values.shape[1])
    y = np.arange(filter_values.shape[0])
    x, y = np.meshgrid(x, y)
    x -= center[1]
    y -= center[0]
    coords = np.dstack([y, x])
    return coords


def get_filter_bounds(coords):
    """Get the slices in x and y to apply a filter

    Parameters
    ----------
    coords: array
        The coordinates of the filter,
        defined by `get_filter_coords`.

    Returns
    -------
    y_start, y_end, x_start, x_end: int
        The start and end of each slice that is passed to `apply_filter`.
    """
    z = np.zeros((len(coords),), dtype=int)
    # Set the y slices
    y_start = np.max([z, coords[:, 0]], axis=0)
    y_end = -np.min([z, coords[:, 0]], axis=0)
    # Set the x slices
    x_start = np.max([z, coords[:, 1]], axis=0)
    x_end = -np.min([z, coords[:, 1]], axis=0)
    return y_start, y_end, x_start, x_end


def get_projection_slices(image, shape, yx0=None):
    """Get slices needed to project an image

    This method returns the bounding boxes needed to
    project `image` into a larger image with `shape`.
    The results can be used with
    `projection[bb] = image[ibb]`.

    Parameters
    ----------
    image: array
        2D input image
    shape: tuple
        Shape of the new image.
    yx0: tuple
        Location of the lower left corner of the image in
        the projection.
        If `yx0` is `None` then the image is centered in
        the projection.

    Returns
    -------
    bb: tuple
        `(yslice, xslice)` of the projected image to place `image`.
    ibb: tuple
        `(iyslice, ixslice)` of `image` to insert into the projection.
    bounds: tuple
        `(bottom, top, left, right)` locations of the corners of `image`
        in the projection. While this isn't needed for slicing it can be
        useful for calculating information about the image before projection.
    """
    Ny, Nx = shape
    iNy, iNx = image.shape
    if yx0 is None:
        y0 = iNy // 2
        x0 = iNx // 2
        yx0 = (-y0, -x0)
    bottom, left = yx0
    bottom += Ny >> 1
    left += Nx >> 1

    top = bottom + iNy
    yslice = slice(max(0, bottom), min(Ny, top))
    iyslice = slice(max(0, -bottom), max(Ny - bottom, -top))

    right = left + iNx
    xslice = slice(max(0, left), min(Nx, right))
    ixslice = slice(max(0, -left), max(Nx - left, -right))
    return (yslice, xslice), (iyslice, ixslice), (bottom, top, left, right)


def project_image(image, shape, yx0=None):
    """Project an image centered in a larger image

    The projection pads the image with zeros if
    necessary or trims the edges if img is larger
    than shape in a given dimension.

    Parameters
    ----------
    image: array
        2D input image
    shape: tuple
        Shape of the new image.
    yx0: tuple
        Location of the lower left corner of the image in
        the projection.
        If `yx0` is `None` then the image is centered in
        the projection.

    Returns
    -------
    result: array
        The result of projecting `image`.
    """
    result = np.zeros(shape)
    bb, ibb, _ = get_projection_slices(image, shape, yx0)
    result[bb] = image[ibb]
    return result


def common_projections(img1, img2):
    """Project two images to a common frame

    It is assumed that the two images have the same center.
    This is mainly used for FFT convolutions of source components,
    where the convolution kernel is a different size than the morphology.

    Parameters
    ----------
    img1: array
        1st 2D image to project
    img2: array
        2nd 2D image to project

    Returns
    -------
    img1: array
        Projection of 1st image
    img2: array
        Projection of 2nd image
    """
    h1, w1 = img1.shape
    h2, w2 = img2.shape
    shape = (max(h1, h2), max(w1, w2))
    return project_image(img1, shape), project_image(img2, shape)


def fft_convolve(*images):
    """Use FFT's to convove an image with a kernel

    Parameters
    ----------
    images: list of array-like
        A list of images to convolve.

    Returns
    -------
    result: array
        The convolution in pixel space of `img` with `kernel`.
    """
    from autograd.numpy.numpy_boxes import ArrayBox

    Images = [np.fft.fft2(np.fft.ifftshift(img)) for img in images]
    if np.any([isinstance(img, ArrayBox) for img in images]):
        Convolved = Images[0]
        for img in Images[1:]:
            Convolved = Convolved * img
    else:
        Convolved = np.prod(Images, 0)
    convolved = np.fft.ifft2(Convolved)
    return np.fft.fftshift(np.real(convolved))


def bilinear(dx):
    """Bilinear interpolation kernel

    Interpolate between neighboring pixels to shift
    by a fractional amount.

    Parameters
    ----------
    dx: float
        Fractional amount that the kernel will be shifted

    Returns
    -------
    result: array
        2x2 linear kernel to use nearest neighbor interpolation
    window: array
        The pixel values for the window containing the kernel
    """
    if np.abs(dx) > 1:
        raise ValueError("The fractional shift dx must be between -1 and 1")
    if dx >= 0:
        window = np.arange(2)
        y = np.array([1 - dx, dx])
    else:
        window = np.array([-1, 0])
        y = np.array([-dx, 1 + dx])
    return y, window


def cubic_spline(dx, a=1, b=0):
    """Generate a cubix spline centered on `dx`.

    Parameters
    ----------
    dx: float
        Fractional amount that the kernel will be shifted
    a: float
        Cubic spline sharpness paremeter
    b: float
        Cubic spline shape parameter

    Returns
    -------
    result: array
        Cubic Spline kernel in a window from floor(dx)-1 to floor(dx) + 3
    window: array
        The pixel values for the window containing the kernel
    """
    if np.abs(dx) > 1:
        raise ValueError("The fractional shift dx must be between -1 and 1")

    def inner(x):
        """Cubic from 0<=abs(x)<=1
        """
        third = (-6 * a - 9 * b + 12) * x ** 3
        second = (6 * a + 12 * b - 18) * x ** 2
        zero = -2 * b + 6
        return (zero + second + third) / 6

    def outer(x):
        """Cubic from 1<=abs(x)<=2
        """
        third = (-6 * a - b) * x ** 3
        second = (30 * a + 6 * b) * x ** 2
        first = (-48 * a - 12 * b) * x
        zero = 24 * a + 8 * b
        return (zero + first + second + third) / 6

    window = np.arange(-1, 3) + np.floor(dx)
    x = np.abs(dx - window)
    result = np.piecewise(
        x, [x <= 1, (x > 1) & (x < 2)], [lambda x: inner(x), lambda x: outer(x)]
    )

    return result, np.array(window).astype(int)


def catmull_rom(dx):
    """Cubic spline with a=0.5, b=0

    See `cubic_spline` for details.
    """
    return cubic_spline(dx, a=0.5, b=0)


def mitchel_netravali(dx):
    """Cubic spline with a=1/3, b=1/3

    See `cubic_spline` for details.
    """
    ab = 1 / 3
    return cubic_spline(dx, a=ab, b=ab)


def lanczos(dx, a=3):
    """Lanczos kernel

    Parameters
    ----------
    dx: float
        amount to shift image
    a: int
        Lanczos window size parameter

    Returns
    -------
    result: array-like
        1D Lanczos kernel
    """
    if np.abs(dx) > 1:
        raise ValueError("The fractional shift dx must be between -1 and 1")
    window = np.arange(-a + 1, a + 1) + np.floor(dx)
    y = np.sinc(dx - window) * np.sinc((dx - window) / a)
    return y, window.astype(int)


def quintic_spline(dx, dtype=np.float64):
    def inner(x):
        return 1 + x ** 3 / 12 * (-95 + 138 * x - 55 * x ** 2)

    def middle(x):
        return (x - 1) * (x - 2) / 24 * (-138 + 348 * x - 249 * x ** 2 + 55 * x ** 3)

    def outer(x):
        return (x - 2) * (x - 3) ** 2 / 24 * (-54 + 50 * x - 11 * x ** 2)

    window = np.arange(-3, 4)
    x = np.abs(dx - window)
    result = np.piecewise(
        x,
        [x <= 1, (x > 1) & (x <= 2), (x > 2) & (x <= 3)],
        [lambda x: inner(x), lambda x: middle(x), lambda x: outer(x)],
    )
    return result, window


def get_separable_kernel(dy, dx, kernel=lanczos, **kwargs):
    """Create a 2D kernel from a 1D kernel separable in x and y

    Parameters
    ----------
    dy: float
        amount to shift image in x
    dx: float
        amount to shift image in y
    kernel: function
        1D kernel that is separable in x and y
    kwargs: dict
        Keyword arguments for the kernel

    Returns
    -------
    kernel: Tensor
        2D separable kernel
    x_window: Tensor
        The pixel values for the window containing the kernel in the x-direction
    y_window: Tensor
        The pixel values for the window containing the kernel in the y-direction
    """
    kx, x_window = kernel(dx, **kwargs)
    ky, y_window = kernel(dy, **kwargs)
    kyx = np.outer(ky, kx)
    return kyx, y_window, x_window


def mk_shifter(shape, real=False):
    """ Performs shifts in the Fourier domain on Fourier objects

    Parameters:
    -----------
    shape: array
        shape of the 2-D array to shift
    real: bool
        if true, the frequencies are all returned for real transforms (all dimension are half of the shape).
        if False, only the last dimension is considered a real transform.
    Returns:
    --------
    result: Fourier
        A Fourier object with shifted arrays
    """

    # Name of the chached shifts.
    name = "mk_shifter"
    key = shape[0], shape[1], real
    try:
        shifters = Cache.check(name, key)
    except KeyError:
        freq_x = np.fft.rfftfreq(shape[-1])
        if real is True:
            freq_y = np.fft.rfftfreq(shape[-2])
        else:
            freq_y = np.fft.fftfreq(shape[-2])
        # Shift the signal to recenter it, negative because math is opposite from
        # pixel direction
        shift_y = (-1j * 2 * np.pi * freq_y)
        shift_x = (-1j * 2 * np.pi * freq_x)

        shifters = (shift_y, shift_x)
    Cache.set(name, key, shifters)
    return shifters

def get_affine(wcs):
    try:
        model_affine = wcs.wcs.pc
    except AttributeError:
        model_affine = wcs.cd

    return model_affine

def get_pixel_size(model_affine):
    """ Extracts the pixel size from a wcs
    """
    pix = np.sqrt(
        np.abs(model_affine[0, 0])
        * np.abs(model_affine[1, 1] - model_affine[0, 1] * model_affine[1, 0]))
    return pix

def get_angles(frame_wcs, model_wcs):

    """ Computes the angles between two WCS
    Parameters
    ----------
        frame_wcs: WCS
            WCS of the observation's frame
        model_WCS:
            WCS of the model frame.
    """
    model_affine = get_affine(model_wcs)
    frame_affine = get_affine(frame_wcs)
    model_pix = get_pixel_size(model_affine)
    frame_pix = get_pixel_size(frame_affine)
    # Pixel scale ratio
    h = frame_pix / model_pix
    # Vector giving the direction of the x-axis of each frame
    self_framevector = np.sum(frame_affine, axis=0)[:2] / frame_pix
    model_framevector = np.sum(model_affine, axis=0)[:2] / model_pix
    # normalisation
    self_framevector /= np.sum(self_framevector ** 2) ** 0.5
    model_framevector /= np.sum(model_framevector ** 2) ** 0.5

    # sin of the angle between datasets (normalised cross product)
    sin_rot = np.cross(self_framevector, model_framevector)
    # cos of the angle. (normalised scalar product)
    cos_rot = np.dot(self_framevector, model_framevector)
    return [cos_rot, sin_rot], h

def sinc_interp(images, coord_hr, coord_lr, angle=None, padding=3):
    """
    Parameters
    ----------
    image: array
        image whose pixels are at positions coord_lr
    coord_hr: array (2xN)
        Coordinates of the high resolution grid
    coord_lr: array (2xM)
        Coordinates of the low resolution grid
    angle: float
        rotation angle between coordinate sets coord_hr and coord_lr
    padding: int
        value of zero padding for fft
    Returns
    -------
        result:  interpolated  samples at positions coord_hr
    """
    y_hr, x_hr = coord_hr
    y_lr, x_lr = coord_lr
    hy = np.abs(y_lr[1] - y_lr[0])
    hx = np.abs(x_lr[1] - x_lr[0])

    assert hy != 0
    assert hx != 0

    if (angle is None) or (1 - angle[0] < np.finfo(float).eps):
        result = [
            np.dot(
                np.dot(
                    np.sinc((y_lr[np.newaxis, :] - y_hr[:, np.newaxis]) / hy), image.T
                ),
                np.sinc((x_lr[:, np.newaxis] - x_hr[np.newaxis, :]) / hx),
            )
            for image in images
        ]
        return np.array(result)

    cos = angle[0]
    sin = angle[1]

    fft_shape = fft._get_fft_shape(images, images, padding=padding, axes=[1, 2])

    X = fft.Fourier(images)
    # Fourier transform
    X_fft = X.fft(fft_shape, (-2, -1))

    # Shift elementary kernel
    shifter_y, shifter_x = mk_shifter(fft_shape)

    #Shifts values
    shift_y = np.exp(shifter_y[np.newaxis, :] * (- (y_hr[:, np.newaxis]) * cos))
    shift_x = np.exp(shifter_x[np.newaxis, :] * (- (y_hr[:, np.newaxis]) * sin))
    #Apply shifts

    result_fft = X_fft[:, np.newaxis, :, :] * shift_y[np.newaxis, :, :, np.newaxis]
    result_fft = result_fft * shift_x[np.newaxis, :, np.newaxis, :]

    # Shape of the expected array
    result_shape = np.array(
        [result_fft.shape[0], result_fft.shape[1], X.image.shape[1], X.image.shape[2]]
    )
    # Shifts applied in one direction
    result_shift = fft.Fourier.from_fft(result_fft, fft_shape, result_shape, [2, 3])
    # sinc kernels
    shy = np.sinc((y_lr[np.newaxis, :] + x_hr[:, np.newaxis] * sin) / hy)
    shx = np.sinc((x_lr[np.newaxis, :] - x_hr[:, np.newaxis] * cos) / hx)

    # Sinc kernels in both direction
    result_y = (
        result_shift.image[:, :, np.newaxis, :, :]
        * shy[np.newaxis, np.newaxis, :, :, np.newaxis]
    ).sum(axis=-2)
    result = (result_y * shx[np.newaxis, np.newaxis, :, :]).sum(axis=-1)

    return result


def sinc_interp_inplace(image, h_image, h_target, angle, pad_shape = None):
    """ In place interpolation of a cube of images

    Performs interpolation from a grid defined by the grid of `image` to a grid spanning the same physical area scaled
    by a factor `h` and rotated by `angle` radians. The center for the rotation is the central pixel of the image.
    This procedure is advised for odd-sized images only.

    Parameters
    ----------
    image: `ndarray`
        Cube of images with shape BxNyxNx with B the number of bands and NyxNx, the number of pixels
    h_image: `float`
        Phisical scale of a pixel in image
    h_target: `float`
        Physical scale of the target pixel to which to interpolate
    angle: float
        angle between the grid of image and the target grid where to interpolate
    Returns
    -------
    interp_image: `ndarray`
        padded interpolated image
    """
    assert len(image.shape) == 3, "images should be provided as a cube. If only one image is provided, " \
                                  "image should be a cube with image.shape[0] = 1"
    if pad_shape is not None:
        # Padding. This is never explicitelly undone in this function on purpose. Proceed with caution.
        image = fft._pad(image, pad_shape, axes = [-2,-1])

    ny_lr, nx_lr = image.shape[-2:]
    coord_lr = np.array([np.array(range(ny_lr)) - (ny_lr-1)/2, np.array(range(nx_lr))-(nx_lr-1)/2])
    ny_hr, nx_hr = (image.shape[-2] * h_image / h_target).astype(int), \
                   (image.shape[-1] * h_image / h_target).astype(int)
    if (ny_hr % 2) == 0:
        ny_hr += 1
    if (nx_hr % 2) == 0:
        nx_hr += 1
    coord_hr = np.array([np.array(range(ny_hr.astype(int)))-(ny_hr-1)/2,
                         np.array(range(nx_hr.astype(int)))-(nx_hr-1)/2]) / h_image * h_target
    return sinc_interp(image, coord_hr, coord_lr, angle=angle)


def fft_resample(img, dy, dx, kernel=lanczos, **kwargs):
    """Translate the image by a fraction of a pixel

    This method uses FFT's to convolve the image with a
    kernel to shift the position of the image by a fraction
    of a pixel.

    Parameters
    ----------
    img: array-like
        Image
    dx: float
        Fractional amount to shift the convolution kernel in x.
    dy: float
        Fractional amount to shift the convolution kernel in y.
    kernel: function
        Kernel to use for the convolution.
    kwargs: dict
        Keyword arguments to build the kernel.

    Returns
    -------
    result: Tensor
        The convolved image.
    """
    # Build the kernel
    kernel, ywin, xwin = get_separable_kernel(dy, dx, kernel=kernel, **kwargs)
    # We have to project the image and kernels to the same
    # shape to multiply them in Fourier space
    hx, wx = img.shape
    hk, wk = kernel.shape
    # We have to pad the input image by the width of the kernel,
    # because the Fourier Transform is periodic and will wrap the solution
    shape = (hx + hk + 3, wx + wk + 3)

    # Project the kernel onto the same space as the output image
    yx0 = (ywin[0], xwin[0])
    _kernel = project_image(kernel, shape, yx0)
    _img = project_image(img, shape)
    result = fft_convolve(_img, _kernel)
    return project_image(result, img.shape)


def get_common_padding(img1, img2, padding=None):
    """Project two images to a common frame

    It is assumed that the two images have the same center.
    This is mainly used for FFT convolutions of source components,
    where the convolution kernel is a different size than the morphology.

    Parameters
    ----------
    img1: array
        1st 2D or 3D image to project
    img2: array
        2nd 2D or 3D image to project

    Returns
    -------
    img1: array
        Projection of 1st image
    img2: array
        Projection of 2nd image
    """
    h1, w1 = img1.shape[-2:]
    h2, w2 = img2.shape[-2:]
    height = h1 + h2
    width = w1 + w2
    if padding is not None:
        height += padding
        width += padding

    def get_padding(h, w):
        bottom = (height - h) // 2
        top = height - h - bottom
        left = (width - w) // 2
        right = width - w - left
        return ((bottom, top), (left, right))

    return get_padding(h1, w1), get_padding(h2, w2)


def sinc2D(y, x):
    """
    2-D sinc function based on the product of 2 1-D sincs

    Parameters
    ----------
        x, y: arrays
            Coordinates where to evaluate the 2-D sinc
    Returns
    -------
    result: array
        2-D sinc evaluated in x and y
    """
    return np.dot(np.sinc(y), np.sinc(x))


def subsample_function(y, x, f, dNy, dNx=None, dy=None, dx=None):
    """Subsample a function
    Given the expected pixel grid of a function, subsample that function
    at a grid subdivided in x by `dNx` and y by `dNy`.
    """
    # Use the spacing between x values to define the subsampled regions
    if dx is None:
        dx = x[1] - x[0]
    if dy is None:
        dy = y[1] - y[0]
    if dNx is None:
        dNx = dNy
    assert dNy % 2 == 0, "dNy must be even, received {0}".format(dNy)
    assert dNx % 2 == 0, "dNx must be even, received {0}".format(dNx)
    assert np.all(np.isclose(x[1:] - x[:-1], x[1] - x[0])), "x must have equal spacing"
    assert np.all(np.isclose(y[1:] - y[:-1], y[1] - y[0])), "y must have equal spacing"

    # Create the subsampled interval and use it to sample `f`
    _x = np.linspace(x[0] - dx / 2, x[-1] + dx / 2, len(x) * dNx + 1)
    _y = np.linspace(y[0] - dy / 2, y[-1] + dy / 2, len(y) * dNy + 1)
    return f(_y, _x), _y, _x


def apply_2D_trapezoid_rule(y, x, f, dNy, dNx=None, dy=None, dx=None):
    """Use the trapezoid rule to integrate over a subsampled function
    2D implementation of the trapezoid rule.
    See `apply_trapezoid_rule` for a description, with the difference
    that `f` is a function `f(y,x)`, where we note the c ++`(y,x)` ordering.
    """
    if dy is None:
        dy = y[1] - y[0]
    if dx is None:
        dx = x[1] - x[0]
    if dNx is None:
        dNx = dNy
    z, _y, _x = subsample_function(y, x, f, dNy, dNx, dy, dx)

    # Calculate the volume of each sub region
    dz = 0.4 * (z[:-1, :-1] + z[1:, :-1] + z[:-1, 1:] + z[1:, 1:])
    volumes = dy * dx * dz / dNy / dNx

    # Sum up the sub regions around each point to
    # give it the same shape as the original `(y,x)`
    _dNy = len(_y) // dNy
    _dNx = len(_x) // dNx
    volumes = np.array(
        np.split(np.array(np.split(volumes, _dNx, axis=1)), _dNy, axis=1)
    ).sum(axis=(2, 3))
    return volumes


def get_psf_size(psf):
    """ Measures the size of a psf by computing the size of the area in 3 sigma around the center.

    This is an approximate method to estimate the size of the psf for setting the size of the frame,
    which does not require a precise measurement.

    Parameters
    ----------
        PSF: `scarlet.PSF` object
            PSF for whic to compute the size
    Returns
    -------
        sigma3: `float`
            radius of the area inside 3 sigma around the center in pixels
    """
    # Normalisation by maximum
    psf_frame = psf/np.max(psf)

    # Pixels in the FWHM set to one, others to 0:
    psf_frame[psf_frame>0.5] = 1
    psf_frame[psf_frame<=0.5] = 0

    # Area in the FWHM:
    area = np.sum(psf_frame)

    # Diameter of this area
    d = 2*(area/np.pi)**0.5

    # 3-sigma:
    sigma3 = 3*d/(2*(2*np.log(2))**0.5)

    return sigma3
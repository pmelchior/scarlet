import numpy as np


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
    iyslice = slice(max(0, -bottom), max(Ny-bottom, -top))

    right = left + iNx
    xslice = slice(max(0, left), min(Nx, right))
    ixslice = slice(max(0, -left), max(Nx-left, -right))
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
    Images = [np.fft.fft2(np.fft.ifftshift(img)) for img in images]
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
        y = np.array([1-dx, dx])
    else:
        window = np.array([-1, 0])
        y = np.array([-dx, 1+dx])
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

    def inner(x, a, b):
        """Cubic from 0<=abs(x)<=1
        """
        third = (-6*a - 9*b + 12) * x**3
        second = (6*a + 12*b - 18) * x**2
        zero = -2*b + 6
        return (zero + second + third)/6

    def outer(x, a, b):
        """Cubic from 1<=abs(x)<=2
        """
        third = (-6*a-b) * x**3
        second = (30*a + 6*b) * x**2
        first = (-48*a-12*b) * x
        zero = 24*a + 8*b
        return (zero + first + second + third)/6

    window = np.arange(-1, 3) + np.floor(dx)
    result = np.zeros(window.shape)
    _x = np.abs(dx-window)
    outer_cut = (_x > 1) & (_x < 2)
    inner_cut = _x <= 1
    result[outer_cut] = outer(_x[outer_cut], a, b)
    result[inner_cut] = inner(_x[inner_cut], a, b)
    return result, np.array(window).astype(int)


def catmull_rom(dx):
    """Cubic spline with a=0.5, b=0

    See `cubic_spline` for details.
    """
    return cubic_spline(dx, a=.5, b=0)


def mitchel_netravali(dx):
    """Cubic spline with a=1/3, b=1/3

    See `cubic_spline` for details.
    """
    ab = 1/3
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
        return 1+x**3/12*(-95+138*x-55*x**2)

    def middle(x):
        return (x-1)*(x-2)/24*(-138+348*x-249*x**2+55*x**3)

    def outer(x):
        return (x-2)*(x-3)**2/24*(-54+50*x-11*x**2)
    window = np.arange(-3, 4)
    abs_x = np.abs(dx-window)
    inner_cut = abs_x <= 1
    middle_cut = (abs_x <= 2) & (abs_x > 1)
    outer_cut = (abs_x <= 3) & (abs_x > 2)

    result = np.zeros(7, dtype=dtype)
    result[inner_cut] = inner(abs_x[inner_cut])
    result[middle_cut] = middle(abs_x[middle_cut])
    result[outer_cut] = outer(abs_x[outer_cut])
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

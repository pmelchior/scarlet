import numpy as np


def project_image(img, shape, img_window=None):
    """Project an image centered in a larger image

    The projection pads the image with zeros if
    necessary or trims the edges if img is larger
    than shape in a given dimension.

    Parameters
    ----------
    img: array
        2D input image
    shape: tuple
        Shape of the new image.
    img_window: list of arrays
        Window that contains the image,
        where (0,0) is the center of the output image.
        If `img_window` is `None` then the input and
        output images are co-centered.

    Returns
    -------
    result: array
        The result of projecting `img`.
    """
    result = np.zeros(shape)
    Ny, Nx = shape
    iNy, iNx = img.shape
    dNy = Ny - iNy
    dNx = Nx - iNx
    if img_window is None:
        # We have to handle slices different depending on whether the
        # input image or target image is larger
        if dNy > 0:
            bottom = dNy >> 1
            yslice = slice(bottom, bottom - dNy)
            iyslice = slice(None)
        else:
            yslice = slice(None)
            bottom = -dNy >> 1
            iyslice = slice(bottom, bottom + Ny)
        if dNx > 0:
            left = dNx >> 1
            xslice = slice(left, left - dNx)
            ixslice = slice(None)
        else:
            xslice = slice(None)
            left = -dNx >> 1
            ixslice = slice(left, left + Nx)
    else:
        ywin = np.array(img_window[0])
        xwin = np.array(img_window[1])
        ywin += Ny >> 1
        xwin += Nx >> 1

        bottom = ywin[0]
        top = ywin[-1] + 1
        yslice = slice(max(0, bottom), min(Ny, top))
        iyslice = slice(max(0, -bottom), max(Ny-bottom, -top))

        left = xwin[0]
        right = xwin[-1] + 1
        xslice = slice(max(0, left), min(Nx, right))
        ixslice = slice(max(0, -left), max(Nx-left, -right))

    # Project the image
    bb = (yslice, xslice)
    ibb = (iyslice, ixslice)
    result[bb] = img[ibb]
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
    result: Tensor
        1D Lanczos kernel
    """
    # sinc is slow in pytorch because of the handling of zero,
    # which requires indexing, so we calculate the kernel
    # in numpy and then convert to pytorch
    window = np.arange(-a + 1, a + 1) + np.floor(dx)
    y = np.sinc(dx - window) * np.sinc((dx - window) / a)
    return y, window.astype(int)


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
    kxy = np.outer(ky, kx)
    return kxy, y_window, x_window


def fft_resample(img, dx, dy, kernel=lanczos, **kwargs):
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
    _kernel, xwin, ywin = get_separable_kernel(dx, dy, kernel=kernel, **kwargs)
    # Project the kernel onto the same space as the output image
    full_kernel = np.zeros(img.shape)
    cy = img.shape[0] // 2
    cx = img.shape[1] // 2
    y_slice = slice(int(ywin[0].item()) + cy, int(ywin[-1].item()) + cy + 1)
    x_slice = slice(int(xwin[0].item()) + cx, int(xwin[-1].item()) + cx + 1)
    full_kernel[y_slice, x_slice] = _kernel
    return fft_convolve(img, full_kernel)

import numpy as np
import torch
from .utils import complex_mul, ifftshift


def fft_convolve(img, kernel):
    """Use FFT's to convove an image with a kernel

    Parameters
    ----------
    img: Tensor
        The input image.
    kernel: Tensor
        The kernel to convolve the image with

    Returns
    -------
    result: Tensor
        The convolution in pixel space of `img` with `kernel`.
    """
    Img = torch.rfft(img, 2)
    Kernel = torch.rfft(kernel, 2)
    Convolved = complex_mul(Img, Kernel)
    convolved = torch.irfft(Convolved, 2)
    return ifftshift(convolved)


def bilinear_interpolation(dx):
    """Bilinear interpolation kernel

    Interpolate between neighboring pixels to shift
    by a fractional amount.

    Parameters
    ----------
    dx: float
        Fractional amount that the kernel will be shifted

    Returns
    -------
    result: Tensor
        2x2 linear kernel to use nearest neighbor interpolation
    window: array
        The pixel values for the window containing the kernel
    """
    if dx >= 0:
        window = torch.arange(2, dtype=torch.float32)
        y = torch.tensor([1-dx, dx], dtype=torch.float32)
    else:
        window = torch.tensor([-1, 0], dtype=torch.float32)
        y = torch.tensor([-dx, 1+dx])
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
    result: Tensor
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

    window = torch.arange(-1, 3, dtype=torch.float32) + np.floor(dx)
    result = torch.zeros(window.shape)
    _x = torch.abs(dx-window)
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
    return torch.tensor(y), window.astype(int)


def get_separable_kernel(dx, dy, kernel=lanczos, **kwargs):
    """Create a 2D kernel from a 1D kernel separable in x and y

    Parameters
    ----------
    dx: float
        amount to shift image in x
    dy: float
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
    kxy = torch.ger(ky, kx)
    return kxy, x_window, y_window


def fft_resample(img, dx, dy, kernel=lanczos, **kwargs):
    """Translate the image by a fraction of a pixel

    This method uses FFT's to convolve the image with a
    kernel to shift the position of the image by a fraction
    of a pixel.

    Parameters
    ----------
    img: Tensor
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
    full_kernel = torch.zeros(img.shape)
    cy = img.shape[0] // 2
    cx = img.shape[1] // 2
    y_slice = slice(int(ywin[0].item()) + cy, int(ywin[-1].item()) + cy + 1)
    x_slice = slice(int(xwin[0].item()) + cx, int(xwin[-1].item()) + cx + 1)
    full_kernel[y_slice, x_slice] = _kernel
    return fft_convolve(img, full_kernel)

import torch
from numpy.compat import integer_types


def complex_mul(t1, t2):
    """Multiply two complex tensors
    The pytorch implementation of complex numbers is still being added,
    see https://github.com/pytorch/pytorch/issues/755.
    In the meantime, pytorch adds an extra dimensions to tensors to
    make them complex but does not have any special operations for the
    matrices that are input to and output from torch.fft and torch.ifft.
    This function is used to multiply to complex tensors.
    Parameters
    ----------
    t1: Tensor with shape (N, M, 2)
        The first tensor.
    t2: Tensor with shae (N, M, 2)
        The second tensor.
    Returns
    -------
    result: Tensor with shape (N, M, 2)
        A complex tensor that represents the
        complex product of `t1` and `t2`.
    """
    t1_real = t1[..., 0]
    t1_imag = t1[..., 1]
    t2_real = t2[..., 0]
    t2_imag = t2[..., 1]
    real = t1_real * t2_real - t1_imag * t2_imag
    imag = t1_real * t2_imag + t1_imag * t2_real
    return torch.stack([real, imag], dim=-1)


def ifftshift(x, axes=None):
    """
    pytorch version of of numpy.fft.fftshift
    Like numpy, FFT's and inverse FFT's return a
    spectral/image array in "standard form," which is
    typically not the way we view an image.
    pytorch does not yet have a method to reshift the
    tensor from standard order to the more expected
    (and useful) arrangement, so we implement that method
    here. This is just a copy of `numpy.ifftshift` except
    that it creates a torch `Tensor` instead of a numpy `array`.
    Parameters
    ----------
    x : Tensor
        Input tensor.
    axes : int or shape tuple, optional
        Axes over which to calculate.
        Defaults to None, which shifts all axes.
    Returns
    -------
    y : Tensor
        The shifted tensor.
    """
    if axes is None:
        axes = tuple(range(len(x.shape)))
        shift = [-(dim // 2) for dim in x.shape]
    elif isinstance(axes, integer_types):
        shift = -(x.shape[axes] // 2)
    else:
        shift = [-(x.shape[ax] // 2) for ax in axes]

    return torch.roll(x, shift, axes)


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

    def get_padding(img, h, w):
        bottom = (height-h) // 2
        top = height-h-bottom
        left = (width-w) // 2
        right = width-w-left
        return (left, right, bottom, top)

    return get_padding(img1, h1, w1), get_padding(img2, h2, w2)


def fft_convolve(image, kernel, padding=3):
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
    if padding == 0:
        reshape = False
        _image = image
        _kernel = kernel
    else:
        reshape = True
        image_padding, kernel_padding = get_common_padding(image, kernel, padding=padding)
        left, right, bottom, top = image_padding
        _image = torch.nn.functional.pad(image, image_padding)
        _kernel = torch.nn.functional.pad(kernel, kernel_padding)

    Image = torch.rfft(_image, 2)
    Kernel = torch.rfft(_kernel, 2)
    Convolved = complex_mul(Image, Kernel)
    convolved = torch.irfft(Convolved, 2, signal_sizes=_image.shape)
    result = ifftshift(convolved)

    if reshape:
        result = result[bottom:-top, left:-right]
    return result

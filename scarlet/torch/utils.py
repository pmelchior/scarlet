from numpy.compat import integer_types
import torch


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

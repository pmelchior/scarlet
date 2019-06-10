import numpy as np
from astropy.visualization.lupton_rgb import LinearMapping, AsinhMapping


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
    rgb = norm.make_rgb_image(*RGB)
    return rgb

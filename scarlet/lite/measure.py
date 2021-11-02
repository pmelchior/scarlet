import numpy as np

from ..bbox import Box, overlapped_slices
from .utils import insert_image


def calculate_snr(images, variance, psfs, center):
    """Calculate the signal to noise for a source

    This is done by weighting the image with the PSF in each band
    and dividing by the PSF weighted variance.

    Parameters
    ----------
    images: `numpy.ndarray`
        The 3D (channels, y, x) image containing the data.
    variance: `numpy.array`
        The variance of `images`.
    psfs: `numpy.ndarray`
        The PSF in each channel.
    center: `list` of `int`
        The center of the signal.

    Returns
    -------
    snr: `numpy.ndarray`
        The signal to noise of the source.
    """
    py = psfs.shape[1] // 2
    px = psfs.shape[2] // 2
    bbox = Box(psfs.shape, origin=(0,-py+center[0], -px+center[1]))
    noise = bbox.extract_from(variance)
    img = bbox.extract_from(images)
    numerator = img * psfs
    denominator = (psfs * noise) * psfs
    return np.sum(numerator) / np.sqrt(np.sum(denominator))


def weight_sources(blend, mask_footprint=True):
    """Use the source models as templates to re-weight the data

    This is the standard "deblending" trick, where the models are
    only used as approximations to the data and are used to re-distribute
    the flux in the data according to the ratio of the models for each source.
    There is no return value for this function, instead it adds (or modifies)
    a `flux` attribute and `flux_box` attributes for all of the sources
    that contain their flux and the bounding box containing that flux
    repectively.

    Parameters
    ----------
    blend: `scarlet.lite.LiteBlend`
        The blend that is being fit
    mask_footprint: `bool`
        Whether or not to apply a mask for pixels with zero weight.

    Returns
    -------
    None
    """
    observation = blend.observation
    py = observation.psfs.shape[-2] // 2
    px = observation.psfs.shape[-1] // 2

    images = observation.images.copy()
    if mask_footprint:
        images = images * (observation.weights>0)
    model = blend.get_model()
    model = observation.convolve(model, mode="real")

    for src in blend.sources:
        if len(src.components) == 0:
            src.flux = 0
            continue
        _model = src.get_model()
        bbox = src.bbox.grow((0,py,px))
        _model = insert_image(bbox, src.bbox, _model)
        _model = observation.convolve(_model, mode="real")
        slices = overlapped_slices(observation.bbox, bbox)
        numerator = _model[slices[1]]
        denominator = model[slices[0]]
        ratio = numerator / denominator
        ratio[denominator == 0] = 0
        src.flux = ratio*images[slices[0]]
        src.flux_box = bbox

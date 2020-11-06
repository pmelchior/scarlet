import numpy as np
from . import initialization


def max_pixel(component):
    """Determine pixel with maximum value

    Parameters
    ----------
    component: `scarlet.Component` or `scarlet.ComponentTree`
        Component to analyze
    """
    model = component.get_model()
    return tuple(
        np.unravel_index(np.argmax(model), model.shape) + component.bbox.origin
    )


def flux(component):
    """Determine flux in every channel

    Parameters
    ----------
    component: `scarlet.Component` or `scarlet.ComponentTree`
        Component to analyze
    """
    model = component.get_model()
    return model.sum(axis=(1, 2))


def centroid(component):
    """Determine centroid of model

    Parameters
    ----------
    component: `scarlet.Component` or `scarlet.ComponentTree`
        Component to analyze
    """
    model = component.get_model()
    indices = np.indices(model.shape)
    centroid = np.array([np.sum(ind * model) for ind in indices]) / model.sum()
    return centroid + component.bbox.origin


def snr(component, observations):
    """Calculate SNR with morphology as weight function

    Parameters
    ----------
    morph: array or list thereof
        Morphology for each component in the source
    images: array
        images to get the spectrum amplitude from
    stds: array
        noise standard variation in every pixel of `images`

    Returns
    -------
    SNR
    """
    C = component.frame.C
    model = component.get_model().reshape(C, -1)
    # weights are given by normalized model
    W = model / model.sum(axis=1)[:, None]

    # compute SNR for this component
    detect_all, std_all = initialization.build_detection_image(observations)

    boxed_std = component.bbox.extract_from(std_all)
    var = (boxed_std ** 2).reshape(C, -1)

    # SNR from Erben (2001), eq. 16, extended to multiple bands
    # SNR = (I @ W) / sqrt(W @ Sigma^2 @ W)
    # with W = morph, Sigma^2 = diagonal variance matrix
    snr = (model * W).sum() / np.sqrt(((var * W) * W).sum())

    return snr

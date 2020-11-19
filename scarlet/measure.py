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


# adapted from https://github.com/pmelchior/shapelens/blob/master/src/Moments.cc
def moments(component, N=2, centroid=None, weight=None):

    if hasattr(component, "get_model"):
        model = component.get_model()
        if len(model.shape) == 3 and model.shape[0] == 1:
            model = model[0]
    else:
        model = component
    assert len(model.shape) == 2, "Moment measurement requires a 2D image"

    if weight is None:
        weight = 1
    else:
        assert model.shape == weight.shape

    grid_x, grid_y = np.indices(model.shape, dtype=np.float)

    if centroid is None:
        centroid = np.array(model.shape) // 2
    grid_y -= centroid[0]
    grid_x -= centroid[1]

    M = dict()
    for n in range(N + 1):
        for m in range(n + 1):
            # moments ordered by power in y, then x
            M[m, n - m] = (grid_y ** m * grid_x ** (n - m) * model * weight).sum()
    return M

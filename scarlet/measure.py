import numpy as np
from . import initialization
from .bbox import Box


def max_pixel(component):
    """Determine pixel with maximum value

    Parameters
    ----------
    component: `scarlet.Component` or array
        Component to analyze or its hyperspectral model
    """
    if hasattr(component, "get_model"):
        model = component.get_model()
        origin = component.bbox.origin
    else:
        model = component
        origin = 0

    return tuple(np.array(np.unravel_index(np.argmax(model), model.shape)) + origin)


def flux(component):
    """Determine flux in every channel

    Parameters
    ----------
    component: `scarlet.Component` or array
        Component to analyze or its hyperspectral model
    """
    if hasattr(component, "get_model"):
        model = component.get_model()
    else:
        model = component

    return model.sum(axis=(1, 2))


def centroid(component):
    """Determine centroid of model

    Parameters
    ----------
    component: `scarlet.Component` or array
        Component to analyze or its hyperspectral model
    """
    if hasattr(component, "get_model"):
        model = component.get_model()
        origin = component.bbox.origin
    else:
        model = component
        origin = 0

    indices = np.indices(model.shape)
    centroid = np.array([np.sum(ind * model) for ind in indices]) / model.sum()
    return centroid + origin


def snr(component, observations, prerender=True):
    """Determine SNR with morphology as weight function

    Parameters
    ----------
    component: `scarlet.Component` or array
        Component to analyze or its hyperspectral model

    observations: `scarlet.Observation` or list thereof
    """
    if not hasattr(observations, "__iter__"):
        observations = (observations,)

    if hasattr(component, "get_model"):
        frame = None
        if not prerender:
            frame = observations[0].model_frame
        model = component.get_model(frame=frame)
        bbox = component.bbox
    else:
        model = component
        bbox = Box(model.shape)

    if prerender:
        C = model.shape[0]
        M = model.reshape(C, -1)
        # weights are given by normalized model
        W = M / M.sum(axis=1)[:, None]

        # get variance of the deconvolved coadd
        detect_all, std_all = initialization.build_initialization_image(
            observations, prerender=prerender
        )
        boxed_std = bbox.extract_from(std_all)
        var = (boxed_std ** 2).reshape(C, -1)
    else:
        M = []
        W = []
        var = []
        # convolve model for every observation;
        # flatten in channel direction because it may not have all C channels; concatenate
        # do same thing for noise variance
        for obs in observations:
            model_ = obs.render(model)
            M.append(model_.reshape(-1))
            W.append((model_ / (model_.sum(axis=(-2, -1))[:, None, None])).reshape(-1))
            noise_var = 1 / np.where(obs.weights > 0, obs.weights, np.inf)
            var.append(noise_var.reshape(-1))
        M = np.concatenate(M)
        W = np.concatenate(W)
        var = np.concatenate(var)

    # SNR from Erben (2001), eq. 16, extended to multiple bands
    # SNR = (I @ W) / sqrt(W @ Sigma^2 @ W)
    # with W = morph, Sigma^2 = diagonal variance matrix
    snr = (M * W).sum() / np.sqrt(((var * W) * W).sum())

    return snr


# adapted from https://github.com/pmelchior/shapelens/blob/master/src/Moments.cc
def moments(component, N=2, centroid=None, weight=None):
    """Determine SNR with morphology as weight function

    Parameters
    ----------
    component: `scarlet.Component` or array
        Component to analyze or its hyperspectral model
    N: int >=0
        Moment order
    centroid: array
        2D coordinate in frame of `component`
    weight: array
        weight function with same shape as `component`
    """
    if hasattr(component, "get_model"):
        model = component.get_model()
    else:
        model = component

    if weight is None:
        weight = 1
    else:
        assert model.shape == weight.shape

    if centroid is None:
        centroid = np.array(model.shape) // 2

    grid_x, grid_y = np.indices(model.shape[-2:], dtype=np.float)
    if len(model.shape) == 3:
        grid_y = grid_y[None, :, :]
        grid_x = grid_x[None, :, :]
    grid_y -= centroid[0]
    grid_x -= centroid[1]

    M = dict()
    for n in range(N + 1):
        for m in range(n + 1):
            # moments ordered by power in y, then x
            M[m, n - m] = (grid_y ** m * grid_x ** (n - m) * model * weight).sum(
                axis=(-2, -1)
            )
    return M

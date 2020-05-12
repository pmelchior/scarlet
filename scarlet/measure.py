import numpy as np


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

import numpy as np
from .component import *

def get_model(component):
    frame_ = component.frame
    component.set_frame(component.bbox)
    model = component.get_model()
    component.set_frame(frame_)
    return model

def max_pixel(component):
    """Determine pixel with maximum value

    Parameters
    ----------
    component: `scarlet.Component` or `scarlet.ComponentTree`
        Component to analyze
    """
    model = get_model(component)
    return tuple(np.unravel_index(np.argmax(model_), model.shape) + component.bbox.origin)

def flux(component):
    """Determine flux in every channel

    Parameters
    ----------
    component: `scarlet.Component` or `scarlet.ComponentTree`
        Component to analyze
    """
    model = get_model(component)
    return model.sum(axis=(1,2))

def centroid(component):
    """Determine centroid of model

    Parameters
    ----------
    component: `scarlet.Component` or `scarlet.ComponentTree`
        Component to analyze
    """
    model = get_model(component)
    indices = np.indices(model.shape)
    centroid = np.array([np.sum(ind*model) for ind in indices]) / model.sum()
    return centroid + component.bbox.origin

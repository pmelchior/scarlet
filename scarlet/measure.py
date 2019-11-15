import numpy as np

def model_to_box(model, bbox=None):
    """Limit model to bbox

    Parameters
    ----------
    model: array
        The model of a `scarlet.Component` or `scarlet.ComponentTree`
    bbox: `scarlet.Box`
        Optional only search within bbox
    """
    if bbox is None:
        return model
    else:
        slices = bbox.slices_for(model)
        return model[slices]

def max_pixel(model, bbox=None):
    """Determine pixel with maximum value

    Parameters
    ----------
    model: array
        The model of a `scarlet.Component` or `scarlet.ComponentTree`
    bbox: `scarlet.Box`
        Optional only search within bbox
    """
    model_ = model_to_box(model, bbox=bbox)
    offset = np.array([0,0,0])
    if bbox is not None:
        offset = np.array([0,*bbox.yx0])
    return tuple(np.unravel_index(np.argmax(model_), model_.shape) + offset)

def flux(model, bbox=None):
    """Determine flux in every channel

    Parameters
    ----------
    model: array
        The model of a `scarlet.Component` or `scarlet.ComponentTree`
    bbox: `scarlet.Box`
        Optional only search within bbox
    """
    model_ = model_to_box(model, bbox=bbox)
    return model_.sum(axis=(1,2))

def centroid(model, bbox=None):
    """Determine centroid of model

    Parameters
    ----------
    model: array
        The model of a `scarlet.Component` or `scarlet.ComponentTree`
    bbox: `scarlet.Box`
        Optional only search within bbox
    """
    model_ = model_to_box(model, bbox=bbox)
    offset = np.array([0,0,0])
    if bbox is not None:
        offset = np.array([0,*bbox.yx0])

    # build the indices to use the the centroid calculation
    indices = np.indices(model_.shape)
    centroid = np.array([np.sum(ind*model_) for ind in indices]) / model_.sum()
    return centroid + offset

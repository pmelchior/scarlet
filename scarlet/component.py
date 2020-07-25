from abc import ABC, abstractmethod
from functools import partial
import autograd.numpy as np

from .frame import Frame
from .model import Model
from .parameter import Parameter
from .bbox import Box, overlapped_slices


class Component(Model):
    def __init__(self, frame, parameters=None, children=None, bbox=None, **kwargs):
        if bbox is None:
            bbox = frame.bbox

        # component should always be a data cube
        assert bbox.D == frame.bbox.D

        super().__init__(
            frame, parameters=parameters, children=children, bbox=bbox, **kwargs
        )
        self.set_model_frame(frame)

    def set_model_frame(self, frame):
        """Sets the frame for this component.

        Each component needs to know the properties of the Frame and,
        potentially, the subvolume it covers.

        Parameters
        ----------
        frame: `~scarlet.Frame`
            Frame of the model
        """
        self.model_frame_slices, self.model_slices = overlapped_slices(
            frame.bbox, self.bbox
        )

    def model_to_frame(self, frame=None, model=None):
        """Project a model into a frame


        Parameters
        ----------
        model: array
            Image of the model to project.
            This must be the same shape as `self.bbox`.
            If `model` is `None` then `self.get_model()` is used.
        frame: `~scarlet.frame.Frame`
            The frame to project the model into.
            If `frame` is `None` then the model is projected
            into `self.model_frame`.

        Returns
        -------
        projected_model: array
            (Channels, Height, Width) image of the model
        """
        # Use the current model by default
        if model is None:
            model = self.get_model()
        # Use the full model frame by default
        if frame is None or frame == self.frame:
            frame = self.frame
            frame_slices = self.model_frame_slices
            model_slices = self.model_slices
        else:
            frame_slices, model_slices = overlapped_slices(frame.bbox, self.bbox)

        if hasattr(frame, "dtype"):
            dtype = frame.dtype
        else:
            dtype = model.dtype
        result = np.zeros(frame.shape, dtype=dtype)
        result[frame_slices] = model[model_slices]
        return result


class FactorizedComponent(Component):
    """A single component in a blend.

    Uses the non-parametric factorization sed x morphology.

    Parameters
    ----------
    frame: `~scarlet.Frame`
        The spectral and spatial characteristics of the full model.
    bbox: `~scarlet.Box`
        Hyper-spectral bounding box of this component.
    spectrum: `~scarlet.Spectrum`
        Parameterization of the spectrum
    morphology: `~scarlet.Morphology`
        Parameterization of the morphology.
    """

    def __init__(self, frame, spectrum, morphology, **kwargs):
        from .spectrum import Spectrum
        from .morphology import Morphology

        assert isinstance(spectrum, Factor)
        assert isinstance(morphology, Factor)
        bbox = spectrum.bbox @ morphology.bbox
        super().__init__(frame, children=[spectrum, morphology], bbox=bbox, **kwargs)

    def get_model(self, *parameters, frame=None):
        """Get the model for this component.

        Parameters
        ----------
        parameters: tuple of optimimzation parameters

        frame: `~scarlet.frame.Frame`
            Frame to project the model into. If `frame` is `None`
            then the model contained in `bbox` is returned.

        Returns
        -------
        model: array
            (Channels, Height, Width) image of the model
        """
        spectrum, morphology = self.get_models_of_children(*parameters)
        model = spectrum[:, None, None] * morphology[None, :, :]
        # project the model into frame (if necessary)
        if frame is not None:
            model = self.model_to_frame(frame, model)
        return model


from autograd.extend import defvjp, primitive


@primitive
def _add_models(*models, full_model, slices):
    """Insert the models into the full model

    `slices` is a tuple `(full_model_slice, model_slices)` used
    to insert a model into the full_model in the region where the
    two models overlap.
    """
    for i in range(len(models)):
        if hasattr(models[i], "_value"):
            full_model[slices[i][0]] += models[i][slices[i][1]]._value
        else:
            full_model[slices[i][0]] += models[i][slices[i][1]]
    return full_model


def _grad_add_models(upstream_grad, *models, full_model, slices, index):
    """Gradient for a single model

    The full model is just the sum of the models,
    so the gradient is 1 for each model,
    we just have to slice it appropriately.
    """
    model = models[index]
    full_model_slices = slices[index][0]
    model_slices = slices[index][1]

    def result(upstream_grad):
        _result = np.zeros(model.shape, dtype=model.dtype)
        _result[model_slices] = upstream_grad[full_model_slices]
        return _result

    return result


class CombinedComponent(Component):
    def __init__(self, model_frame, components, mode="add"):
        if hasattr(components, "__iter__"):
            assert all(isinstance(c, Component) for c in components)
        else:
            assert isinstance(components, Component)
            components = (components,)

        assert mode in ["add"]  # , "multiply"]
        self.mode = mode

        super().__init__(model_frame, children=components, bbox=self.bbox)

    def __getitem__(self, i):
        return self.components.__getitem__(i)

    def __iter__(self):
        return self.components.__iter__()

    def __next__(self):
        return self.components.__next__()

    @property
    def bbox(self):
        """Union of all the component `~scarlet.bbox.Box`es
        """
        try:
            return self._bbox
        except AttributeError:
            # Make the bbox of the tree the union of the component bboxes
            box = self.components[0].bbox
            self._bbox = Box(box.shape, box.origin)
            for component in self.components:
                self._bbox |= component.bbox
        return self._bbox

    def get_model(self, *parameters, frame=None):

        model = np.zeros(self.bbox.shape, dtype=self.model_frame.dtype)
        i = 0
        for c in self.components:
            if len(parameters):
                j = len(c.parameters)
                p = parameters[i : i + j]
                i += j
                model_ = c.get_model(*p)
            else:
                model_ = c.get_model()

            model_slices, sub_slices = overlapped_slices(self.bbox, c.bbox)
            if self.mode == "add":
                model[model_slices] += model_[sub_slices]
            elif self.mode == "multiply":
                model[model_slices] *= model_[sub_slices]

        if frame is not None:
            model = self.model_to_frame(frame, model)
        return model


class CubeComponent(Component):
    """A single component in a blend.

    Uses full cube parameterization.

    Parameters
    ----------
    frame: `~scarlet.Frame`
        The spectral and spatial characteristics of this component.
    cube: `~scarlet.Parameter`
        3D array (C, Height, Width) of the initial data cube.
    bbox: `~scarlet.Box`
        Hyper-spectral bounding box of this component.
    """

    def __init__(self, frame, cube, bbox=None, **kwargs):
        if isinstance(image, Parameter):
            assert cube.name == "cube"
        else:
            constraint = PositivityConstraint()
            cube = Parameter(
                cube, name="cube", step=relative_step, constraint=constraint
            )
        super().__init__(frame, cube, bbox=bbox, **kwargs)

    def get_model(self, *parameters, frame=None):
        cube = self.get_parameter(0, *parameters)
        if frame is not None:
            cube = self.model_to_frame(frame, cube)
        return cube

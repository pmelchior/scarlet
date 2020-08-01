from abc import ABC, abstractmethod
from functools import partial
import autograd.numpy as np

from .frame import Frame
from .model import Model
from .parameter import Parameter
from .bbox import Box, overlapped_slices


class Component(Model):
    """A single component in a blend.

    This class acts as base for building models from parameters.

    Parameters
    ----------
    frame: `~scarlet.Frame`
        Characterization of the model
    parameters: list of `~scarlet.Parameter`
    children: list of `~scarlet.Model`
        Subordinate models.
    bbox: `~scarlet.Box`
        Bounding box of this model
    """

    def __init__(self, frame, *parameters, children=None, bbox=None):

        assert isinstance(frame, Frame)
        if bbox is None:
            bbox = frame.bbox
        assert isinstance(bbox, Box)
        self.set_frame(frame, bbox=bbox)

        super().__init__(*parameters, children=children)

    @property
    def bbox(self):
        """Hyper-spectral bounding box of this model
        """
        return self._bbox

    @property
    def frame(self):
        """Hyper-spectral characteristics is this model
        """
        return self._frame

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
            frame_slices = self._model_frame_slices
            model_slices = self._model_slices
        else:
            frame_slices, model_slices = overlapped_slices(frame.bbox, self.bbox)

        if hasattr(frame, "dtype"):
            dtype = frame.dtype
        else:
            dtype = model.dtype
        result = np.zeros(frame.shape, dtype=dtype)
        result[frame_slices] = model[model_slices]
        return result

    def set_frame(self, frame, bbox=None):
        """Sets the frame for this component.

        Each component needs to know the properties of the Frame and,
        potentially, the subvolume it covers.

        Parameters
        ----------
        model_frame: `~scarlet.Frame`
            Frame of the model
        """
        self._frame = frame
        if bbox is not None:
            self._bbox = bbox

        self._model_frame_slices, self._model_slices = overlapped_slices(
            frame.bbox, self._bbox
        )


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

    def __init__(self, frame, spectrum, morphology):
        from .spectrum import Spectrum

        assert isinstance(spectrum, Spectrum)

        from .morphology import Morphology

        assert isinstance(morphology, Morphology)

        bbox = spectrum.bbox @ morphology.bbox

        super().__init__(frame, children=[spectrum, morphology], bbox=bbox)

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

    def __init__(self, frame, cube, bbox=None):
        if isinstance(image, Parameter):
            assert cube.name == "cube"
        else:
            constraint = PositivityConstraint()
            cube = Parameter(
                cube, name="cube", step=relative_step, constraint=constraint
            )
        super().__init__(frame, cube, bbox=bbox)

    def get_model(self, *parameters, frame=None):
        model = self.get_parameter(0, *parameters)

        if frame is not None:
            model = self.model_to_frame(frame, model)
        return model


class CombinedComponent(Component):
    def __init__(self, components, operation=np.sum, check_boxes=True):

        assert len(components)
        frame = components[0].frame
        box = components[0].bbox
        # all children need to have the same bbox for simple autogradable combinations
        for c in components:
            assert isinstance(c, Component)
            assert c.frame is frame
            if check_boxes:
                assert c.bbox == box

        super().__init__(frame, children=components, bbox=box)

        self.op = operation

    def get_model(self, *parameters, frame=None):
        models = self.get_models_of_children(*parameters, frame=frame)
        model = self.op(models, axis=0)

        if frame is not None:
            model = self.model_to_frame(frame, model)
        return model

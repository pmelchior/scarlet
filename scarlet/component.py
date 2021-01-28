import autograd.numpy as np

from .frame import Frame
from .model import Model, UpdateException
from .parameter import Parameter, relative_step
from .constraint import PositivityConstraint
from .bbox import Box, overlapped_slices
from .fft import fast_zero_pad
from .morphology import Morphology
from .spectrum import Spectrum


class Component(Model):
    """Base class for hyperspectral models given parameters.

    The class allows for hierarchical ordering through `children`.

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
        self._bbox = bbox  # don't use th setter bc frame isn't set yet
        self.frame = frame

        super().__init__(*parameters, children=children)

    @property
    def bbox(self):
        """Hyper-spectral bounding box of this model
        """
        return self._bbox

    @bbox.setter
    def bbox(self, b):
        """Sets the bounding box of this component.

        Parameters
        ----------
        b: `~scarlet.Box`
            New bounding box of this model
        """
        if b is None:
            b = self._frame.bbox
        self._bbox = b

        self._model_frame_slices, self._model_slices = overlapped_slices(
            self._frame.bbox, self._bbox
        )

    @property
    def frame(self):
        """Hyper-spectral characteristics is this model
        """
        return self._frame

    @frame.setter
    def frame(self, f):
        """Sets the frame for this component.

        Parameters
        ----------
        f: `~scarlet.Frame`
            New frame of the model
        """
        self._frame = f
        self._model_frame_slices, self._model_slices = overlapped_slices(
            self._frame.bbox, self._bbox
        )

    def model_to_box(self, bbox=None, model=None):
        """Project a model into a frame


        Parameters
        ----------
        model: array
            Image of the model to project.
            This must be the same shape as `self.bbox`.
            If `model` is `None` then `self.get_model()` is used.
        bbox: `~scarlet.bbox.Box`
            The to project the model into.
            If `bbox` is `None` then the model is projected
            into `self.model_frame.bbox`.

        Returns
        -------
        projected_model: array
            (Channels, Height, Width) image of the model
        """
        # Use the current model by default
        if model is None:
            model = self.get_model()
        # Use the full model frame by default
        if bbox is None or bbox == self.frame.bbox:
            bbox = self.frame.bbox
            frame_slices = self._model_frame_slices
            model_slices = self._model_slices
        else:
            frame_slices, model_slices = overlapped_slices(bbox, self.bbox)

        result = np.zeros(bbox.shape, dtype=model.dtype)
        result[frame_slices] = model[model_slices]
        return result


class FactorizedComponent(Component):
    """A single component in a blend

    Uses the non-parametric factorization Spectrum x Morphology.

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
        assert isinstance(spectrum, Spectrum)
        assert isinstance(morphology, Morphology)

        bbox = spectrum.bbox @ morphology.bbox[-2:]

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
        if len(morphology.shape) == 2:
            model = spectrum[:, None, None] * morphology[None, :, :]
        elif len(morphology.shape) == 3:
            model = spectrum[:, None, None] * morphology
        else:
            raise AttributeError("morphology must be 2D or 3D")

        # project the model into frame (if necessary)
        if frame is not None:
            model = self.model_to_box(frame.bbox, model)
        return model

    def update(self):
        for child in self.children:
            try:
                child.update()
            except UpdateException as e:
                spectrum, morphology = self.children
                bbox = spectrum.bbox @ morphology.bbox[-2:]
                self.bbox = bbox
                raise e


class CubeComponent(Component):
    """A single component in a blend

    Uses full hyperspectral cube parameterization.

    Parameters
    ----------
    frame: `~scarlet.Frame`
        The spectral and spatial characteristics of this component.
    cube: `~scarlet.Parameter`
        3D array (C, Height, Width) of the hyperspectral cube.
    bbox: `~scarlet.Box`
        Hyper-spectral bounding box of this component.
    """

    def __init__(self, frame, cube, bbox=None):
        if isinstance(cube, Parameter):
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
            model = self.model_to_box(frame.bbox, model)
        return model


class CombinedComponent(Component):
    """Combination of multiple `~scarlet.Component` instances

    Parameters
    ----------
    components: list of `~scarlet.Component`
    operation: 'add' or 'multiply'
        The combination operation of the children's models
    """

    def __init__(self, components, operation="add"):

        assert len(components)
        frame = components[0].frame
        box = components[0].bbox
        # all children need to have the same bbox for simple autogradable combinations
        for c in components:
            assert isinstance(c, Component)
            assert c.frame is frame

        super().__init__(frame, children=components, bbox=box)

        assert operation in ["add", "multiply"]
        self.operation = operation

    def get_model(self, *parameters, frame=None):
        # boxed models
        models = self.get_models_of_children(*parameters, frame=None)

        bbox = self.bbox
        model = np.zeros(bbox.shape)

        for k, model_ in enumerate(models):
            c = self.children[k]

            if c.bbox != bbox:
                padding = tuple(
                    (c.bbox.start[d] - bbox.start[d], bbox.stop[d] - c.bbox.stop[d])
                    for d in range(bbox.D)
                )
                model_ = fast_zero_pad(model_, padding)

            if self.operation == "add":
                model += model_
            elif self.operation == "multiply":
                model *= model_

        if frame is not None:
            model = self.model_to_box(frame.bbox, model)
        return model

    def update(self):
        for child in self.children:
            try:
                child.update()
            except UpdateException as e:
                # Make the bbox of the tree the union of the component bboxes
                box = self.children[0].bbox.copy()
                for c in self.children[1:]:
                    box |= c.bbox
                self.bbox = box  # super setter method
                raise e

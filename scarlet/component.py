from abc import ABC, abstractmethod
from functools import partial
import autograd.numpy as np

from .frame import Frame
from .parameter import Parameter
from .bbox import Box, overlapped_slices


def _get_parameter(self, name, *parameters):
    # check parameters first during optimization
    for p in parameters:
        # if parameters are autograd ArrayBoxes
        # need to access the wrapped class with _value
        if p._value.name == name:
            return p

    # find them from self (use all even if fixed!)
    names = tuple(p.name for p in self._parameters)
    try:
        return self._parameters[names.index(name)]
    except ValueError:
        return None


class Component(ABC):
    """A single component in a blend.

    This class acts as base for building a complex :class:`scarlet.blend.Blend`.

    Parameters
    ----------
    model_frame: `~scarlet.Frame`
        The spectral and spatial characteristics of the model.
    parameters: list of `~scarlet.Parameter`
    bbox: `~scarlet.Box`
        Hyper-spectral bounding box of this component
    kwargs: dict
        Auxiliary information attached to this component.
    """

    def __init__(self, model_frame, *parameters, bbox=None, **kwargs):

        if bbox is None:
            bbox = model_frame.bbox
        self._bbox = bbox
        self.set_model_frame(model_frame)

        if hasattr(parameters, "__iter__"):
            for p in parameters:
                assert isinstance(p, Parameter)
            self._parameters = parameters
        else:
            assert isinstance(parameters, Parameter)
            self._parameters = tuple(parameters)
        self.check_parameters()

        # additional non-optimization parameters of component
        self.kwargs = kwargs

    @property
    def bbox(self):
        """Hyper-spectral bounding box of this component (Channel, Height, Width)
        """
        return self._bbox

    @property
    def parameters(self):
        """The list of non-fixed parameters

        Returns
        -------
        list of parameters available for optimization
        If `parameter.fixed == True`, the parameter will not returned here.
        """
        return tuple(p for p in self._parameters if not p.fixed)

    @property
    def parameter_names(self):
        names = tuple(p.name for p in self._parameters if not fixed)

    def get_parameter(self, name, *parameters):
        return _get_parameter(self, name, *parameters)

    @abstractmethod
    def get_model(self, *parameters, frame=None):
        """Get the model for this component

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
        pass

    def set_model_frame(self, model_frame):
        """Sets the frame for this component.

        Each component needs to know the properties of the Frame and,
        potentially, the subvolume it covers.

        Parameters
        ----------
        model_frame: `~scarlet.Frame`
            Frame of the model
        """
        self.model_frame = model_frame
        self.model_frame_slices, self.model_slices = overlapped_slices(
            model_frame.bbox, self.bbox
        )

    def check_parameters(self):
        """Check that all parameters have finite elements

        Raises
        ------
        `ArithmeticError` when non-finite elements are present
        """
        for k, p in enumerate(self._parameters):
            if not np.isfinite(p).all():
                msg = "Component {}, Parameter '{}' is not finite:\n{}".format(
                    self.__class__.__name__, p.name, p
                )
                raise ArithmeticError(msg)

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
        if frame is None or frame == self.model_frame:
            frame = self.model_frame
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


class Factor(ABC):
    def __init__(self, model_frame, *parameters, bbox=None):
        self._parameters = parameters
        self._bbox = bbox

    @property
    def parameters(self):
        return self._parameters

    @property
    def bbox(self):
        return self._bbox

    @abstractmethod
    def get_model(self, *parameters):
        pass

    def get_parameter(self, name, *parameters):
        return _get_parameter(self, name, *parameters)


class FactorizedComponent(Component):
    """A single component in a blend.

    Uses the non-parametric factorization sed x morphology.

    Parameters
    ----------
    model_frame: `~scarlet.Frame`
        The spectral and spatial characteristics of the full model.
    bbox: `~scarlet.Box`
        Hyper-spectral bounding box of this component.
    spectrum: `~scarlet.Spectrum`
        Parameterization of the spectrum
    morphology: `~scarlet.Morphology`
        Parameterization of the morphology.
    """

    def __init__(self, model_frame, spectrum, morphology):
        assert isinstance(spectrum, Factor)
        assert isinstance(morphology, Factor)
        bbox = spectrum.bbox @ morphology.bbox
        parameters = spectrum.parameters + morphology.parameters
        super().__init__(model_frame, *parameters, bbox=bbox)
        self._spectrum = spectrum
        self._morphology = morphology

    @property
    def spectrum(self):
        """Numpy view of the component SED
        """
        return self._spectrum.get_model()

    @property
    def morphology(self):
        """Numpy view of the component morphology
        """
        return self._morphology.get_model()

    @property
    def center(self):
        if hasattr(self._morphology, "center"):
            return self._morphology.center
        else:
            return None

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
        spectrum = self._spectrum.get_model(*parameters)
        morph = self._morphology.get_model(*parameters)
        model = spectrum[:, None, None] * morph[None, :, :]
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
        assert isinstance(cube, Parameter) and cube.name == "cube"
        parameters = (cube,)
        super().__init__(frame, *parameters, bbox=bbox)

    @property
    def cube(self):
        return self.get_parameter("cube")._data

    def get_model(self, *parameters, frame=None):
        cube = self.get_parameter("cube", *parameters)
        if frame is not None:
            cube = self.model_to_frame(frame, cube)
        return cube


class CombinedComponent(Component):
    def __init__(self, model_frame, components, mode="add"):
        if hasattr(components, "__iter__"):
            self.components = components
        else:
            self.components = (components,)

        assert mode in ["add", "multiply"]
        self.mode = mode

        parameters = []
        for c in self.components:
            parameters += c.parameters

        super().__init__(model_frame, *parameters, bbox=self.bbox)

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

from abc import ABC, abstractmethod
from functools import partial

import autograd.numpy as np
from autograd.extend import defvjp, primitive

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

        # Properties used for indexing in the ComponentTree
        self._index = None
        self._parent = None

    @property
    def bbox(self):
        """Hyper-spectral bounding box of this component (Channel, Height, Width)
        """
        return self._bbox

    @property
    def coord(self):
        """The coordinate in a `~scarlet.component.ComponentTree`.
        """
        if self._index is not None:
            if self._parent._index is not None:
                return tuple(self._parent.coord) + (self._index,)
            else:
                return (self._index,)

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


class ComponentTree:
    """Base class for hierarchical collections of Components.
    """

    def __init__(self, components):
        """Constructor

        Group a list of `~scarlet.component.Component`s in a hierarchy.

        Parameters
        ----------
        components: list of `~scarlet.component.Component` or `~scarlet.component.ComponentTree`
        """
        if not hasattr(components, "__iter__"):
            components = (components,)

        # check type and set coords of subordinate nodes in tree
        self._tree = tuple(components)
        self._index = None
        self._parent = None
        for i, c in enumerate(self._tree):
            if not isinstance(c, ComponentTree) and not isinstance(c, Component):
                raise NotImplementedError(
                    "argument needs to be list of Components or ComponentTrees"
                )
            assert (
                c.model_frame is self.model_frame
            ), "All components need to share the same model Frame"
            c._index = i
            c._parent = self

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

    @property
    def components(self):
        """Flattened tuple of all components in the tree.

        CAUTION: Each component in a tree can only be a leaf of a single node.
        While one can construct trees that hold the same component multiple
        times, this method will only return that component at its first
        encountered location
        """
        try:
            return self._components
        except AttributeError:
            self._components = self._tree_to_components()
            return self._components

    def _tree_to_components(self):
        components = []
        for c in self._tree:
            if isinstance(c, ComponentTree):
                _c = c.components
            else:
                _c = [c]
            # check uniqueness
            for __c in _c:
                if __c not in components:
                    components.append(__c)
        return tuple(components)

    @property
    def n_components(self):
        """Number of components.
        """
        return len(self.components)

    @property
    def K(self):
        """Number of components.
        """
        return self.n_components

    @property
    def model_frame(self):
        """Frame of the components.
        """
        return self._tree[0].model_frame

    @property
    def sources(self):
        """Initial list of components or sources that generate the tree.

        This can be different than `self.components` because sources can
        have multiple components.

        Returns
        -------
        The arguments of `__init__`
        """
        return self._tree

    @property
    def n_sources(self):
        """Number of initial sources or components.

        This can be different than `self.n_components` because sources can
        have multiple components.

        Returns
        -------
        int: number of initial sources
        """
        return len(self._tree)

    @property
    def coord(self):
        """The coordinate in tree.

        The coordinate can be used to traverse the tree and for `__getitem__`.
        """
        if self._index is not None:
            if self._parent._index is not None:
                return tuple(self._parent.coord) + (self._index,)
            else:
                return (self._index,)

    @property
    def parameters(self):
        """The list of non-fixed parameters

        Returns
        -------
        list of parameters available for optimization
        If `parameter.fixed == True`, the parameter will not returned here.
        """
        pars = []
        for c in self.components:
            pars += c.parameters
        return pars

    def check_parameters(self):
        """Check that all parameters have finite elements

        Raises
        ------
        `ArithmeticError` when non-finite elements are present
        """
        for c in self.components:
            c.check_parameters()

    def get_model(self, *params, frame=None):
        """Get the model of this component tree

        Parameters
        ----------
        params: tuple of optimization parameters

        Returns
        -------
        model: array
            (Bands, Height, Width) data cube
        """
        if frame is None:
            frame = Frame(
                self.bbox,
                dtype=self.model_frame.dtype,
                psfs=self.model_frame.psf,
                channels=self.model_frame.channels,
            )
        # If this is the model frame then the slices are already cached
        if frame == self.model_frame:
            use_cached = True
        else:
            use_cached = False

        full_model = np.zeros(frame.shape, dtype=frame.dtype)

        models = []
        slices = []
        i = 0

        for k, c in enumerate(self.components):
            if len(params):
                j = len(c.parameters)
                p = params[i : i + j]
                i += j
                model = c.get_model(*p)
            else:
                model = c.get_model()

            models.append(model)

            if use_cached:
                slices.append((c.model_frame_slices, c.model_slices))
            else:
                # Get the slices needed to insert the model
                slices.append(overlapped_slices(frame.bbox, c.bbox))

        # We have to declare the function that inserts sources
        # into the blend with autograd.
        # This has to be done each time we fit a blend,
        # since the number of components => the number of arguments,
        # which must be linked to the autograd primitive function.
        defvjp(
            _add_models,
            *([partial(_grad_add_models, index=k) for k in range(len(self.components))])
        )

        full_model = _add_models(*models, full_model=full_model, slices=slices)

        return full_model

    def set_model_frame(self, model_frame):
        """Set the frame for all components in the tree

        see `~scarlet.Component.set_model_frame` for details.

        Parameters
        ----------
        frame: `~scarlet.Frame`
            Frame to adopt for this component
        """
        for c in self.components:
            c.set_model_frame(model_frame)

    def __iadd__(self, c):
        """Add another component or tree.

        Parameters
        ----------
        c: `~scarlet.component.Component` or `~scarlet.component.ComponentTree`
        """
        c_index = self.n_sources
        if isinstance(c, ComponentTree):
            self._tree = self._tree + c._tree
        elif isinstance(c, Component):
            self._tree = self._tree + (c,)
        else:
            raise NotImplementedError("argument needs to be Component or ComponentTree")
        c._index = c_index
        c._parent = self
        self._components = self._tree_to_components()
        return self

    def __getitem__(self, coord):
        """Access node in the tree.

        Parameters
        ----------
        coords: int or tuple of ints

        Returns
        -------
        `~scarlet.component.Component` or `~scarlet.component.ComponentTree`
        """
        if isinstance(coord, (tuple, list)):
            if len(coord) > 1:
                return self._tree[coord[0]].__getitem__(coord[1:])
            else:
                return self._tree[coord[0]]
        elif isinstance(coord, int):
            return self._tree[coord]
        else:
            raise NotImplementedError("coord needs to be index or list of indices")

    def __getstate__(self):
        # needed for pickling to understand what to save
        return (self._tree,)

    def __setstate__(self, state):
        self._tree = state[0]
        self._tree_to_components()

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
        if frame is None:
            frame = self.model_frame

        if hasattr(frame, "dtype"):
            dtype = frame.dtype
        else:
            dtype = self.model_frame.dtype

        frame_slices, model_slices = overlapped_slices(frame, self.bbox)

        result = np.zeros(frame.shape, dtype=dtype)
        result[frame_slices] = model[model_slices]
        return result

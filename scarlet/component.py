from abc import ABC, abstractmethod
from functools import partial

import autograd.numpy as np
from autograd.extend import defvjp, primitive

from .frame import Frame
from .parameter import Parameter
from . import fft
from . import interpolation
from .bbox import Box, overlapped_slices


class Component(ABC):
    """A single component in a blend.

    This class acts as base for building a complex :class:`scarlet.blend.Blend`.

    Parameters
    ----------
    model_frame: `~scarlet.Frame`
        The spectral and spatial characteristics of the model.
    bbox: `~scarlet.Box`
        Hyper-spectral bounding box
    parameters: list of `~scarlet.Parameter`
    kwargs: dict
        Auxiliary information attached to this component.
    """

    def __init__(self, model_frame, bbox, *parameters, **kwargs):
        self.bbox = bbox
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
    def shape(self):
        """Shape of the image (Channel, Height, Width)
        """
        return self.bbox.shape

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
        return [p for p in self._parameters if not p.fixed]

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
        self.model_frame_slices, self.model_slices = overlapped_slices(model_frame, self.bbox)

    def check_parameters(self):
        """Check that all parameters have finite elements

        Raises
        ------
        `ArithmeticError` when non-finite elements are present
        """
        for k, p in enumerate(self._parameters):
            if not np.isfinite(p).all():
                msg = "Component {} Parameter {} is not finite:\n{}".format(self, k, p)
                raise ArithmeticError(msg)

    def project(self, model=None, frame=None):
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
            frame_slices, model_slices = overlapped_slices(frame, self.bbox)

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
    model_frame: `~scarlet.Frame`
        The spectral and spatial characteristics of the full model.
    bbox: `~scarlet.Box`
        Hyper-spectral bounding box
    sed: `~scarlet.Parameter`
        1D array (channels) of the initial SED.
    morph: `~scarlet.Parameter`
        Image (Height, Width) of the initial morphology.
    shift: `~scarlet.Parameter`
        2D position for the shift of the center
    """

    def __init__(self, model_frame, bbox, sed, morph, shift=None, **kwargs):
        if shift is None:
            parameters = (sed, morph)
        else:
            parameters = (sed, morph, shift)
        super().__init__(model_frame, bbox, *parameters, **kwargs)

        # store shifting structures
        if shift is not None:
            padding = 10
            self.fft_shape = fft._get_fft_shape(morph, morph, padding=padding)
            self.shifter_y, self.shifter_x = interpolation.mk_shifter(self.fft_shape)

    @property
    def sed(self):
        """Numpy view of the component SED
        """
        return self._parameters[0]._data

    @property
    def morph(self):
        """Numpy view of the component morphology
        """
        return self._shift_morph(self.shift, self._parameters[1]._data)

    @property
    def shift(self):
        """Numpy view of the component shift
        """
        if len(self._parameters) == 3:
            return self._parameters[2]._data
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
        sed, morph, shift = None, None, None

        # if params are set they are not Parameters, but autograd ArrayBoxes
        # need to access the wrapped class with _value
        for p in parameters:
            if p._value is self._parameters[0]:
                sed = p
            if p._value is self._parameters[1]:
                morph = p
            if len(self._parameters) == 3 and p._value is self._parameters[2]:
                shift = p

        if sed is None:
            sed = self.sed

        if morph is None:
            morph = self._parameters[1]._data

        if shift is None:
            shift = self.shift

        morph = self._shift_morph(shift, morph)
        model = sed[:, None, None] * morph[None, :, :]
        # project the model into frame (if necessary)
        if frame is not None:
            model = self.project(model, frame)
        return model

    def _shift_morph(self, shift, morph):
        if shift is not None:
            X = fft.Fourier(morph)
            X_fft = X.fft(self.fft_shape, (0, 1))

            # Apply shift in Fourier
            result_fft = (
                X_fft
                * np.exp(self.shifter_y[:, None] * shift[0])
                * np.exp(self.shifter_x[None, :] * shift[1])
            )

            X = fft.Fourier.from_fft(result_fft, self.fft_shape, X.shape, [0, 1])
            return np.real(X.image)
        return morph


class FunctionComponent(FactorizedComponent):
    """A single component in a blend.

    Uses the non-parametric sed x morphology, with the morphology specified
    by a functional expression.

    Parameters
    ----------
    model_frame: `~scarlet.Frame`
        The spectral and spatial characteristics of the full model.
    bbox: `~scarlet.Box`
        Hyper-spectral bounding box
    sed: `~scarlet.Parameter`
        1D array (channels) of the initial SED.
    fparams: `~scarlet.Parameter`
        Parameters of the initial morphology.
    func: `autograd` function
        Signature: func(*fparams, y=None, x=None) -> Image (Height, Width)
    """

    def __init__(self, model_frame, bbox, sed, fparams, func):
        parameters = (sed, fparams)
        super().__init__(model_frame, bbox, *parameters, func=func)

    @property
    def morph(self):
        """Numpy view of the component morphology
        """
        try:
            return self._morph
        except AttributeError:
            # Cache morph. This is updated in get_model if fparams changes
            self._morph = self._func(*self._parameters[1])
        return self._morph

    def _func(self, *parameters):
        return self.kwargs["func"](*parameters)

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
        sed, fparams = None, None

        # if params are set they are not Parameters, but autograd ArrayBoxes
        # need to access the wrapped class with _value
        for p in parameters:
            if p._value is self._parameters[0]:
                sed = p
            if p._value is self._parameters[1]:
                fparams = p

        if sed is None:
            sed = self.sed
        if fparams is None:
            morph = self.morph
        else:
            morph = self._func(*fparams)
            self._morph = morph._value

        model = sed[:, None, None] * morph[None, :, :]
        if frame is not None:
            model = self.project(model, frame)
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
        Hyper-spectral bounding box
    """

    def __init__(self, frame, bbox, cube):
        parameters = (cube,)
        super().__init__(frame, bbox, *parameters)

    @property
    def cube(self):
        return self._parameters[0]._data

    def get_model(self, *parameters, frame=None):
        cube = None
        for p in parameters:
            if p._value is self._parameters[0]:
                cube = p

        if cube is None:
            cube = self.cube
        if frame is not None:
            cube = self.project(cube, frame)
        return cube


@primitive
def _add_models(*models, full_model, slices):
    """Insert the models into the full model
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
            assert c.model_frame is self.model_frame, "All components need to share the same model Frame"
            c._index = i
            c._parent = self

        # Make the bounding box of the tree the union of the component boxes
        box = self.components[0].bbox
        self._bbox = Box(box.shape, box.origin)
        for component in components:
            self._bbox |= component.bbox

    @property
    def bbox(self):
        """Union of all the component `~scarlet.bbox.Box`es
        """
        return self._bbox

    @property
    def shape(self):
        """Shape of this model
        """
        return self._bbox.shape

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

        # We have to declare the function that inserts sources
        # into the blend with autograd.
        # This has to be done each time we fit a blend,
        # since the number of components => the number of arguments,
        # which must be linked to the autograd primitive function
        defvjp(_add_models, *([partial(_grad_add_models, index=k) for k in range(len(self.components))]))

        if frame is None:
            frame = Frame(self.bbox, dtype=self.model_frame.dtype, psfs=self.model_frame.psf)
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
                p = params[i: i + j]
                i += j
                model = c.get_model(*p)
            else:
                model = c.get_model()

            models.append(model)

            if use_cached:
                slices.append((c.model_frame_slices, c.model_slices))
            else:
                # Get the slices needed to insert the model
                slices.append(overlapped_slices(frame, c.bbox))
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

    def project(self, model=None, frame=None):
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

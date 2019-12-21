from abc import ABC, abstractmethod
from .parameter import *
from . import fft
from . import interpolation
from .bbox import Box
import autograd.numpy as np


class Component(ABC):
    """A single component in a blend.

    This class acts as base for building a complex :class:`scarlet.blend.Blend`.

    Parameters
    ----------
    frame: `~scarlet.Frame`
        The spectral and spatial characteristics of this component.
    parameters: list of `~scarlet.Parameter`
    bbox: `~scarlet.Box`
        Spatial bounding box of the morphology.
    kwargs: dict
        Auxiliary information attached to this component.
    """
    def __init__(self, frame, *parameters, bbox=None, **kwargs):
        self.bbox = bbox
        self.set_frame(frame)

        if hasattr(parameters, '__iter__'):
            for p in parameters:
                assert isinstance(p, Parameter)
            self._parameters = parameters
        else:
            assert isinstance(parameters, Parameter)
            self._parameters = tuple(parameters,)

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
        return [ p for p in self._parameters if not p.fixed ]

    @abstractmethod
    def get_model(self, *parameters):
        """Get the model for this component

        Parameters
        ----------
        parameters: tuple of optimimzation parameters

        Returns
        -------
        model: array
            (Channels, Height, Width) image of the model
        """
        pass

    def freeze(self):
        """Fix all parameters

        The component will not provide optimizable parameters anymore.
        """
        for p in self._parameters:
            p.fixed = True

    def unfreeze(self):
        """Release all parameters

        The component will provide *all* parameters as optimizable parameters.
        Calling this function overrides previous setting of `parameter.fixed` for every
        parameter of this component.
        """
        for p in self._parameters:
            p.fixed = False

    def set_frame(self, frame):
        """Sets the frame for this component.

        Each component needs to know the properties of the Frame and, potentially, the
        subvolume it covers.

        Parameters
        ----------
        frame: `~scarlet.Frame`
            Frame to adopt for this component
        """
        self.frame = frame

        # store padding and slicing structures
        if self.bbox is not None:
            assert isinstance(self.bbox, Box)
            # TODO: full 3D bbox and slicing support
            # determine pad from box into full frame
            # yields superset of frame pixels
            # pad_width is ((before1, after1), (before2, after2)...)
            self.pad_width = (
            (max(0, self.bbox.front - self.frame.front), max(0, self.frame.back - self.bbox.back)),
            (max(0, self.bbox.bottom - self.frame.bottom), max(0, self.frame.top - self.bbox.top)),
            (max(0, self.bbox.left - self.frame.left), max(0, self.frame.right- self.bbox.right)))

            # get slicing of padded box so that the result covers
            # all of the model frame
            front = self.bbox.front - self.pad_width[0][0]
            back = self.bbox.back + self.pad_width[0][1]
            bottom = self.bbox.bottom - self.pad_width[1][0]
            top = self.bbox.top + self.pad_width[1][1]
            left = self.bbox.left - self.pad_width[2][0]
            right = self.bbox.right + self.pad_width[2][1]
            padded_box = Box.from_bounds(front, back, bottom, top, left, right)

            model_box = self.frame
            overlap = model_box & padded_box
            overlap -= padded_box.origin # now in padded frame
            self.slices = overlap.slices_for(padded_box.shape)


class FactorizedComponent(Component):
    """A single component in a blend.

    Uses the non-parametric factorization sed x morphology.

    Parameters
    ----------
    frame: `~scarlet.Frame`
        The spectral and spatial characteristics of this component.
    sed: `~scarlet.Parameter`
        1D array (channels) of the initial SED.
    morph: `~scarlet.Parameter`
        Image (Height, Width) of the initial morphology.
    shift: `~scarlet.Parameter`
        2D position for the shift of the center
    bbox: `~scarlet.Box`
        Spatial bounding box of the morphology.
    """
    def __init__(self, frame, sed, morph, shift=None, bbox=None, **kwargs):
        if shift is None:
            parameters = (sed, morph)
        else:
            parameters = (sed, morph, shift)
        super().__init__(frame, *parameters, bbox=bbox, **kwargs)

        # store shifting structures
        if shift is not None:
            padding = 10
            self.fft_shape = fft._get_fft_shape(morph, morph, padding=padding)
            self.shifter_y, self.shifter_x = interpolation.mk_shifter(self.fft_shape)

    @property
    def sed(self):
        """Numpy view of the component SED
        """
        return self._pad_sed(self._parameters[0]._data)

    @property
    def morph(self):
        """Numpy view of the component morphology
        """
        return self._pad_morph(self._shift_morph(self.shift, self._parameters[1]._data))

    @property
    def shift(self):
        """Numpy view of the component shift
        """
        if len(self._parameters) == 3:
            return self._parameters[2]._data
        return None

    def get_model(self, *parameters):
        """Get the model for this component.

        Parameters
        ----------
        parameters: tuple of optimimzation parameters

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

        if shift is None:
            shift = self.shift

        if morph is None:
            # dont' use self._morph because we could have shift as parameter
            morph =  self._pad_morph(self._shift_morph(shift, self._parameters[1]._data))
        else:
            morph =  self._pad_morph(self._shift_morph(shift, morph))

        return sed[:, None, None] * morph[None, :, :]

    def _pad_sed(self, sed):
        if self.bbox is not None:
            padded = np.pad(sed, self.pad_width[0], mode='constant', constant_values=0)
            return padded[self.slices[0]]
        else:
            return sed

    def _pad_morph(self, morph):
        if self.bbox is not None:
                padded = np.pad(morph, self.pad_width[1:], mode='constant', constant_values=0)
                return padded[self.slices[1:]]
        return morph

    def _shift_morph(self, shift, morph):
        if shift is not None:
            X = fft.Fourier(morph)
            X_fft = X.fft(self.fft_shape, (0,1))

            # Apply shift in Fourier
            result_fft = X_fft * (self.shifter_y[:, None] ** shift[0]) * (self.shifter_x[None, :] ** shift[1])

            X = fft.Fourier.from_fft(result_fft, self.fft_shape, X.shape, [0,1])
            return np.real(X.image)
        return morph


class FunctionComponent(FactorizedComponent):
    """A single component in a blend.

    Uses the non-parametric sed x morphology, with the morphology specified
    by a functional expression.

    Parameters
    ----------
    frame: `~scarlet.Frame`
        The spectral and spatial characteristics of this component.
    sed: `~scarlet.Parameter`
        1D array (channels) of the initial SED.
    fparams: `~scarlet.Parameter`
        Parameters of the initial morphology.
    func: `autograd` function
        Signature: func(*fparams, y=None, x=None) -> Image (Height, Width)
    bbox: `~scarlet.Box`
        Spatial bounding box of the morphology.
    """
    def __init__(self, frame, sed, fparams, func, bbox=None):
        parameters = (sed, fparams)
        super().__init__(frame, *parameters, bbox=bbox, func=func)

    @property
    def morph(self):
        """Numpy view of the component morphology
        """
        try:
            return self._pad_morph(self._morph)
        except AttributeError:
            self._morph = self._func(*self._parameters[1])
            return self._pad_morph(self._morph)

    def _func(self, *parameters):
        return self.kwargs['func'](*parameters)

    def get_model(self, *parameters):
        """Get the model for this component.

        Parameters
        ----------
        parameters: tuple of optimimzation parameters

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
            morph = self._pad_morph(morph)

        return sed[:, None, None] * morph[None, :, :]


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
        Spatial bounding box of the morphology.
    """
    def __init__(self, frame, cube, bbox=None):
        parameters = (cube,)
        super().__init__(frame, *parameters, bbox=bbox)

    @property
    def cube (self):
        return self._pad_cube(self._parameters[0]._data)

    def get_model(self, *parameters):
        cube = None
        for p in parameters:
            if p._value is self._parameters[0]:
                cube = self._pad_cube(p)

        if cube is None:
            cube = self.cube

        return cube

    def _pad_cube(self, cube):
        if self.bbox is not None:
            padded = np.pad(cube, self.pad_width, mode='constant', constant_values=0)
            return padded[self.slices]
        return cube


class ComponentTree():
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
                raise NotImplementedError("argument needs to be list of Components or ComponentTrees")
            assert c.frame is self.frame, "All components need to share the same Frame"
            c._index = i
            c._parent = self

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
    def frame(self):
        """Frame of the components.
        """
        return self._tree[0].frame

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

    def get_model(self, *params):
        """Get the model of this component tree

        Parameters
        ----------
        params: tuple of optimization parameters

        Returns
        -------
        model: array
            (Bands, Height, Width) data cube
        """
        model = np.zeros(self.frame.shape)
        if len(params):
            i = 0
            for k,c in enumerate(self.components):
                j = len(c.parameters)
                p = params[i:i+j]
                i += j
                model = model + c.get_model(*p)
        else:
            for c in self.components:
                model = model + c.get_model()

        return model

    def set_frame(self, frame):
        """Set the frame for all components in the tree

        see `~scarlet.Component.set_frame` for details.

        Parameters
        ----------
        frame: `~scarlet.Frame`
            Frame to adopt for this component
        """
        for c in self.components:
            c.set_frame(frame)

    def freeze(self):
        """Fix all parameters

        The tree will not provide optimizable parameters anymore.
        """
        for c in self.components:
            c.freeze()

    def unfreeze(self):
        """Release all parameters

        The tree will provide *all* parameters as optimizable parameters.
        Calling this function overrides previous setting of `parameter.fixed` for every
        parameter of this component tree.
        """
        for c in self.components:
            c.unfreeze()

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
        return (self._tree, )

    def __setstate__(self, state):
        self._tree = state[0]
        self._tree_to_components()

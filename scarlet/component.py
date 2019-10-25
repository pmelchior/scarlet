from .parameter import *
from . import fft
from . import interpolation
import pickle
import autograd.numpy as np

import logging
logger = logging.getLogger("scarlet.component")


class Component():
    """A single component in a blend.

    This class acts as base for building a complex :class:`scarlet.blend.Blend`.

    Parameters
    ----------
    frame: `~scarlet.Frame`
        The spectral and spatial characteristics of this component.
    parameters: list of `~scarlet.Parameter`
    """

    def __init__(self, frame, *parameters):
        self._frame = frame

        if hasattr(parameters, '__iter__'):
            for p in parameters:
                isinstance(p, Parameter)
            self._parameters = parameters
        else:
            assert isinstance(parameters, Parameter)
            self._parameters = tuple(parameters,)

        # Properties used for indexing in the ComponentTree
        self._index = None
        self._parent = None

    @property
    def shape(self):
        """Shape of the image (Channel, Height, Width)
        """
        return self._frame.shape

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
    def frame(self):
        """The frame of this component
        """
        return self._frame

    @property
    def parameters(self):
        return [ p for p in self._parameters if not p.fixed ]

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
        pass

    def get_flux(self):
        """Get flux in every band
        """
        return self.get_model().sum(axis=(1,2))

    def update(self):
        """Update the component

        This method can be overwritten in inherited classes to
        run proximal operators or other component update functions
        that will be executed during fitting.
        """
        return self

    def __getstate__(self):
        # needed for pickling to understand what to save
        return tuple([self._sed.copy(), self._morph.copy()])

    def __setstate__(self, state):
        self._sed, self._morph = state

    def save(self, filename):
        fp = open(filename, "wb")
        pickle.dump(self, fp)
        fp.close()

    @classmethod
    def load(cls, filename):
        fp = open(filename, "rb")
        return pickle.load(fp)


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
    """
    def __init__(self, frame, sed, morph):
        self._sed = sed
        self._morph = morph
        parameters = (self._sed, self._morph)
        super().__init__(frame, *parameters)

    @property
    def sed(self):
        """Numpy view of the component SED
        """
        return self._sed._data

    @property
    def morph(self):
        """Numpy view of the component morphology
        """
        return self._morph._data

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
        sed, morph = self.sed, self.morph

        # if params are set they are not Parameters, but autograd ArrayBoxes
        # need to access the wrapped class with _value
        for p in parameters:
            if p._value is self._sed:
                sed = p
            if p._value is self._morph:
                morph = p
        return sed[:, None, None] * morph[None, :, :]

    def get_flux(self):
        """Get flux in every band
        """
        return self.morph.sum() * self.sed


class ShiftedFactorizedComponent(Component):
    """A single component in a blend.

    Uses the non-parametric factorization sed x morphology with additional
    optimization of the center shift.
    This is important when constraints require accurate centering
    (e.g. symmetry and monotonicity).

    Parameters
    ----------
    frame: `~scarlet.Frame`
        The spectral and spatial characteristics of this component.
    shift: `~scarlet.Parameter`
        2D position for the shift of the center
    sed: `~scarlet.Parameter`
        array (Channels) of the initial SED.
    morph: `~scarlet.Parameter`
        Image (Height, Width) of the initial morphology.
    """
    def __init__(self, frame, shift, sed, morph):
        self._shift = shift
        self._sed = sed
        self._morph = morph
        parameters = (self._shift, self._sed, self._morph)
        super().__init__(frame, *parameters)

    @property
    def shift(self):
        """Numpy view of the component center
        """
        return self._shift._data

    @property
    def sed(self):
        """Numpy view of the component SED
        """
        return self._sed._data

    @property
    def morph(self):
        """Numpy view of the component morphology
        """
        return self._shift_morph(self._shift._data, self._morph._data)

    def get_flux(self):
        """Get flux in every band
        """
        return self.morph.sum() * self.sed

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
        shift, sed, morph = self.shift, self.sed, self.morph

        # if params are set they are not Parameters, but autograd ArrayBoxes
        # need to access the wrapped class with _value
        for p in parameters:
            if p._value is self._shift:
                shift = p
            if p._value is self._sed:
                sed = p
            if p._value is self._morph:
                morph = p
            morph = self._shift_morph(shift, morph)
            #self._morph[:,:] = morph._value
        return sed[:, None, None] * morph[None, :, :]

    def _shift_morph(self, shift, X):
        padding = 10
        fft_shape = fft._get_fft_shape(X, X, padding=padding)
        X = fft.Fourier(X)
        X_fft = X.fft(fft_shape, (0,1))
        #zeroMask = X.image <= 0

        shifter_y, shifter_x = interpolation.mk_shifter(fft_shape)
        # Apply shift in Fourier
        result_fft = X_fft * shifter_y[:, np.newaxis] ** (-shift[0])
        result_fft *= shifter_x[np.newaxis, :] ** (-shift[1])

        X = fft.Fourier.from_fft(result_fft, fft_shape, X.image.shape, [0,1])

        #X.image[zeroMask] = 0
        return np.real(X.image)

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
    """
    def __init__(self, frame, sed, fparams, func):
        self._sed = sed
        self._fparams = fparams
        parameters = (self._sed, self._fparams)
        super().__init__(frame, *parameters)

        y = np.arange(frame.shape[1])
        x = np.arange(frame.shape[2])
        from functools import partial
        self._func = partial(func, y=y, x=x)
        self._morph = self._func(*self._fparams)

    @property
    def morph(self):
        """Numpy view of the component morphology
        """
        return self._morph

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
        sed, morph = self.sed, self.morph

        # if params are set they are not Parameters, but autograd ArrayBoxes
        # need to access the wrapped class with _value
        for p in parameters:
            if p._value is self._sed:
                sed = p
            if p._value is self._fparams:
                morph = self._func(*p)
                self._morph[:,:] = morph._value
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
    """
    def __init__(self, frame, cube):
        self._cube = cube
        self._morph = morph
        parameters = (self._cube,)
        super().__init__(frame, *parameters)


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

        self._components = None

    @property
    def components(self):
        """Flattened tuple of all components in the tree.

        CAUTION: Each component in a tree can only be a leaf of a single node.
        While one can construct trees that hold the same component multiple
        times, this method will only return that component at its first
        encountered location
        """
        if self._components is None:
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
            self._components = tuple(components)
        return self._components

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
        pars = []
        for c in self.components:
            pars += c.parameters
        return pars

    def get_model(self, *params):
        """Get the model this component tree

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

    def get_flux(self):
        """Get the total flux for all the components in the tree
        """
        for k, component in enumerate(self.components):
            if k == 0:
                model = component.get_flux()
            else:
                model += component.get_flux()
        return model

    def update(self):
        """Update each component

        This method may be overwritten in inherited classes to
        perform updates on multiple components at once
        (for example separating a buldge and disk).
        """
        for node in self._tree:
            node.update()

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
        self._components = None
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

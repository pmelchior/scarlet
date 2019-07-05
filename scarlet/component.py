import autograd.numpy as np
from autograd.numpy.numpy_boxes import ArrayBox
from autograd.core import VSpace
import logging

logger = logging.getLogger("scarlet.component")

class Parameter(np.ndarray):
    def __new__(cls, array, step=0, converged=None, name="", **kwargs):
        obj = np.asarray(array, dtype=array.dtype).view(cls)
        obj.step = step
        obj.converged = converged
        obj.name = name
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        self.step = getattr(obj, 'step', 0)
        self.converged = getattr(obj, 'converged', None)
        self.name = getattr(obj, 'name', "")

ArrayBox.register(Parameter)
VSpace.register(Parameter, vspace_maker=VSpace.mappings[np.ndarray])

class Prior():
    """Differentiable Prior

    Attributes
    ----------
    grad_func: `function`
        Function used to create the gradient
    L_func: `function`
        Function used to calculate the Lipschitz constant
        of the gradient.
    """

    def __init__(self, grad_func, L_func):
        self._grad_func = grad_func
        self._L_func = L_func
        self.sed_grad = 0
        self.morph_grad = 0

    # can be overloaded but needs to set these 4 members
    # for batch processing: cache results with c as key
    def compute_grad(self, component):
        """Calculate the gradient

        This method is called by the `Component` that owns
        it during fitting to calculate the gradient update
        for the component due to the prior.
        """
        self.sed_grad, self.morph_grad = self._grad_func(component._sed, component._morph)
        self.L_sed, self.L_morph = self._L_func(component._sed, component._morph)


class Component():
    """A single component in a blend.

    This class acts as base for building complex :class:`scarlet.source.Source`.

    Parameters
    ----------
    frame: a `~scarlet.Frame` instance
        The spectral and spatial characteristics of this component.
    sed: array
        1D array (bands) of the initial SED.
    morph: array
        Data cube (Height, Width) of the initial morphology.
    prior: list of `~scarlet.component.Prior`s
        Prior that generates gradients for the component.
    fix_sed: bool, default=`False`
        Whether or not the SED is fixed, or can be updated
    fix_morph: bool, default=`False`
        Whether or not the morphology is fixed, or can be updated
    """

    def __init__(self, frame, sed, morph, prior=None, fix_sed=False, fix_morph=False):
        self._frame = frame

        # set sed and morph
        self._sed = Parameter(np.array(sed))
        self._morph = Parameter(np.array(morph))

        self._prior = prior

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
    def sed(self):
        """Numpy view of the component SED
        """
        if isinstance(self._sed, Parameter):
            return self._sed.view(np.ndarray)
        return self._sed

    @property
    def morph(self):
        """Numpy view of the component morphology
        """
        if isinstance(self._morph, Parameter):
            return self._morph.view(np.ndarray)
        return self._morph

    @property
    def parameters(self):
        return self._sed, self._morph

    def get_model(self, *params):
        """Get the model for this component.

        Parameters
        ----------
        params: tuple of optimimzation parameters

        Returns
        -------
        model: array
            (Bands, Height, Width) image of the model
        """
        if len(params):
            sed, morph = params
            return sed[:, None, None] * morph[None, :, :]

        return self.sed[:, None, None] * self.morph[None, :, :]

    def get_flux(self):
        """Get flux in every band
        """
        return self.morph.sum() * self.sed

    def update(self):
        """Update the component

        This method can be overwritten in inherited classes to
        run proximal operators or other component update functions
        that will be executed during fitting.
        """
        return self


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

        This will be different than `self.components` because sources can
        have multiple components.

        Returns
        -------
        The arguments of `__init__`
        """
        return self._tree

    @property
    def n_nodes(self):
        """Number of direct attached nodes.
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
        for component in self.components:
            component.update()

    def __iadd__(self, c):
        """Add another component or tree.

        Parameters
        ----------
        c: `~scarlet.component.Component` or `~scarlet.component.ComponentTree`
        """
        c_index = self.n_nodes
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

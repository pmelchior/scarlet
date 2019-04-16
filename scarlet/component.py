try:
    from enum import Flag, auto
except ImportError:
    from aenum import Flag, auto

import numpy as np
import torch

import logging
logger = logging.getLogger("scarlet.component")


class BlendFlag(Flag):
    """Flags that can be set by scarlet

    Attributes
    ----------
    NONE:
        There are no flags for this object.
    SED_NOT_CONVERGED:
        The SED has not yet converged.
    MORPH_NOT_CONVERGED:
        The morphology has not yet converged.
    EDGE_PIXELS:
        There is flux at the edge of the model,
        meaning the shape and intensity of the source
        might be incorrect.
    NO_VALID_PIXELS:
        All of the pixels of the source were below
        the detection threshold.
    """
    NONE = 0
    SED_NOT_CONVERGED = auto()
    MORPH_NOT_CONVERGED = auto()
    EDGE_PIXELS = auto()
    NO_VALID_PIXELS = auto()


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

    # can be overloaded but needs to set these 4 members
    # for batch processing: cache results with c as key
    def compute_grad(self, component):
        """Calculate the gradient

        This method is called by the `Component` that owns
        it during fitting to calculate the gradient update
        for the component due to the prior.
        """
        self.grad_morph, self.grad_sed = self._grad_func(component._morph, component._sed)
        self.L_morph, self.L_sed = self._L_func(component._morph, component._sed)


class Component():
    """A single component in a blend.

    This class acts as base for building complex :class:`scarlet.source.Source`.

    Parameters
    ----------
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
    def __init__(self, sed, morph, prior=None, fix_sed=False, fix_morph=False):
        self.B = sed.shape[0]
        self.Ny, self.Nx = morph.shape
        # set sed and morph
        sed = torch.Tensor(sed)
        morph = torch.Tensor(morph)
        self._sed = sed.detach().clone()
        self._morph = morph.detach().clone()
        self._sed.requires_grad_(not fix_sed)
        self._morph.requires_grad_(not fix_morph)
        self.prior = prior
        # Initially the component has not converged
        self.flags = BlendFlag.SED_NOT_CONVERGED | BlendFlag.MORPH_NOT_CONVERGED
        # Store the SED and morphology from the previous iteration
        self._last_sed = np.zeros_like(sed.detach().numpy())
        self._last_morph = np.zeros_like(morph.detach().numpy())

        # Properties used for indexing in the ComponentTree
        self._index = None
        self._parent = None

    @property
    def shape(self):
        """Shape of the image (Band, Height, Width)
        """
        return (self.B, self.Ny, self.Nx)

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
    def sed(self):
        """Numpy view of the component SED
        """
        return self._sed.data.detach().numpy()

    @property
    def morph(self):
        """Numpy view of the component morphology
        """
        return self._morph.data.detach().numpy()

    def get_model(self, numpy=True):
        """Get the model for this component.

        Parameters
        ----------
        numpy: bool
            Whether to return the model as a numpy array
            (`numpy=True`) or a `torch.tensor`.

        Returns
        -------
        model: array or tensor
            (Bands, Height, Width) image of the model
        """
        if numpy:
            return self.sed[:, None, None] * self.morph[None, :, :]
        return self._sed[:, None, None] * self._morph[None, :, :]

    def get_flux(self):
        """Get flux in every band
        """
        return self.morph.sum() * self.sed

    def backward_prior(self):
        """Use the prior to update the gradient
        """
        if self.prior is not None:
            self.prior.compute_grad(self)
            if self.morph.requires_grad:
                self.morph.grad += self.prior.grad_morph
                self.L_morph += self.prior.L_morph
            if self.sed.requires_grad:
                self.morph.grad += self.prior.grad_morph
                self.L_morph += self.prior.L_morph

    def update(self):
        """Update the component

        This method can be overwritten in inherited classes to
        run proximal operators or other component update functions
        that will be executed during fitting.
        """
        return self

    @property
    def step_morph(self):
        try:
            return 1 / self.L_morph
        except AttributeError:
            return None

    @property
    def step_sed(self):
        try:
            return 1 / self.L_sed
        except AttributeError:
            return None


class ComponentTree():
    """Base class for hierarchical collections of `~scarlet.component.Component`s.
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
    def B(self):
        """Number of bands
        """
        return self.components[0].B

    @property
    def nodes(self):
        """Initial list that generates the tree.

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

    def get_model(self, numpy=True):
        """Get the model this component tree

        Parameters
        ----------
        numpy: bool
            Whether to return the model as a numpy array
            (`numpy=True`) or a `torch.tensor`.

        Returns
        -------
        model: `~torch.tensor`
            (Bands, Height, Width) data cube
        """
        for k, component in enumerate(self.components):
            if k == 0:
                model = component.get_model(numpy)
            else:
                model += component.get_model(numpy)
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

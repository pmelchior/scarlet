from enum import Flag, auto

import torch

import logging
logger = logging.getLogger("scarlet.component")


class BlendFlag(Flag):
    NONE = 0
    SED_NOT_CONVERGED = auto()
    MORPH_NOT_CONVERGED = auto()
    EDGE_PIXELS = auto()
    CONVERSION_FAILED = auto()
    NO_VALID_PIXELS = auto()
    SATURATED = auto()
    LOW_FLUX = auto()


class Prior():
    def __init__(self, grad, L):
        self._grad_func = grad
        self._L_func = L

    # can be overloaded but needs to set these 4 members
    # for batch processing: cache results with c as key
    def compute_grad(self, c):
        self.grad_morph, self.grad_sed = self._grad_func(c.morph, c.sed)
        self.L_morph, self.L_sed = self._L_func(c.morph, c.sed)


class Component(object):
    """A single component in a blend.

    This class acts as base for building complex :class:`scarlet.source.Source`.
    """
    def __init__(self, sed, morph, priors=None, fix_sed=False, fix_morph=False, bboxes=None,
                 flags=None):
        """Constructor

        Create component from a SED vector and morphology image.

        Parameters
        ----------
        sed: array
            1D array (bands) of the initial SED.
        morph: array
            Data cube (Height, Width) of the initial morphology.
        constraints: :class:`scarlet.constraint.Constraint` or list thereof
            Constraints used to constrain the SED and/or morphology.
            When `constraints` is `None` then
            :class:`scarlet.constraint.MinimalConstraint` is used.
        fix_sed: bool, default=`False`
            Whether or not the SED is fixed, or can be updated
        fix_morph: bool, default=`False`
            Whether or not the morphology is fixed, or can be updated
        """
        # set sed and morph
        self.B = sed.size()[0]
        self.Ny, self.Nx = morph.shape
        if isinstance(sed, torch.Tensor):
            self._sed = sed.detach().clone()
            self._sed.requires_grad_(not fix_sed)
        else:
            self._sed = torch.tensor(morph, requires_grad=not fix_sed)
        if isinstance(morph, torch.Tensor):
            self._morph = morph.detach().clone()
            self._morph.requires_grad_(not fix_morph)
        else:
            self._morph = torch.tensor(morph, requires_grad=not fix_morph)

        self.priors = priors

        if bboxes is None:
            bboxes = {}
        self._bboxes = bboxes
        if flags is None:
            flags = BlendFlag.SED_NOT_CONVERGED | BlendFlag.MORPH_NOT_CONVERGED
        self.flags = flags
        self._last_sed = torch.zeros_like(sed)
        self._last_morph = torch.zeros_like(morph)

        # for ComponentTree
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
        return self._sed.data

    @property
    def morph(self):
        return self._morph.data

    @property
    def bboxes(self):
        return self._bboxes

    def get_model(self, bbox=None):
        """Get the model this component.

        Parameters
        ----------
        None

        Returns
        -------
        model: `~numpy.array`
            (Bands, Height, Width) image of the model
        """
        model = self._sed[:, None, None] * self._morph[None, :, :]

        # Optionally extract the model in a given bounding box
        if bbox is not None:
            if bbox in self.bboxes:
                bbox = self.bboxes[bbox]
            model = model[(slice(None), *bbox.slices)]
        return model

    def get_flux(self):
        """Get flux in every band
        """
        return self._morph.sum() * self._sed

    def backward_prior(self):
        if self.priors is not None:
            self.priors.compute_grad(self)
            if self.morph.requires_grad:
                self.morph.grad += self.priors.grad_morph
                self.L_morph += self.priors.L_morph
            if self.sed.requires_grad:
                self.morph.grad += self.priors.grad_morph
                self.L_morph += self.priors.L_morph

    def update(self):
        return self


class ComponentTree(object):
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
    def bbox(self):
        """Union of all the component bounding boxes
        """
        bbox = self.components[0].bbox.copy()
        for c in self.components[1:]:
            bbox |= c.bbox
        return bbox

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

    def get_model(self):
        """Get the model this component tree
         Returns
        -------
        model: `~torch.tensor`
            (Bands, Height, Width) data cube
        """
        for k, component in enumerate(self.components):
            if k == 0:
                model = component.get_model()
            else:
                model += component.get_model()
        return model

    def get_flux(self):
        for k, component in enumerate(self.components):
            if k == 0:
                model = component.get_flux()
            else:
                model += component.get_flux()
        return model

    def update(self):
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

import torch
import numpy as np

from . import constraint as sc
from .convolution import fft_convolve
from .utils import BoundingBox, resize

import logging
logger = logging.getLogger("scarlet.component")


class Component(object):
    """A single component in a blend.

    This class acts as base for building complex :class:`scarlet.source.Source`.
    """
    def __init__(self, sed, morph, constraints=None, fix_sed=False, fix_morph=False, config=None,
                 center=None, bbox=None):
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
        config: :class:`scarlet.config.Config`
            The configuration parameters of the entire blend.
        """
        # set sed and morph
        self.B = sed.size()[0]
        self.Ny, self.Nx = morph.shape
        if isinstance(sed, torch.Tensor):
            self._sed = sed.clone()
            self._sed.requires_grad_(not fix_sed)
        else:
            self._sed = torch.tensor(morph, requires_grad=not fix_sed)
        if isinstance(morph, torch.Tensor):
            self._morph = morph.clone()
            self._morph.requires_grad_(not fix_morph)
        else:
            self._morph = torch.tensor(morph, requires_grad=not fix_morph)

        center = np.unravel_index(np.argmax(self.morph), self.morph.shape)
        self.update_center(center)

        # updates for frame, sed, morph?
        self.fix_sed = fix_sed
        self.fix_morph = fix_morph

        self.set_constraints(constraints)

        # for ComponentTree
        self._index = None
        self._parent = None

        self._center = center
        if bbox is None:
            bbox = BoundingBox()
        self._bbox = bbox
        self._model = None
        self.model_step = -1

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
    def center(self):
        return self._center

    @property
    def bbox(self):
        return self._bbox

    def set_constraints(self, constraints):
        """Set constraints for component.

        Parameters
        ----------
        constraints: None, constraint or array-like, default=none
            Constraints can be either `~scarlet.constraints.Constraint` or
            `~scarlet.constraints.ConstraintList`. If an array is given, it is
            one constraint per component.

        Returns
        -------
        None
        """

        if constraints is None:
            constraints = sc.MinimalConstraint()

        self.constraints = sc.ConstraintAdapter(constraints, self)

    def get_model(self, psfs=None, padding=3, trim=True):
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
        if psfs is not None:
            model = torch.stack([
                fft_convolve(model[b], psfs[b], padding=padding)
                for b in range(self.B)
            ])
        if trim:
            model = model[(slice(None), *self.bbox.slices)]
        return model

    def update_center(self, center):
        """Given a (y,x) `center`, update the frame and `Gamma`
        """
        assert len(center) == 2
        self._center = tuple(center)

    def update_bbox(self, bbox=None, min_value=None):
        """Update the bounding box

        If `bbox` is not `None` then the bounding boxe is set to `bbox`.
        Otherwise `min_value` must be specified and the component is
        checked to see if it has any flux on its edge larger than `min_value`.
        If it does then the bounding box is resized and trimmed.

        Parameters
        ----------
        bbox: `BoundingBox`
            The new bounding box.
        `min_value`: float
            Minimum value in the component to avoid being trimmed.
        """
        if bbox is not None:
            self._bbox = bbox
        elif min_value is not None:
            self._bbox = resize(self.get_model(trim=False), self.bbox, min_value)
        else:
            msg = "Either `bbox` or `min_value` must be set to update the bounding boxes"
            raise ValueError(msg)

    def get_flux(self):
        """Get flux in every band
        """
        return self._morph.sum() * self._sed

    def _normalize(self, normalization):
        """Apply normalization to SED & Morphology, assuming an initial Smax norm

        Parameters
        ----------
        normalization: `~scarlet.constraint.Normalization`
        """
        if normalization == sc.Normalization.A:
            norm = self.sed.sum()
            self.sed[:] = self.sed / norm
            self.morph[:] = self.morph / norm
        elif normalization == sc.Normalization.S:
            norm = self.morph.sum()
            self.morph[:] = self.morph / norm
            self.sed[:] = self.sed / norm


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

    def _update(self, update_type):
        """Recursively update the tree"""
        for c in self._tree:
            if hasattr(c, update_type):
                getattr(c, update_type)()

    def update_center(self):
        """Update the center location of attached nodes.

        This methods recursively call the same function of all attached tree nodes.
        """
        self._update("update_center")

    def update_sed(self):
        """Update the SEDs of attached nodes.

        This methods recursively call the same function of all attached tree nodes.
        While the method has complete freedom to perform updates, it is
        recommended that its behavior mimics a proximal operator in the direct domain.
        """
        self._update("update_sed")

    def update_morph(self):
        """Update the morphologies of attached nodes.

        This methods recursively call the same function of all attached tree nodes.
        While the method has complete freedom to perform updates, it is
        recommended that its behavior mimics a proximal operator in the direct domain.
        """
        self._update("update_morph")

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

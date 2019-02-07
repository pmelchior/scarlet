import numpy as np
from . import constraint as sc

import logging
logger = logging.getLogger("scarlet.component")

class Component(object):
    """A single component in a blend.

    This class acts as base for building complex :class:`scarlet.source.Source`.
    """
    def __init__(self, sed, morph, center=None, constraints=None, psf=None, fix_sed=False,
                 fix_morph=False, fix_frame=False, shift_center=0.2):
        """Constructor

        Create component from a SED vector and morphology image.

        Parameters
        ----------
        sed: array
            1D array (bands) of the initial SED.
        morph: array
            Data cube (Height, Width) of the initial morphology.
        center: array-like
            (y,x) coordinates of the component in the larger image
        constraints: :class:`scarlet.constraint.Constraint` or list thereof
            Constraints used to constrain the SED and/or morphology.
            When `constraints` is `None` then
            :class:`scarlet.constraint.MinimalConstraint` is used.
        psf: array-like or `~scarlet.transformation.Gamma`, default=`None`
            2D image of the psf in a single band (Height, Width),
            or 2D image of the psf in each band (Bands, Height, Width),
            or `~scarlet.transformation.Gamma` created from a psf array.
        fix_sed: bool, default=`False`
            Whether or not the SED is fixed, or can be updated
        fix_morph: bool, default=`False`
            Whether or not the morphology is fixed, or can be updated
        fix_frame: bool, default=`False`
            Whether or not the frame dimensions are fixed, or can be updated
        shift_center: float, default=0.2
            Amount to shift the differential image in x and y to fit
            changes in position.
        """
        # set sed and morph
        self.B = sed.size
        self.sed = sed.copy()

        # check that morph has odd dimensions
        assert len(morph.shape) == 2
        Ny, Nx = morph.shape
        if all([morph.shape[i] % 2 == 1 for i in range(2)]):
            self.morph = morph.copy()
        else:
            _Ny, _Nx = Ny, Nx
            if _Ny % 2 == 0:
                _Ny += 1
            if _Nx % 2 ==0:
                _Nx += 1
            self.morph = np.zeros((_Ny, _Nx))
            self.morph[:Ny,:Nx] = morph[:,:]

        # set up psf and translations matrices
        from . import transformation
        if isinstance(psf, transformation.Gamma):
            self._gamma = psf
        else:
            if psf is not None and len(psf.shape)==2:
                psf = np.array([psf]*self.B)
            self._gamma = transformation.Gamma(psfs=psf)

        # set center coordinates and translation operators
        # needs to have Gamma set up first
        if center is None:
            center = (morph.shape[0] // 2, morph.shape[1] // 2)
        self.set_center(center)
        self.set_frame()
        self.shift_center = shift_center

        # updates for frame, sed, morph?
        self.fix_frame = fix_frame
        self.fix_sed = fix_sed
        self.fix_morph = fix_morph

        self.set_constraints(constraints)

        # for ComponentTree
        self._index = None
        self._parent = None

    @property
    def Nx(self):
        """Width of the frame
        """
        return self.right-self.left

    @property
    def Ny(self):
        """Height of the frame
        """
        return self.top - self.bottom

    @property
    def shape(self):
        """Shape of the image (Band, Height, Width)
        """
        return (self.B, self.Ny, self.Nx)

    @property
    def bb(self):
        # TODO: docstring
        # since slice wrap around if start or stop are negative, need to sanitize
        # start values (stop always postive)
        return (slice(None), slice(max(0, self.bottom), self.top), slice(max(0, self.left), self.right))

    @property
    def center_int(self):
        """Rounded (not truncated) integer pixel position of the center
        """
        return Component.get_int(self.center)

    @property
    def has_psf(self):
        """Whether the component has a psf
        """
        return self._gamma.psfs is not None

    @property
    def coord(self):
        """The coordinate in a `~scarlet.component.ComponentTree`.
        """
        if self._index is not None:
            if self._parent._index is not None:
                return tuple(self._parent.coord) + (self._index,)
            else:
                return (self._index,)

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

    def get_slice_for(self, im_shape):
        """Return the slice of the component frame in the full multiband image

        In other words, return the slice so that
        self.get_model()[slice] corresponds to image[self.bb],
        where image has shape (Band, Height, Width).

        Parameters
        ----------
        im_shape: tuple
            Shape of the full image
        """
        NY, NX = im_shape[1:]

        left = max(0, -self.left)
        bottom = max(0, -self.bottom)
        right = self.Nx - max(0, self.right - NX)
        top = self.Ny - max(0, self.top - NY)
        return (slice(None), slice(bottom, top), slice(left, right))

    def get_model(self, Gamma=None, use_sed=True):
        """Get the model this component.

        Parameters
        ----------
        Gamma: `~scarlet.transformation.Gamma`, default=`None`
            Gamma transformation to convolve with PSF and perform linear transform.
            If `Gamma` is `None` then `self.Gamma` is used.
        use_sed: bool, default=`True`
            Whether to use the SED to create a multi-color model or model of just the morphology

        Returns
        -------
        model: `~numpy.array`
            (Bands, Height, Width) image of the model
        """
        if Gamma is None:
            Gamma = self.Gamma
        if use_sed:
            sed = self.sed
        else:
            sed = np.ones_like(self.sed)

        if not self.has_psf:
            model = np.empty((self.B, self.Ny, self.Nx))
            model = np.outer(sed, Gamma.dot(self.morph)).reshape(self.B, self.Ny, self.Nx)
        else:
            model = np.zeros((self.B, self.Ny, self.Nx))
            for b in range(self.B):
                model[b] += sed[b] * Gamma[b].dot(self.morph)

        return model

    @staticmethod
    def get_int(x):
        """Return rounded integer version of argument.
        """
        return np.round(x).astype('int')

    @staticmethod
    def get_frame(shape, center, new_shape):
        """Create a frame and bounding box

        To save memory and computation time, each component is contained in a small
        subset of the entire blended image. This method takes the coordinates of
        the component and the size of the frame and creates a bonding box (`self.bb`).

        Sets `self.bottom`, `self.top`, `self.left`, `self.right` as
        the edges of the frame, `self.bb` as the slices (bounding box)
        containing the frame, and `self.center` as the center of the frame.

        Parameters
        ----------
        center: array-like
            (y,x) coordinates of the center of the component in the full image
        size: float or array-like
            Either a (height,width) shape or a single size to create a
            square (size,size) frame.

        Returns
        -------
        old_slice, new_slice: to map subsections of `self.morph` from the old to
        the new shape.

        """
        # store old edge coordinates
        (top, right), (bottom, left) = shape, [0,] * 2

        # ensure odd pixel number
        y, x = Component.get_int(center)
        _bottom, _top = y - int(new_shape[0]//2), y + int(new_shape[0]//2) + 1
        _left, _right = x - int(new_shape[1]//2), x + int(new_shape[1]//2) + 1

        # slices to update _morph: check if new size is larger or smaller
        new_slice_y = slice(max(0, bottom - _bottom),
                            min(_top - _bottom, _top - _bottom - (_top - top)))
        old_slice_y = slice(max(0, _bottom - bottom), min(top - bottom, top - bottom - (top - _top)))
        new_slice_x = slice(max(0, left - _left),
                            min(_right - _left, _right - _left - (_right - right)))
        old_slice_x = slice(max(0, _left - left), min(right - left, right - left - (right - _right)))
        new_slice = (new_slice_y, new_slice_x)
        old_slice = (old_slice_y, old_slice_x)
        return old_slice, new_slice

    def set_center(self, center):
        """Given a (y,x) `center`, update the frame and `Gamma`
        """
        assert len(center) == 2
        self.center = np.array(center)

        # TODO: check if needed
        # frame update for tracking moving centers
        self.set_frame()

        # update translation operator
        dyx = self.center - self.center_int
        self.Gamma = self._gamma(dyx)

    def set_frame(self):
        # ensure odd pixel number
        y, x = Component.get_int(self.center)
        shape = self.morph.shape
        self.bottom, self.top = y - int(shape[0]//2), y + int(shape[0]//2) + 1
        self.left, self.right = x - int(shape[1]//2), x + int(shape[1]//2) + 1

    def resize(self, size):
        """Resize the frame

        Set the new frame size and update relevant parameters like the morphology,
        Gamma matrices, and constraint operators.

        Parameters
        ----------
        size: float or array-like
            Either a (height,width) shape or a single size to create a
            square (size,size) frame.
        """
        if hasattr(size, '__iter__'):
            size = size[:2]
        else:
            size = (size,) * 2

        morph_center = self.center - np.array([self.bottom, self.left])
        old_slice, new_slice = Component.get_frame(self.morph.shape, morph_center, size)
        if new_slice != old_slice:
            # change morph
            _morph = self.morph.copy()
            self.morph.resize(size, refcheck=False)
            self.morph[:,:] = 0
            self.morph[new_slice] = _morph[old_slice]
            self.set_frame()

    def normalize(self, normalization):
        """Apply normalization to SED & Morphology

        Parameters
        ----------
        normalization: `~scarlet.constraint.Normalization`
        """
        if normalization == sc.Normalization.A:
            norm = self.sed.sum()
            self.sed /= norm
            self.morph *= norm
        else:
            if normalization == sc.Normalization.S:
                norm = self.morph.sum()
            elif normalization == sc.Normalization.Smax:
                norm = self.morph.max()
            self.morph /= norm
            self.sed *= norm

    def get_flux(self):
        """Get flux in every band
        """
        return self.morph.sum() * self.sed

    def get_morph_error(self, weights):
        """Get error in the morphology

        This error estimate uses linear error propagation and assumes that the
        component was isolated (it ignores blending).

        CAVEAT: If the component has a PSF, the inversion of the covariance matrix
        is likely unstable.

        Parameters
        ----------
        weights: `~numpy.array`
            Weights of the images in each band (Bands, Height, Width).

        Returns
        -------
        me: `~numpy.array`
            Error in morphology for each pixel
        """
        w = np.zeros(self.shape)
        w[self.get_slice_for(weights.shape)] = weights[self.bb]
        w = w.reshape(self.B, -1)
        # prevent zeros from messing up:
        # set them at a very small value, and zero them out at the end
        mask = (w.sum(axis=0) == 0).flatten()
        if mask.sum():
            w[:,mask] = 1e-3 * w[:,~mask].min(axis=1)[:,None]

        # compute direct error propagation assuming only this component SED(s)
        # and the pixel covariances: Sigma_morph = diag((A^T Sigma^-1 A)^-1)
        # CAVEAT: If done on the entire A matrix, degeneracies in the linear
        # solution arise and substantially amplify the error estimate:
        # Instead, estimate noise for each component separately:
        # simple multiplication for diagonal pixel covariance matrix
        if not self.has_psf:
            me = 1./np.sqrt(np.dot(self.sed.T, np.multiply(w, self.sed[:,None])))
        else:
            # see Blend.steps_f for details for the complete covariance matrix
            import scipy.sparse
            Sigma_pix = scipy.sparse.diags(w.flatten(), 0)
            PA = scipy.sparse.bmat([self.sed[b] * self.Gamma[b] for b in range(self.B)])
            Sigma_s = PA.T.dot(Sigma_pix.dot(PA))
            me = np.sqrt(np.diag(np.linalg.inv(Sigma_s.toarray())))

            # TODO: the matrix inversion is instable if the PSF gets wide
            # possible options: Tikhonov regularization or similar
        if mask.sum():
            me[mask] = 0
        return me

    def get_sed_error(self, weights):
        """Get error in the SED's

        This error estimate uses linear error propagation and assumes that the
        component was isolated (it ignores blending).

        Parameters
        ----------
        weights: `~numpy.array`
            Weights of the images in each band (Bands, Height, Width).

        Returns
        -------
        error: `~numpy.array`
            Estimated error in the SED.
        """
        w = np.zeros(self.shape)
        w[self.get_slice_for(weights.shape)] = weights[self.bb]
        w = w.reshape(self.B, -1)
        # NOTE: zeros weights would only be a problem if an entire band is missing

        # See explanation in get_morph_error and Blend.steps_f
        if not self.has_psf:
            return 1./np.sqrt(np.dot(self.morph,np.multiply(w.T, self.morph[None,:].T)))
        else:
            import scipy.sparse
            Sigma_pix = scipy.sparse.diags(w.flatten(), 0)
            model = self.get_model(combine=False, use_sed=False)
            PS = scipy.sparse.block_diag([model[b,:,:].reshape((1,-1)).T for b in range(self.B)])
            return np.sqrt(np.diag(np.linalg.inv(PS.T.dot(Sigma_pix.dot(PS)).toarray())))


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
        for i,c in enumerate(self._tree):
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

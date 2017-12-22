from __future__ import print_function, division
import numpy as np
from functools import partial
from enum import IntFlag

import proxmin
from . import transformations
from . import operators

import logging
logger = logging.getLogger("scarlet.source")

# When we drop python2 support we can use the following code
try:
    from enum import IntFlag

    class InitMethod(IntFlag):
        PEAK = 1 # Use the value at the peak
        SYMMETRIC = 2 # Use a symmetric template
        MONOTONIC = 4 # Use a monotonic template
        MONOSYM = 6 # Use a monotonic and symmetric template
# Until then use a python 2 version
except ImportError:
    class InitMethod:
        """Mock Enum
        """
        PEAK = 1 # Use the value at the peak
        SYMMETRIC = 2 # Use a symmetric template
        MONOTONIC = 4 # Use a monotonic template
        MONOSYM = 6 # Use a monotonic and symmetric template

class Source(object):
    """A single source in a blend
    """
    def __init__(self, center, shape, K=1, psf=None, constraints=None, fix_sed=False, fix_morph=False,
                 fix_frame=False, shift_center=0.2):
        """Constructor

        Parameters
        ----------
        center: array-like
            (y,x) coordinates of the source in the larger image
        shape: tuple
            Shape (B, Ny, Nx) of the frame that contains the source.
            This can be (and usually is) smaller than the size of the full blend
        K: int, default='1'
            Number of components with the same center position
        psf: array-like or `~scarlet.transformations.GammaOp`, default=`None`
            2D image of the psf in a single band (Height, Width),
            or 2D image of the psf in each band (Bands, Height, Width),
            or `~scarlet.transformations.GammaOp` created from a psf array.
        constraints: dict or list of dicts, default=`None`
            Each key in `constraints` contains any parameters
            (such as a treshold for "l0") needed by the proximal operator.
            If K>1, a list of dicts can be given, one for each component.
        fix_sed: bool or list of bools, default=`False`
            Whether or not the SED is fixed, or can be updated
            If K>1, a list of bools can be given, one for each component.
        fix_morph: bool or list of bools, default=`False`
            Whether or not the morphology is fixed, or can be updated
            If K>1, a list of bools can be given, one for each component.
        fix_frame: bool or list of bools, default=`False`
            Whether or not the frame dimensions are fixed, or can be updated
            If K>1, a list of bools can be given, one for each component.
        shift_center: float, default=0.2
            Amount to shift the differential image in x and y to fit
            changes in position.
        """

        # set size of the source frame
        assert len(shape) == 3
        self.B = shape[0]
        size = shape[1:]
        self._set_frame(center, size)
        size = (self.Ny, self.Nx)
        self.fix_frame = fix_frame

        # create containers
        self.K = K
        self.sed = np.zeros((self.K, self.B))
        self.morph = np.zeros((self.K, self.Ny*self.Nx))

        # set up psf and translations matrices
        if isinstance(psf, transformations.GammaOp):
            self._gammaOp = psf
        else:
            self._gammaOp = transformations.GammaOp(self.shape, psf=psf)

        # set center coordinates and translation operators
        # needs to have GammaOp set up first
        self.set_center(center)
        self.shift_center = shift_center

        # updates for sed or morph?
        if hasattr(fix_sed, '__iter__') and len(fix_sed) == self.K:
            self.fix_sed = fix_sed
        else:
            self.fix_sed = [fix_sed] * self.K
        if hasattr(fix_morph, '__iter__') and len(fix_morph) == self.K:
            self.fix_morph = fix_morph
        else:
            self.fix_morph = [fix_morph] * self.K

        # set sed and morph constraints
        self.set_constraints(constraints)

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
        """Shape of the source image (Band, Height, Width)
        """
        return (self.B, self.Ny, self.Nx)

    @property
    def image(self):
        """Reshaped morphology into an array of 2D images for each component
        """
        morph_shape = (self.K, self.Ny, self.Nx)
        return self.morph.reshape(morph_shape) # this *should* be a view

    @property
    def center_int(self):
        """Rounded (not truncated) integer pixel position of the center
        """
        return np.round(self.center).astype('int')

    @property
    def has_psf(self):
        """Whether the source has a psf
        """
        return self._gammaOp.psf is not None

    def get_slice_for(self, im_shape):
        """Return the slice of the source frame in the full multiband image

        In other words, return the slice so that
        self.image[k][slice] corresponds to image[self.bb],
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

    def get_model(self, combine=True, Gamma=None, use_sed=True):
        """Get the model of all components for the current source

        Parameters
        ----------
        combine: bool, default=`True`
            Whether or not to combine all of the components into a single model
        Gamma: `~scarlet.transformations.GammaOp`, default=`None`
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
        # model for all components of this source
        if not self.has_psf:
            model = np.empty((self.K, self.B, self.Ny*self.Nx))
            for k in range(self.K):
                model[k] = np.outer(sed[k], Gamma.dot(self.morph[k]))
        else:
            model = np.zeros((self.K, self.B, self.Ny*self.Nx))
            for k in range(self.K):
                for b in range(self.B):
                    model[k,b] += sed[k,b] * Gamma[b].dot(self.morph[k])

        # reshape the image into a 2D array
        model = model.reshape(self.K, self.B, self.Ny, self.Nx)

        if combine:
            model = model.sum(axis=0)
        return model

    def _set_frame(self, center, size):
        """Create a frame and bounding box

        To save memory and computation time, each source is contained in a small
        subset of the entire blended image. This method takes the coordinates of
        the source and the size of the frame and creates a bonding box (`self.bb`).

        Parameters
        ----------
        center: array-like
            (y,x) coordinates of the center of the source in the full image
        size: float or array-like
            Either a (height,width) shape or a single size to create a
            square (size,size) frame.

        Returns
        -------
        None.
        But it defines `self.bottom`, `self.top`, `self.left`, `self.right` as
        the edges of the frame, `self.bb` as the slices (bounding box) containing the frame,
        and `self.center` as the center of the frame.

        """
        assert len(center) == 2
        self.center = np.array(center)
        if hasattr(size, '__iter__'):
            size = size[:2]
        else:
            size = (size,) * 2
        # make cutout of in units of the original image frame (that defines xy)
        # ensure odd pixel number
        y_, x_ = self.center_int
        self.bottom, self.top = y_ - int(size[0]//2), y_ + int(size[0]//2) + 1
        self.left, self.right = x_ - int(size[1]//2), x_ + int(size[1]//2) + 1

        # since slice wrap around if start or stop are negative, need to sanitize
        # start values (stop always postive)
        self.bb = (slice(None), slice(max(0, self.bottom), self.top), slice(max(0, self.left), self.right))

    def set_center(self, center):
        """Given a (y,x) `center`, update the frame and `Gamma`
        """
        if not hasattr(self, '_init_center'):
            self._init_center = np.array([center[0], center[1]])
        size = (self.Ny, self.Nx)
        self._set_frame(center, size)

        # update translation operator
        dx = self.center - self.center_int
        self.Gamma = self._gammaOp(dx, self.shape)

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
        # store old edge coordinates
        top, right, bottom, left = self.top, self.right, self.bottom, self.left
        self._set_frame(self.center, size)

        # check if new size is larger or smaller
        new_slice_y = slice(max(0, bottom - self.bottom),
                            min(self.top - self.bottom, self.top - self.bottom - (self.top - top)))
        old_slice_y = slice(max(0, self.bottom - bottom), min(top - bottom, top - bottom - (top - self.top)))
        if top-bottom == self.Ny:
            new_slice_y = old_slice_y = slice(None)
        new_slice_x = slice(max(0, left - self.left),
                            min(self.right - self.left, self.right - self.left - (self.right - right)))
        old_slice_x = slice(max(0, self.left - left), min(right - left, right - left - (right - self.right)))
        if right-left == self.Nx:
            new_slice_x = old_slice_x = slice(None)
        new_slice = (slice(None), new_slice_y, new_slice_x)
        old_slice = (slice(None), old_slice_y, old_slice_x)

        if new_slice != old_slice:
            # change morph
            _morph = self.morph.copy().reshape((self.K, top-bottom, right-left))
            self.morph = np.zeros((self.K, self.Ny, self.Nx))

            self.morph[new_slice] = _morph[old_slice]
            self.morph = self.morph.reshape((self.K, self.Ny*self.Nx))

            # update GammaOp and center (including subpixel shifts)
            self.set_center(self.center)

            # set constraints
            self.set_constraints(self.constraints)

    def init_source(self, img, bg_rms, weights=None, init_method=InitMethod.MONOSYM):
        """Initialize the source

        Parameters
        ----------
        img: `~numpy.array`
            (Bands, Height, Width) data array that contains a 2D image for each band
        weights: `~numpy.array`
            (Bands, Height, Width) data array that contains a 2D weight image for each band
            Currently not implemented in initialization.
        init_method: InitMethods or None, default=`InitMethods.MONOSYM`
            Method to use for initialization. If `init_method` is `None` then
            the sources are not initialized.

        This default implementation initializes takes the sed from the pixel in
        the center of the frame and sets morphology to only comprise that pixel,
        which works well for point sources and poorly resolved galaxies.

        Returns
        -------
        None.
        But `self.sed` and `self.morph` are set.
        """
        # init with SED of the peak pixels
        # TODO: what should we do if peak is saturated?
        B = img.shape[0]
        self.sed = np.empty((self.K, B))
        _y, _x = self.center_int
        _sed = img[:,_y,_x]
        for k in range(self.K):
            self.sed[k] = _sed
            # ensure proper normalization
            self.sed[k] = proxmin.operators.prox_unity_plus(self.sed[k], 0)
            if self.K>1:
                epsilon = 1e-2
                self.sed[k] += np.random.rand(self.sed.shape[1])*epsilon
                # Normalize again
                self.sed[k] = proxmin.operators.prox_unity_plus(self.sed[k], 0)

        cx, cy = self.Nx // 2, self.Ny // 2
        if InitMethod.PEAK in init_method:
            self.morph = np.zeros((self.K, self.Ny, self.Nx))
            # Turn on a single pixel at the peak
            for k in range(self.K):
                # Make each component's radius one pixel larger
                self.morph[k, cy-k:cy+k+1,cx-k:cx+k+1] = img[:,_y-k:_y+k+1,_x-k:_x+k+1].sum(axis=0) + epsilon
            self.morph = self.morph.reshape(self.K, self.Ny*self.Nx)
        else:
            self.morph = np.zeros((self.K, self.Ny * self.Nx))
            Ny = 2*min(img.shape[1]-_y, _y)-1
            Nx = 2*min(img.shape[2]-_x, _x)-1
            cx, cy = Nx // 2, Ny // 2
            morph = np.zeros((Ny,Nx))
            # use the band with maximum flux for the source
            band = np.argmax(self.sed[0])
            morph[:] = img[band,_y-cy:_y+cy+1,_x-cx:_x+cx+1]
            morph = morph.reshape((morph.size,))
            morph[morph<0] = 0
            # For now, use a python 2 compatible version of an Enum
            #if InitMethod.SYMMETRIC in init_method:
            if InitMethod.SYMMETRIC & init_method:
                # Make the model symmetric
                symmetric = morph[::-1]
                morph = np.min([morph, symmetric], axis=0)
            #if InitMethod.MONOTONIC in init_method:
            if InitMethod.MONOTONIC & init_method:
                # Make the model monotonic
                prox_monotonic = operators.prox_strict_monotonic((Ny, Nx), thresh=0, use_nearest=False)
                morph = prox_monotonic(morph.reshape(morph.size,), 0)
            # Trim the source to set the new size
            morph = morph.reshape(Ny,Nx)
            ypix, xpix = np.where(morph>bg_rms[band]/2)
            if len(ypix)==0:
                ypix, xpix = np.where(morph>0)
            Ny = np.max(ypix)-np.min(ypix)
            Nx = np.max(xpix)-np.min(xpix)
            Ny += 1 - Ny % 2
            Nx += 1 - Nx % 2
            _cx = Nx//2
            _cy = Ny//2
            morph = morph[cy-_cy:cy+_cy+1, cx-_cx:cx+_cx+1]
            self.resize([Ny, Nx])
            for k in range(self.K):
                _morph = np.zeros_like(morph)
                if 4*k>Nx:
                    xrad = Nx//2-1
                else:
                    xrad = 2*k
                if 4*k>Ny:
                    yrad = Ny//2-1
                else:
                    yrad = 2*k
                _morph[yrad:Ny-yrad, xrad:Nx-xrad] = morph[yrad:Ny-yrad, xrad:Nx-xrad]
                _morph = _morph.reshape((1,_morph.size))
                self.morph[k] = _morph+np.random.rand(_morph.shape[0], _morph.shape[1])*np.max(_morph)/100

    def get_morph_error(self, weights):
        """Get error in the morphology

        This error estimate uses linear error propagation and assumes that the
        source was isolated (it ignores blending).

        CAVEAT: If the source has a PSF, the inversion of the covariance matrix
        is likely instable.

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

        # compute direct error propagation assuming only this source SED(s)
        # and the pixel covariances: Sigma_morph = diag((A^T Sigma^-1 A)^-1)
        # CAVEAT: If done on the entire A matrix, degeneracies in the linear
        # solution arise and substantially amplify the error estimate:
        # Instead, estimate noise for each component separately:
        # simple multiplication for diagonal pixel covariance matrix
        if not self.has_psf:
            me = [1./np.sqrt(np.dot(a.T, np.multiply(w, a[:,None]))) for a in self.sed]
        else:
            # see Blend.steps_f for details for the complete covariance matrix
            import scipy.sparse
            Sigma_pix = scipy.sparse.diags(w.flatten(), 0)
            PA = [scipy.sparse.bmat([[self.sed[k,b] * self.Gamma[b]] for b in range(self.B)])
                    for k in range(self.K)]
            Sigma_s = [PAk.T.dot(Sigma_pix.dot(PAk)) for PAk in PA]
            me = [np.sqrt(np.diag(np.linalg.inv(Sigma_sk.toarray()))) for Sigma_sk in Sigma_s]

            # TODO: the matrix inversion is instable if the PSF gets wide
            # possible options: Tikhonov regularization or similar
        if mask.sum():
            for mek in me:
                mek[mask] = 0
        return me

    def get_sed_error(self, weights):
        """Get error in the SED's

        This error estimate uses linear error propagation and assumes that the
        source was isolated (it ignores blending).

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
            return [1./np.sqrt(np.dot(s,np.multiply(w.T, s[None,:].T))) for s in self.morph]
        else:
            import scipy.sparse
            Sigma_pix = scipy.sparse.diags(w.flatten(), 0)
            model = self.get_model(combine=False, use_sed=False)
            PS = [scipy.sparse.block_diag([model[k,b,:,:].reshape((1,-1)).T for b in range(self.B)])
                        for k in range(self.K)]
            return [np.sqrt(np.diag(np.linalg.inv(PSk.T.dot(Sigma_pix.dot(PSk)).toarray()))) for PSk in PS]

    def set_constraints(self, constraints):
        """Set the constraints for each component in the source

        Parameters
        ----------
        constraints: dict, or list of dicts
            Each key in `constraints` contains any parameters
            (such as a treshold for "l0") needed by the proximal operator.
            If a list of dicts is given, each elements refers to one component.

        Returns
        -------
        None.
        But `self.constraints`, `self.progs_g` (ADMM-like proximal operators),
        `self.Ls` (linear matrices for each proxs_g),
        `self.prox_sed` (prox_f proximal operator for the SED's),
        and `self.prox_morph` (prox_f for morphologies) are set.
        """
        if constraints is None:
            self.constraints = {}
        else:
            self.constraints = constraints
        # one set of constraints for each component
        if not isinstance(self.constraints, list):
            self.constraints = [self.constraints,] * self.K

        self.proxs_g_A = [None] * self.K
        self.proxs_g_S = [[],] * self.K
        self.LA = [None] * self.K
        self.LS = [[],] * self.K

        if self.constraints is None:
            self.proxs_g_A = [None] * self.K
            self.proxs_g_S = [None] * self.K
            self.LA = [None] * self.K
            self.LS = [None] * self.K
            return

        self.prox_sed = [proxmin.operators.prox_unity_plus] * self.K
        self.prox_morph = [[],] * self.K

        # superset of all constraint keys (in case they are different)
        keys = set(self.constraints[0].keys())
        for k in range(1, self.K):
            keys |= set(self.constraints[k].keys())
        shape = (self.Ny, self.Nx)
        # because of the constraint matrices being costly to construct:
        # iterate over keys, not over components
        for c in keys:

            if c =="+":
                for k in range(self.K):
                    if c in self.constraints[k].keys():
                        self.prox_morph[k].append(proxmin.operators.prox_plus)

            # Note: don't use hard/soft thresholds with _plus (non-negative) because
            # that is either happening with prox_plus before or is not indended
            if c == "l0":
                for k in range(self.K):
                    if c in self.constraints[k].keys():
                        thresh = self.constraints[k][c]
                        self.prox_morph[k].append(partial(proxmin.operators.prox_hard, thresh=thresh))

            elif c == "l1":
                for k in range(self.K):
                    if c in self.constraints[k].keys():
                        thresh = self.constraints[k][c]
                        self.prox_morph[k].append(partial(proxmin.operators.prox_soft, thresh=thresh))

            elif c == "m":
                for k in range(self.K):
                    if c in self.constraints[k].keys():
                        kwargs = {}
                        if self.constraints[k][c] is not None:
                            kwargs.update(self.constraints[k][c])
                        self.prox_morph[k].append(operators.prox_strict_monotonic(shape, **kwargs))

            elif c == "M":
                # positive gradients
                # NOTE: we're using useNearest from component k=0 only!
                M = transformations.getRadialMonotonicOp(shape, useNearest=self.constraints[0][c])
                for k in range(self.K):
                    if c in self.constraints[k].keys():
                        self.LS[k].append(M)
                        self.proxs_g_S[k].append(proxmin.operators.prox_plus)
                    else:
                        self.LS[k].append(None)
                        self.proxs_g_S[k].append(None)

            elif c == "S":
                # zero deviation of mirrored pixels
                S = transformations.getSymmetryOp(shape)
                for k in range(self.K):
                    if c in self.constraints[k].keys():
                        self.LS[k].append(S)
                        self.proxs_g_S[k].append(proxmin.operators.prox_zero)
                    else:
                        self.LS[k].append(None)
                        self.proxs_g_S[k].append(None)

            elif c == "C":
                # cone method for monotonicity: exact but VERY slow
                useNearest = self.constraints.get("M", False)
                G = transformations.getRadialMonotonicOp(shape, useNearest=useNearest).toarray()
                for k in range(self.K):
                    if c in self.constraints[k].keys():
                        self.proxs_g_S[k].append(partial(operators.prox_cone, G=G))
                    else:
                        self.proxs_g_[k].append(None)
                    self.LS[k].append(None)
            elif c == "X":
                # l1 norm on gradient in X for TV_x
                cx = int(self.Nx)
                Gx = proxmin.transformations.get_gradient_x(shape, cx)
                for k in range(self.K):
                    if c in self.constraints[k].keys():
                        self.LS[k].append(Gx)
                        thresh = self.constraints[k][c]
                        self.proxs_g_S[k].append(partial(proxmin.operators.prox_soft, thresh=thresh))
                    else:
                        self.LS[k].append(None)
                        self.proxs_g_S[k].append(None)
            elif c == "Y":
                # l1 norm on gradient in Y for TV_y
                cy = int(self.Ny)
                Gy = proxmin.transformations.get_gradient_y(shape, cy)
                for k in range(self.K):
                    if c in self.constraints[k].keys():
                        self.LS[k].append(Gy)
                        thresh = self.constraints[k][c]
                        self.proxs_g_S[k].append(partial(proxmin.operators.prox_soft, thresh=thresh))
                    else:
                        self.LS[k].append(None)
                        self.proxs_g_S[k].append(None)

        # with several projection operators in prox_morph:
        # use AlternatingProjections to link them together
        for k in range(self.K):
            # dummy operator: identity
            if len(self.prox_morph[k]) == 0:
                self.prox_morph[k] = proxmin.operators.prox_id
            elif len(self.prox_morph[k]) == 1:
                self.prox_morph[k] = self.prox_morph[k][0]
            else:
                self.prox_morph[k] = proxmin.operators.AlternatingProjections(self.prox_morph[k], repeat=1)

    def remove_component(self, idx, ref_idx=None):
        """Remove a component from the Source

        Parameters
        ----------
        idx: int
            Index of the component in the source to remove
        ref_idx: int, defaul=`None`
            Index of the primary component that `idx` is degenerate with.
            If `ref_idx` is not `None`, the morphology of component `int` is
            added to the morphology of component `ref_idx`.

        Returns
        -------
        None
        """
        # Add the flux from the degenerate component to the primary component
        if ref_idx is not None:
            self.morph[ref_idx] += self.morph[idx]
        # Delete the degenerate morphology and SED
        self.morph = np.delete(self.morph, (idx), axis=0)
        self.sed = np.delete(self.sed, (idx), axis=0)
        # Clear out all of the parameters for the degenerate component
        self.K -= 1
        del self.constraints[idx]
        del self.prox_morph[idx]
        del self.proxs_g_A[idx]
        del self.proxs_g_S[idx]
        del self.LA[idx]
        del self.LS[idx]
        del self.fix_sed[idx]
        del self.fix_morph[idx]

    def remove_degenerate_components(self, sed_diff=1e-5, idx=0):
        """Remove all degenerate components

        Parameters
        ----------
        sed_diff: float, default=1e-5
            Maximum difference between component SED's to consider
            them degenerate.
        idx: int, default=0
            Initial component to begin degenerate search.
            Because the algorithm removes components from the source
            list in place, it is easier to make remove_degenerates a
            recursive function and that is called again after
            degenerate components have been removed.
            To save processing time, we being with the next component
            (`idx`) from the component that just had it's degenerates removed.

        Returns
        -------
        result: bool
            Whether or not any degenerate components were removed
        """
        for l in range(idx, self.K-1):
            degenerates = []
            for ll in range(l+1,self.K):
                diff = np.sum((self.sed[l]-self.sed[ll])**2)
                if diff<sed_diff:
                    degenerates.append(ll)
            # Remove any degenerates for the current source, then restart
            if len(degenerates) > 0:
                for degenerate_idx in degenerates:
                    self.remove_component(degenerate_idx, l)
                logger.warn("Removed degenerate components {0} from source at {1}".format(
                    degenerates, self.center))
                self.remove_degenerate_components(sed_diff, l+1)
                return True
        return False

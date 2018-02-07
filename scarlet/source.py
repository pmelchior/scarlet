from __future__ import print_function, division
import numpy as np
from functools import partial

import proxmin
from . import constraints as sc
from . import config

import logging
logger = logging.getLogger("scarlet.source")

class SourceInitError(Exception):
    pass

class Source(object):

    """A single source in a blend
    """
    def __init__(self, sed, morph_image, constraints=None, center=None, psf=None, fix_sed=False, fix_morph=False,
                 fix_frame=False, shift_center=0.2):
        """Constructor

        Parameters
        ----------
        center: array-like
            (y,x) coordinates of the source in the larger image
        psf: array-like or `~scarlet.transformations.GammaOp`, default=`None`
            2D image of the psf in a single band (Height, Width),
            or 2D image of the psf in each band (Bands, Height, Width),
            or `~scarlet.transformations.GammaOp` created from a psf array.
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
        assert len(sed.shape) == 2
        self.K, self.B = sed.shape
        self.sed = sed.copy()
        assert len(morph_image.shape) == 3
        assert morph_image.shape[0] == sed.shape[0]
        self.morph = morph_image.copy().reshape(self.K, -1)

        if center is None:
            center = (morph_image.shape[1] // 2, morph_image.shape[2] // 2)
        self._set_frame(center, morph_image.shape[1:])
        size = (self.Ny, self.Nx)
        self.fix_frame = fix_frame

        # set up psf and translations matrices
        from . import transformations
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

        # set up constraints: should be Constraint or ConstraintList for each component
        if constraints is None:
            self.constraints = [sc.SimpleConstraint()] * self.K
        elif isinstance(constraints, sc.Constraint) or isinstance(constraints, sc.ConstraintList):
            self.constraints = [constraints] * self.K
        elif len(constraints) == self.K:
            self.constraints = constraints
        else:
            raise NotImplementedError("constraint %r not understood" % constraints)

        # check if prox_sed and prox_morph are set in constraints
        for k in range(self.K):
            if self.constraints[k].prox_sed is None or self.constraints[k].prox_morph is None:
                self.constraints[k] &= sc.SimpleConstraint()
        # needs to set constraints when shape of source is known
        self.set_constraints()

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

    def set_constraints(self):
        for k in range(self.K):
            self.constraints[k].reset(self)

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
            self.set_constraints()

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


def get_pixel_sed(img, position):
    """Get the SED at `position` in `img`

    Parameters
    ----------
    img: `~numpy.array`
        (Bands, Height, Width) data array that contains a 2D image for each band
    position: array-like
        (y,x) coordinates of the source in the larger image

    Returns
    -------
    SED: `~numpy.array`
        SED for a single source
    """
    _y, _x = position
    sed = np.zeros((img.shape[0],))
    sed[:] = img[:,_y,_x]
    if np.all(sed<=0):
        # If the flux in all bands is  <=0,
        # the new sed will be filled with NaN values,
        # which will cause the code to crash later
        msg = "Zero or negative flux at y={0}, x={1}"
        raise SourceInitError(msg.format(_y, _x))

    # ensure proper normalization
    return proxmin.operators.prox_unity_plus(sed, 0)

def get_integrated_sed(img, weight):
    """Calculated the SED by summing the flux in the image in each band
    """
    B, Ny, Nx = img.shape
    sed = (img * weight).reshape(B, -1).sum(axis=1)
    if np.all(sed<=0):
        # If the flux in all bands is  <=0,
        # the new sed will be filled with NaN values,
        # which will cause the code to crash later
        msg = "Zero or negative flux under weight function"
        raise SourceInitError(msg)

    # ensure proper normalization
    return proxmin.operators.prox_unity_plus(sed, 0)


class PointSource(Source):
    def __init__(self, center, img, shape=None, constraints=None, psf=None):
        self.center = center
        if shape is None:
            shape = (np.min(config.source_sizes),) * 2
        sed, morph = self.make_initial(img, shape)

        if constraints is None:
            constraints = sc.SimpleConstraint() & sc.DirectMonotonicityConstraint(use_nearest=False) & sc.SymmetryConstraint()

        super(PointSource, self).__init__(sed, morph, center=center, constraints=constraints, psf=psf, fix_sed=False, fix_morph=False, fix_frame=False, shift_center=0.1)

    def make_initial(self, img, shape, tiny=1e-10):
        """Initialize the source using only the peak pixel

        Parameters
        ----------
        source: `~scarlet.source.Source`
            `Source` to initialize.
        img: `~numpy.array`
            (Bands, Height, Width) data array that contains a 2D image for each band

        This implementation initializes the sed from the pixel in
        the center of the frame and sets morphology to only comprise that pixel,
        which works well for point sources and poorly resolved galaxies.

        """
        # determine initial SED from peak position
        B, Ny, Nx = img.shape
        _y, _x = self.center_int
        try:
            sed = get_pixel_sed(img, self.center_int)
        except SourceInitError:
            # flat weights as fall-back
            sed = np.ones(B) / B
        morph = np.zeros(shape[0] * shape[1])
        # Turn on a single pixel at the peak
        center_pix = morph.size // 2
        morph[center_pix] = max(img[:,_y,_x].sum(axis=0), tiny)
        return sed.reshape((1,B)), morph.reshape((1, shape[0], shape[1]))

class ExtendedSource(Source):
    def __init__(self, center, img, bg_rms, constraints=None, psf=None):
        self.center = center
        sed, morph = self.make_initial(img, bg_rms)

        if constraints is None:
            constraints = sc.SimpleConstraint() & sc.DirectMonotonicityConstraint(use_nearest=False) & sc.SymmetryConstraint()

        super(ExtendedSource, self).__init__(sed, morph, center=center, constraints=constraints, psf=psf, fix_sed=False, fix_morph=False, fix_frame=False, shift_center=0.2)

    def make_initial(self, img, bg_rms, thresh=1., symmetric=True, monotonic=True):

        # every source as large as the entire image, but shifted to its centroid
        B, Ny, Nx = img.shape
        self._set_frame(self.center, (Ny,Nx))

        # determine initial SED from peak position
        try:
            sed = get_pixel_sed(img, self.center_int)
        except SourceInitError:
            # flat weights as fall-back
            sed = np.ones(B) / B

        # build optimal detection coadd
        weights = np.array([sed[b]/bg_rms[b]**2 for b in range(B)])
        jacobian = np.array([sed[b]**2/bg_rms[b]**2 for b in range(B)]).sum()
        detect = np.einsum('i,i...', weights, img) / jacobian

        # copy morph from detect cutout, make non-negative
        source_slice = self.get_slice_for(img.shape)
        morph = np.zeros((self.Ny, self.Nx))
        morph[source_slice[1:]] = detect[self.bb[1:]]

        # check if source_slice is covering the whole of morph:
        # if not, extend morph by coping last row/column from img
        if source_slice[1].stop - source_slice[1].start < self.Ny:
            morph[0:source_slice[1].start,:] = morph[source_slice[1].start,:]
            morph[source_slice[1].stop:,:] = morph[source_slice[1].stop-1,:]
        if source_slice[2].stop - source_slice[2].start < self.Nx:
            morph[:,0:source_slice[2].start] = morph[:,source_slice[2].start][:,None]
            morph[:,source_slice[2].stop:] = morph[:,source_slice[2].stop-1][:,None]

        # symmetric, monotonic
        if symmetric:
            symm = np.fliplr(np.flipud(morph))
            morph = np.min([morph, symm], axis=0)
        if monotonic:
            # use finite thresh to remove flat bridges
            from . import operators
            prox_monotonic = operators.prox_strict_monotonic((self.Ny, self.Nx), thresh=0.1, use_nearest=False)
            morph = prox_monotonic(morph.flatten(), 0).reshape(self.Ny, self.Nx)

        # trim morph to pixels above threshold
        # thresh is multiple above the rms of detect (weighted variance across bands)
        _thresh = thresh * np.sqrt((weights**2 * bg_rms**2).sum()) / jacobian
        mask = morph > _thresh
        if mask.sum() == 0:
            msg = "No flux above threshold={2} for source at y={0}, x={1}"
            _y, _x = self.center_int
            raise SourceInitError(msg.format(_y, _x, _thresh))
        morph[~mask] = 0

        ypix, xpix = np.where(mask)
        _Ny = np.max(ypix)-np.min(ypix)
        _Nx = np.max(xpix)-np.min(xpix)

        # make sure source has odd pixel numbers
        _Ny = config.find_next_source_size(_Ny)
        _Nx = config.find_next_source_size(_Nx)

        # get the model of the source
        Dy, Dx = self.Ny - _Ny, self.Nx - _Nx
        inner = (slice(Dy//2, -Dy//2), slice(Dx//2, -Dx//2))
        morph = morph[inner]

        # updated SED with mean sed under weight function morph
        self._set_frame(self.center, (_Ny, _Nx))
        source_slice = self.get_slice_for(img.shape)
        # use mean sed from image, weighted with the morphology of each component
        try:
            sed = get_integrated_sed(img[self.bb], morph[source_slice[1:]])
        except SourceInitError:
            pass # keep the peak sed
        return sed.reshape((1,B)), morph.reshape((1, morph.shape[0], morph.shape[1]))

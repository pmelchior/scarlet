from __future__ import print_function, division
import numpy as np
from functools import partial

import proxmin
from . import constraints as sc
from .config import Config

import logging
logger = logging.getLogger("scarlet.source")

class SourceInitError(Exception):
    """Error during source initialization
    """
    pass

class Source(object):

    """A single source in a blend.

    This class is fully functional and acts as base class for specialized
    initialization, constraints, etc.
    """
    def __init__(self, sed, morph_image, constraints=None, center=None, psf=None, fix_sed=False,
                 fix_morph=False, fix_frame=False, shift_center=0.2):
        """Constructor

        Parameters
        ----------
        center: array-like
            (y,x) coordinates of the source in the larger image
        psf: array-like or `~scarlet.transformations.Gamma`, default=`None`
            2D image of the psf in a single band (Height, Width),
            or 2D image of the psf in each band (Bands, Height, Width),
            or `~scarlet.transformations.Gamma` created from a psf array.
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
        self.morph = morph_image

        if center is None:
            center = (morph_image.shape[1] // 2, morph_image.shape[2] // 2)
        self._set_frame(center, morph_image.shape[1:])
        size = (self.Ny, self.Nx)
        self.fix_frame = fix_frame

        # set up psf and translations matrices
        from . import transformations
        if isinstance(psf, transformations.Gamma):
            self._gamma = psf
        else:
            self._gamma = transformations.Gamma(psfs=psf)

        # set center coordinates and translation operators
        # needs to have Gamma set up first
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
    def center_int(self):
        """Rounded (not truncated) integer pixel position of the center
        """
        return np.round(self.center).astype('int')

    @property
    def has_psf(self):
        """Whether the source has a psf
        """
        return self._gamma.psfs is not None

    def set_constraints(self):
        """Iterate through all constraints and call their `reset` method
        """
        for k in range(self.K):
            self.constraints[k].reset(self)

    def get_slice_for(self, im_shape):
        """Return the slice of the source frame in the full multiband image

        In other words, return the slice so that
        self.get_model()[k][slice] corresponds to image[self.bb],
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
        Gamma: `~scarlet.transformations.Gamma`, default=`None`
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
            model = np.empty((self.K, self.B, self.Ny, self.Nx))
            for k in range(self.K):
                model[k] = np.outer(sed[k], Gamma.dot(self.morph[k])).reshape(self.B, self.Ny, self.Nx)
        else:
            model = np.zeros((self.K, self.B, self.Ny, self.Nx))
            for k in range(self.K):
                for b in range(self.B):
                    model[k,b] += sed[k,b] * Gamma[b].dot(self.morph[k])

        if combine:
            model = model.sum(axis=0)
        return model

    def _set_frame(self, center, size):
        """Create a frame and bounding box

        To save memory and computation time, each source is contained in a small
        subset of the entire blended image. This method takes the coordinates of
        the source and the size of the frame and creates a bonding box (`self.bb`).

        Sets `self.bottom`, `self.top`, `self.left`, `self.right` as
        the edges of the frame, `self.bb` as the slices (bounding box)
        containing the frame, and `self.center` as the center of the frame.

        Parameters
        ----------
        center: array-like
            (y,x) coordinates of the center of the source in the full image
        size: float or array-like
            Either a (height,width) shape or a single size to create a
            square (size,size) frame.

        Returns
        -------
        old_slice, new_slice: to map subsections of `self.morph` from the old to
        the new shape.

        """
        assert len(center) == 2
        self.center = np.array(center)
        if hasattr(size, '__iter__'):
            size = size[:2]
        else:
            size = (size,) * 2

        # store old edge coordinates
        try:
            top, right, bottom, left = self.top, self.right, self.bottom, self.left
        except AttributeError:
            top, right, bottom, left = [0,] * 4

        # make cutout of in units of the original image frame (that defines xy)
        # ensure odd pixel number
        y_, x_ = self.center_int
        self.bottom, self.top = y_ - int(size[0]//2), y_ + int(size[0]//2) + 1
        self.left, self.right = x_ - int(size[1]//2), x_ + int(size[1]//2) + 1

        # since slice wrap around if start or stop are negative, need to sanitize
        # start values (stop always postive)
        self.bb = (slice(None), slice(max(0, self.bottom), self.top), slice(max(0, self.left), self.right))

        # slices to update self.morph: check if new size is larger or smaller
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
        return old_slice, new_slice

    def set_center(self, center):
        """Given a (y,x) `center`, update the frame and `Gamma`
        """
        if not hasattr(self, '_init_center'):
            self._init_center = np.array([center[0], center[1]])
        size = (self.Ny, self.Nx)
        self._set_frame(center, size)

        # update translation operator
        dyx = self.center - self.center_int
        self.Gamma = self._gamma(dyx)

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
        _, Ny, Nx = self.shape
        old_slice, new_slice = self._set_frame(self.center, size)

        if new_slice != old_slice:
            # change morph
            _morph = self.morph.copy()
            self.morph = np.zeros((self.K, self.Ny, self.Nx))
            self.morph[new_slice] = _morph[old_slice]

            # update Gamma and center (including subpixel shifts)
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
    """Calculate SED from weighted sum of the image in each band
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

def get_best_fit_sed(img, S):
    """Calculate best fitting SED for multiple components.

    Solves min_A ||img - AS||^2 for the SED matrix A, assuming that img only
    contains a single source.
    """
    B = len(img)
    Y = img.reshape(B,-1)
    return np.dot(np.linalg.inv(np.dot(S,S.T)), np.dot(S, Y.T))

class PointSource(Source):
    """Create a point source

    `~scarlet.source.PointSource` objects are initialized with the SED of the peak pixel,
    and the morphology of a single pixel (the peak) turned on.
    While a `~scarlet.source.PointSource` can have any `constraints`, the default constraints are
    symmetry and monotonicity.
    """
    def __init__(self, center, img, shape=None, constraints=None, psf=None, config=None):
        self.center = center
        if config is None:
            config = Config()
        if shape is None:
            shape = (config.source_sizes[0],) * 2
        sed, morph = self.make_initial(img, shape)

        if constraints is None:
            constraints = (sc.SimpleConstraint()
                           & sc.DirectMonotonicityConstraint(use_nearest=False)
                           & sc.DirectSymmetryConstraint())

        super(PointSource, self).__init__(sed, morph, center=center, constraints=constraints, psf=psf,
                                          fix_sed=False, fix_morph=False, fix_frame=False, shift_center=0.1)

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
        morph = np.zeros(shape)
        # Turn on a single pixel at the peak
        cy, cx = (shape[0] // 2, shape[1] //2)
        morph[cy, cx] = max(img[:,_y,_x].sum(axis=0), tiny)
        return sed.reshape((1,B)), morph.reshape((1, shape[0], shape[1]))

class ExtendedSource(Source):
    """Create an extended source

    Extended sources are initialized to have flux that are (optionally) symmetric and
    monotonically decreasing from the peak pixel,
    contained in the minimal box necessary to enclose all of the initial flux.
    By default the model for the source will continue to be monotonic and symmetric,
    but other `constraints` can be used.
    """
    def __init__(self, center, img, bg_rms, constraints=None, psf=None, symmetric=True, monotonic=True,
                 thresh=1., config=None, fix_sed=False, fix_morph=False, fix_frame=False, shift_center=0.2):
        self.center = center
        sed, morph = self.make_initial(img, bg_rms, thresh=thresh, symmetric=symmetric,
                                       monotonic=monotonic, config=config)

        if constraints is None:
            constraints = (sc.SimpleConstraint() &
                           sc.DirectMonotonicityConstraint(use_nearest=False) &
                           sc.DirectSymmetryConstraint())

        super(ExtendedSource, self).__init__(sed, morph, center=center, constraints=constraints, psf=psf,
                                             fix_sed=fix_sed, fix_morph=fix_morph, fix_frame=fix_frame,
                                             shift_center=shift_center)

    def make_initial(self, img, bg_rms, thresh=1., symmetric=True, monotonic=True, config=None):
        """Initialize the source that is symmetric and monotonic

        Parameters
        ----------
        source: :class:`~scarlet.source.Source`
            `Source` to initialize.
        img: :class:`~numpy.array`
            (Bands, Height, Width) data array that contains a 2D image for each band
        bg_rms: array_like
            RMS value of the background in each band. This should have the same shape as `img`.
        symmetric: `bool`
            Whether or not to make the initial morphology symmetric about the peak.
        monotonic: `bool`
            Whether or not to make the initial morphology monotonically decreasing from the peak.
        """
        # Use a default configuration if config is not specified
        if config is None:
            config = Config()
        # every source as large as the entire image, but shifted to its centroid
        B, Ny, Nx = img.shape
        self._set_frame(self.center, (Ny,Nx))
        bg_rms = np.array(bg_rms)

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
            morph = prox_monotonic(morph, 0).reshape(self.Ny, self.Nx)

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

        # make sure source has odd pixel numbers and is from config.source_sizes
        _Ny = config.find_next_source_size(_Ny)
        _Nx = config.find_next_source_size(_Nx)

        # need to reshape morph: store old edge coordinates
        old_slice, new_slice = self._set_frame(self.center, (_Ny, _Nx))

        # update morph
        if new_slice != old_slice:
            _morph = np.zeros((self.Ny, self.Nx))
            _morph[new_slice[1],new_slice[2]] = morph[old_slice[1],old_slice[2]]
            morph = _morph

        # use mean sed from image, weighted with the morphology of each component
        try:
            source_slice = self.get_slice_for(img.shape)
            sed = get_integrated_sed(img[self.bb], morph[source_slice[1:]])
        except SourceInitError:
            # keep the peak sed
            logger.INFO("Using peak SED for source at {0}/{1}".format(self.center_int[0], self.center_int[1]))
        return sed.reshape((1,B)), morph.reshape((1, morph.shape[0], morph.shape[1]))


class MultiComponentSource(ExtendedSource):
    """Create an extended source with multiple components layered vertically.

    Uses `~scarlet.source.ExtendedSource` to define the overall morphology,
    then erodes the outer footprint until it reaches the specified size percentile.
    For the narrower footprint, it evaluates the mean value at the perimeter and
    set the inside to the perimeter value, creating a flat distribution inside.
    The following component(s) is/are set to the difference between the flattened
    and the overall morphology.
    The SED for all components is calculated as the best fit of the multi-component
    morphology to the multi-band image.
    """
    def __init__(self, center, img, bg_rms, size_percentiles=[50], constraints=None, psf=None, symmetric=True, monotonic=True,
                 thresh=1., config=None, fix_sed=False, fix_morph=False, fix_frame=False, shift_center=0.2):
        self.center = center
        sed, morph = self.make_initial(img, bg_rms, size_percentiles=size_percentiles,
                                       thresh=thresh, symmetric=symmetric,
                                       monotonic=monotonic, config=config)
        K = len(sed)
        if constraints is None:
            constraints = [sc.SimpleConstraint() &
                           sc.DirectMonotonicityConstraint(use_nearest=False) &
                           sc.DirectSymmetryConstraint()] * K

        super(ExtendedSource, self).__init__(sed, morph, center=center, constraints=constraints, psf=psf, fix_sed=fix_sed, fix_morph=fix_morph, fix_frame=fix_frame, shift_center=shift_center)

    def make_initial(self, img, bg_rms, size_percentiles=[50], thresh=1., symmetric=True, monotonic=True, config=None):
        """Initialize multi-component source, where the inner components begin
        at the given size_percentiles.

        See `~scarlet.source.ExtendedSource` for details.
        """
            # call make_initial from ExtendedSource to give single-component morphology and sed
        sed, morph = super(MultiComponentSource, self).make_initial(img, bg_rms, thresh=thresh, symmetric=symmetric, monotonic=monotonic, config=config)

        # create a list of components from morph by layering them on top of each
        # other and sum up to morph
        from scipy.ndimage.morphology import binary_erosion
        K = len(size_percentiles) + 1
        Ny, Nx = morph.shape[1:]
        morph_ = np.zeros((K, Ny, Nx))
        morph_[0,:,:] = morph[0]
        mask = morph[0] > 0
        radius = np.sqrt(mask.sum()/np.pi)
        # make sure they are in decendind order
        percentiles_ = np.sort(size_percentiles)[::-1]
        for k in range(1,K):
            perc = percentiles_[k-1]
            while True:
                # erode footprint from the outside
                mask_ = binary_erosion(mask)
                # keep central pixel on
                mask_[Ny//2,Nx//2] = True
                if np.sqrt(mask_.sum()/np.pi) < perc*radius/100 or mask_.sum() == 1:
                    # set inside of prior component to value at perimeter
                    perimeter = mask & (~mask_)
                    perimeter_val = morph[0][perimeter].mean()
                    morph_[k-1][mask_] = perimeter_val
                    # set this component to morph - perimeter_val, bounded by 0
                    morph[0] -= perimeter_val
                    morph_[k][mask_] = np.maximum(morph[0][mask_], 0)
                    # correct for negative pixels by putting them into k-1 component
                    below = mask_ & (morph[0] < 0)
                    if below.sum():
                        morph_[k-1][below] += morph[0][below]
                    mask = mask_
                    break
                mask = mask_

        # optimal SED assuming img only has that source
        source_slice = self.get_slice_for(img.shape)
        S = morph_[:,source_slice[1], source_slice[2]].reshape(K, -1)
        sed_ = get_best_fit_sed(img[self.bb], S)

        return sed_, morph_

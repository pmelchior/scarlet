from __future__ import print_function, division
import numpy as np
from functools import partial

import proxmin
from . import constraints as sc
from .config import Config

import logging
logger = logging.getLogger("scarlet.source")

class Component(object):
    """A single component in a blend.

    This class acts as base for building complex sources.
    """
    def __init__(self, sed, morph, center=None, constraints=None, psf=None, fix_sed=False,
                 fix_morph=False, fix_frame=False, shift_center=0.2):
        """Constructor

        Create source with K components from a matrix of SEDs and morphologies.

        Parameters
        ----------
        sed: array
            1D array (bands) of the initial SED.
        morph: array
            Data cube (Height, Width) of the initial morphology.
        center: array-like
            (y,x) coordinates of the component in the larger image
        constraints: :class:`scarlet.constraint.Constraint` or :class:`scarlet.constraint.ConstraintList`
            Constraints used to constrain the SED and/or morphology.
            When `constraints` is `None` then
            :class:`scarlet.constraint.DirectMonotonicityConstraint`
            and :class:`scarlet.constraint.SimpleConstraint` are used.
        psf: array-like or `~scarlet.transformations.Gamma`, default=`None`
            2D image of the psf in a single band (Height, Width),
            or 2D image of the psf in each band (Bands, Height, Width),
            or `~scarlet.transformations.Gamma` created from a psf array.
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
        # set size of the source frame
        self.B = sed.size
        self.sed = sed.copy()
        assert len(morph.shape) == 2
        self.morph = morph.copy()

        if center is None:
            center = (morph.shape[0] // 2, morph.shape[1] // 2)
        self._set_frame(center, morph.shape)

        # set up psf and translations matrices
        from . import transformations
        if isinstance(psf, transformations.Gamma):
            self._gamma = psf
        else:
            if psf is not None and len(psf.shape)==2:
                psf = np.array([psf]*self.B)
            self._gamma = transformations.Gamma(psfs=psf)

        # set center coordinates and translation operators
        # needs to have Gamma set up first
        self.set_center(center)
        self.shift_center = shift_center

        # updates for frame, sed, morph?
        self.fix_frame = fix_frame
        self.fix_sed = fix_sed
        self.fix_morph = fix_morph

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
        """Shape of the image (Band, Height, Width)
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

    def set_constraints(self, constraints):
        """Set up constraints for component.

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
            constraints = sc.SimpleConstraint()

        self.constraints = sc.ConstraintAdapter(constraints, self)

    def get_slice_for(self, im_shape):
        """Return the slice of the component frame in the full multiband image

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

    def get_model(self, Gamma=None, use_sed=True):
        """Get the model this component.

        Parameters
        ----------
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
            model = np.empty((self.B, self.Ny, self.Nx))
            model = np.outer(sed, Gamma.dot(self.morph)).reshape(self.B, self.Ny, self.Nx)
        else:
            model = np.zeros((self.B, self.Ny, self.Nx))
            for b in range(self.B):
                model[b] += sed[b] * Gamma[b].dot(self.morph)

        return model

    def _set_frame(self, center, size):
        """Create a frame and bounding box

        To save memory and computation time, each source is contained in a small
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
        new_slice = (new_slice_y, new_slice_x)
        old_slice = (old_slice_y, old_slice_x)
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
        old_slice, new_slice = self._set_frame(self.center, size)

        if new_slice != old_slice:
            # change morph
            _morph = self.morph.copy()
            self.morph = np.zeros((self.Ny, self.Nx))
            self.morph[new_slice] = _morph[old_slice]

            # update Gamma and center (including subpixel shifts)
            self.set_center(self.center)

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

        # compute direct error propagation assuming only this source SED(s)
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


class SourceInitError(Exception):
    """Error during source initialization
    """
    pass

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


class Source(object):
    def __init__(self, components, label=None):
        if isinstance(components, Component):
            components = [components]
        self.components = components
        self.label = label

    def update_center(self):
        _flux = np.array([c.morph.sum() for c in self.components])
        _center = np.sum([_flux[k]*self.components[k].center for k in range(len(self.components))], axis=0)
        _center /= _flux.sum()
        if len(self.components) > 1:
            for k in range(len(self.components)):
                c = self.components[k]
                if c.shift_center:
                    c.center = _center
                    msg = "updating component {0}.{1} center to ({2:.3f}/{3:.3f})"
                    logger.debug(msg.format(self.label, k, source.center[0], source.center[1]))

    def update_sed(self):
        pass

    def update_morph(self):
        pass


class PointSource(Source):
    """Create a point source.

    Point sources are initialized with the SED of the center pixel,
    and the morphology of a single pixel (the center) turned on.
    While the source can have any `constraints`, the default constraints are
    symmetry and monotonicity.
    """
    def __init__(self, center, img, label=None, shape=None, constraints=None, psf=None, config=None):
        """Initialize

        This implementation initializes the sed from the pixel in
        the center of the frame and sets morphology to only comprise that pixel,
        which works well for point sources and poorly resolved galaxies.

        See :class:`~scarlet.source.Source` for parameter descriptions not listed below.

        Parameters
        ----------
        img: :class:`~numpy.array`
            (Bands, Height, Width) data array that contains a 2D image for each band
        shape: tuple
            Shape of the initial morphology.
            If `shape` is `None` then the smallest shape specified by `config.source_sizes`
            is used.
        """
        if config is None:
            config = Config()
        if shape is None:
            shape = (config.source_sizes[0],) * 2
        sed, morph = self._make_initial(img, center, shape)

        if constraints is None:
            constraints = (sc.SimpleConstraint(),
                           sc.DirectMonotonicityConstraint(use_nearest=False),
                           sc.DirectSymmetryConstraint())

        component = Component(sed, morph, center=center, constraints=constraints, psf=psf, fix_sed=False, fix_morph=False, fix_frame=False, shift_center=0.1)
        super(PointSource, self).__init__(component, label=label)

    def _make_initial(self, img, center, shape, tiny=1e-10):
        """Initialize the source using only the peak pixel

        See `self.__init__` for parameters not listed below

        Parameters
        ----------
        tiny: float
            Minimal non-zero value allowed for a source.
            This ensures that the source is initialized with
            some non-zero flux.
        """
        # determine initial SED from peak position
        B, Ny, Nx = img.shape
        _y, _x = center_int = np.round(center).astype('int')
        try:
            sed = get_pixel_sed(img, center_int)
        except SourceInitError:
            # flat weights as fall-back
            sed = np.ones(B) / B
        morph = np.zeros(shape)
        # Turn on a single pixel at the peak
        cy, cx = (shape[0] // 2, shape[1] //2)
        morph[cy, cx] = max(img[:,_y,_x].sum(axis=0), tiny)
        return sed, morph

class ExtendedSource(Source):
    """Create an extended source.

    Extended sources are initialized to have a morphology given by the pixels in the
    multi-band image that are detectable above the background noise.
    The initial morphology can be constrained to be symmetric and monotonically
    decreasing from the center pixel, and will be enclosed in a frame with the
    minimal box size, as specied by `~scarlet.config.Config`.

    By default the model for the source will continue to be monotonic and symmetric,
    but other `constraints` can be used.
    """
    def __init__(self, center, img, bg_rms, label=None, constraints=None, psf=None, symmetric=True, monotonic=True,
                 thresh=1., config=None, fix_sed=False, fix_morph=False, fix_frame=False, shift_center=0.2):
        """Initialize

        See :class:`~scarlet.source.Source` for parameter descriptions not listed below.

        Parameters
        ----------
        img: :class:`~numpy.array`
            (Bands, Height, Width) data array that contains a 2D image for each band
        bg_rms: array_like
            RMS value of the background in each band.
        symmetric: `bool`
            Whether or not to make the initial morphology symmetric about the peak.
        monotonic: `bool`
            Whether or not to make the initial morphology monotonically decreasing from the peak.
        thresh: float
            Multiple of the RMS used to set the minimum non-zero flux.
            Use `thresh=1` to just use `bg_rms` to set the flux floor.
        """
        sed, morph = self._make_initial(img, center, bg_rms, thresh=thresh, symmetric=symmetric, monotonic=monotonic, config=config)

        if constraints is None:
            constraints = (sc.SimpleConstraint(),
                           sc.DirectMonotonicityConstraint(use_nearest=False),
                           sc.DirectSymmetryConstraint())

        component = Component(sed, morph, center=center, constraints=constraints, psf=psf, fix_sed=fix_sed, fix_morph=fix_morph, fix_frame=fix_frame, shift_center=shift_center)
        super(ExtendedSource, self).__init__(component, label=label)

    def _make_initial(self, img, center, bg_rms, thresh=1., symmetric=True, monotonic=True, config=None):
        """Initialize the source that is symmetric and monotonic

        See `self.__init__` for a description of the parameters
        """
        # Use a default configuration if config is not specified
        if config is None:
            config = Config()

        # every source as large as the entire image, but shifted to its centroid
        # using a temp Component for its frame methods
        B, Ny, Nx = img.shape
        sed = np.empty(B)
        morph = np.empty((Ny, Nx))
        component = Component(sed, morph, center=center)
        bg_rms = np.array(bg_rms)

        # determine initial SED from peak position
        try:
            sed = get_pixel_sed(img, component.center_int)
        except SourceInitError:
            # flat weights as fall-back
            sed = np.ones(B) / B

        # build optimal detection coadd
        weights = np.array([sed[b]/bg_rms[b]**2 for b in range(B)])
        jacobian = np.array([sed[b]**2/bg_rms[b]**2 for b in range(B)]).sum()
        detect = np.einsum('i,i...', weights, img) / jacobian

        # thresh is multiple above the rms of detect (weighted variance across bands)
        bg_cutoff = thresh * np.sqrt((weights**2 * bg_rms**2).sum()) / jacobian
        morph = self._init_morph(morph, detect, component, bg_cutoff, symmetric, monotonic, config)

        # use mean sed from image, weighted with the morphology of each component
        try:
            component_slice = component.get_slice_for(img.shape)
            sed = get_integrated_sed(img[component.bb], morph[component_slice[1:]])
        except SourceInitError:
            # keep the peak sed
            logger.INFO("Using peak SED for source at {0}/{1}".format(component.center_int[0], component.center_int[1]))
        return sed, morph

    def _init_morph(self, morph, detect, component, bg_cutoff=0, symmetric=True, monotonic=True, config=None):
        """Initialize the morphology

        Parameters
        ----------
        morph: array
            Initial morphology guess
        source_slice: list of slices
            Slices corresponding to the pixels in the image data that
            are contained in `morph`.
        bg_cutoff: float
            Minimum non-zero flux value allowed before truncating the morphology
        """

        # copy morph from detect cutout, make non-negative
        shape = (component.B,) + detect.shape
        component_slice = component.get_slice_for(shape)
        morph[:,:] = 0
        morph[component_slice[1:]] = detect[component.bb[1:]]

        # check if component_slice is covering the whole of morph:
        # if not, extend morph by coping last row/column from img
        if component_slice[1].stop - component_slice[1].start < component.Ny:
            morph[0:component_slice[1].start,:] = morph[component_slice[1].start,:]
            morph[component_slice[1].stop:,:] = morph[component_slice[1].stop-1,:]
        if component_slice[2].stop - component_slice[2].start < component.Nx:
            morph[:,0:component_slice[2].start] = morph[:,component_slice[2].start][:,None]
            morph[:,component_slice[2].stop:] = morph[:,component_slice[2].stop-1][:,None]

        # symmetric, monotonic
        if symmetric:
            symm = np.fliplr(np.flipud(morph))
            morph = np.min([morph, symm], axis=0)
        if monotonic:
            # use finite thresh to remove flat bridges
            from . import operators
            prox_monotonic = operators.prox_strict_monotonic((component.Ny, component.Nx), thresh=0.1, use_nearest=False)
            morph = prox_monotonic(morph, 0).reshape(component.Ny, component.Nx)

        # trim morph to pixels above threshold
        mask = morph > bg_cutoff
        if mask.sum() == 0:
            msg = "No flux above threshold={2} for source at y={0}, x={1}"
            _y, _x = component.center_int
            raise SourceInitError(msg.format(_y, _x, bg_cutoff))
        morph[~mask] = 0

        ypix, xpix = np.where(mask)
        _Ny = np.max(ypix)-np.min(ypix)
        _Nx = np.max(xpix)-np.min(xpix)

        # make sure source has odd pixel numbers and is from config.source_sizes
        _Ny = config.find_next_source_size(_Ny)
        _Nx = config.find_next_source_size(_Nx)

        # need to reshape morph: store old edge coordinates
        old_slice, new_slice = component._set_frame(component.center, (_Ny, _Nx))

        # update morph
        if new_slice != old_slice:
            _morph = np.zeros((component.Ny, component.Nx))
            _morph[new_slice] = morph[old_slice]
            morph = _morph
        return morph


"""
class MultiComponentSource(ExtendedSource):
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
"""

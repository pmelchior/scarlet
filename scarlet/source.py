from __future__ import print_function, division
import numpy as np

import proxmin
from . import constraint as sc
from .config import Config
from .component import Component, ComponentTree

import logging
logger = logging.getLogger("scarlet.source")


class Source(ComponentTree):
    """Base class for co-centered `~scarlet.component.Component`s.

    The class implements `update_center` to set all components with `shift_center > 0`
    to the flux-weighted mean center position of all components.
    """
    def __init__(self, components):
        """Constructor.

        Parameters
        ----------
        components: list of `~scarlet.component.Component` or `~scarlet.component.ComponentTree`
        """
        super(Source, self).__init__(components)

    def get_model(self):
        """Compute the model for this source.

        NOTE: If individual components have different shape, the resulting
        model will be set in a box that can contain all of them.

        Returns
        -------
        `~numpy.array` with shape (B, Ny, Nx)
        """
        models = [c.get_model() for c in self.components]
        # model may have different boxes, need to put into box that fits all
        maxNy, maxNx = 0,0
        for model in models:
            Ny, Nx = model.shape[1:]
            if Ny > maxNy:
                maxNy = Ny
            if Nx > maxNx:
                maxNx = Nx
        for k in range(len(models)):
            Ny, Nx = models[k].shape[1:]
            if (Ny, Nx) != (maxNy, maxNx):
                _model = np.zeros((models[k].shape[0], maxNy, maxNx))
                _model[:, (maxNy-Ny)//2:maxNy-(maxNy-Ny)//2 , (maxNx-Nx)//2:maxNx-(maxNx-Nx)//2] = models[k][:,:,:]
                models[k] = _model
        return np.sum(models, axis=0)

    def update_center(self):
        """Center update to set all component centers to flux-weighted mean position.

        NOTE: Only components with `shift_center > 0` will be moved.
        """
        if len(self.components) > 1:
            _flux = np.array([c.morph.sum() for c in self.components])
            _center = np.sum([_flux[k]*self.components[k].center for k in range(self.K)], axis=0)
            _center /= _flux.sum()
            for c in self.components:
                if c.shift_center:
                    c.center = _center
                    msg = "updating component {0} center to ({1:.3f}/{2:.3f})"
                    logger.debug(msg.format(c.coord, c.center[0], c.center[1]))


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


class PointSource(Source):
    """Create a point source.

    Point sources are initialized with the SED of the center pixel,
    and the morphology of a single pixel (the center) turned on.
    While the source can have any `constraints`, the default constraints are
    symmetry and monotonicity.
    """
    def __init__(self, center, img, shape=None, constraints=None, psf=None, config=None):
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
        sed, morph = self._make_initial(center, img, shape)

        if constraints is None:
            constraints = (sc.SimpleConstraint(),
                           sc.DirectMonotonicityConstraint(use_nearest=False),
                           sc.DirectSymmetryConstraint())

        component = Component(sed, morph, center=center, constraints=constraints, psf=psf, fix_sed=False, fix_morph=False, fix_frame=False, shift_center=0.1)
        super(PointSource, self).__init__(component)

    def _make_initial(self, center, img, shape, tiny=1e-10):
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
    def __init__(self, center, img, bg_rms, constraints=None, psf=None, symmetric=True, monotonic=True,
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
        # Use a default configuration if config is not specified
        if config is None:
            config = Config()

        sed, morph = self._make_initial(center, img, bg_rms, thresh=thresh, symmetric=symmetric, monotonic=monotonic, config=config)

        if constraints is None:
            constraints = (sc.SimpleConstraint(),
                           sc.DirectMonotonicityConstraint(use_nearest=False),
                           sc.DirectSymmetryConstraint())

        component = Component(sed, morph, center=center, constraints=constraints, psf=psf, fix_sed=fix_sed, fix_morph=fix_morph, fix_frame=fix_frame, shift_center=shift_center)
        super(ExtendedSource, self).__init__(component)

    def _make_initial(self, center, img, bg_rms, thresh=1., symmetric=True, monotonic=True, config=None):
        """Initialize the source that is symmetric and monotonic

        See `self.__init__` for a description of the parameters
        """
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

class MultiComponentSource(ExtendedSource):
    """Create an extended source with multiple components layered vertically.
    Uses `~scarlet.source.ExtendedSource` to define the overall morphology,
    then erodes the outer footprint until it reaches the specified size percentile.
    For the narrower footprint, it evaluates the mean value at the perimeter and
    sets the inside to the perimeter value, creating a flat distribution inside.
    The subsequent component(s) is/are set to the difference between the flattened
    and the overall morphology.
    The SED for all components is calculated as the best fit of the multi-component
    morphology to the multi-band image in the region of the source.
    """
    def __init__(self, center, img, bg_rms, size_percentiles=[50], constraints=None, psf=None, symmetric=True, monotonic=True,
                 thresh=1., config=None, fix_sed=False, fix_morph=False, fix_frame=False, shift_center=0.2):
        """Initialize multi-component source, where the inner components begin
        at the given size_percentiles.
        See `~scarlet.source.ExtendedSource` for details.
        """
        # Use a default configuration if config is not specified
        if config is None:
            config = Config()

        if constraints is None:
            constraints = (sc.SimpleConstraint(),
                           sc.DirectMonotonicityConstraint(use_nearest=False),
                           sc.DirectSymmetryConstraint())

        # start from ExtendedSource for single-component morphology and sed
        super(MultiComponentSource, self).__init__(center, img, bg_rms, constraints=constraints, psf=psf, symmetric=symmetric, monotonic=monotonic, thresh=thresh, config=config, fix_sed=fix_sed, fix_morph=fix_morph, fix_frame=fix_frame, shift_center=shift_center)

        # create a list of components from base morph by layering them on top of
        # each other so that they sum up to morph
        from scipy.ndimage.morphology import binary_erosion
        K = len(size_percentiles) + 1

        morph = self.components[0].morph
        Ny, Nx = morph.shape
        morphs = [np.zeros((Ny, Nx)) for k in range(K)]
        morphs[0][:,:] = morph[:,:]
        mask = morph > 0
        radius = np.sqrt(mask.sum()/np.pi)
        percentiles_ = np.sort(size_percentiles)[::-1] # decending order
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
                    perimeter_val = morph[perimeter].mean()
                    morphs[k-1][mask_] = perimeter_val
                    # set this component to morph - perimeter_val, bounded by 0
                    morph -= perimeter_val
                    morphs[k][mask_] = np.maximum(morph[mask_], 0)
                    # correct for negative pixels by putting them into k-1 component
                    below = mask_ & (morph < 0)
                    if below.sum():
                        morphs[k-1][below] += morph[below]
                    mask = mask_
                    break
                mask = mask_

        # optimal SEDs given the morphologies, assuming img only has that source
        c = self.components[0]
        component_slice = c.get_slice_for(img.shape)
        S = np.array(morphs)[:,component_slice[1], component_slice[2]].reshape(K, -1)
        seds = get_best_fit_sed(img[c.bb], S)

        for k in range(K):
            if k == 0:
                self.components[0].morph = morphs[0]
                self.components[0].sed = seds[0]
            else:
                component = Component(seds[k], morphs[k], center=center, constraints=constraints, psf=psf, fix_sed=fix_sed, fix_morph=fix_morph, fix_frame=fix_frame, shift_center=shift_center)

                # reduce the shape of the additional components as much as possible
                mask = morphs[k] > 0
                ypix, xpix = np.where(mask)
                _Ny = np.max(ypix)-np.min(ypix)
                _Nx = np.max(xpix)-np.min(xpix)
                # make sure source has odd pixel numbers and is from config.source_sizes
                _Ny = config.find_next_source_size(_Ny)
                _Nx = config.find_next_source_size(_Nx)
                component.resize((_Ny, _Nx))

                self += component

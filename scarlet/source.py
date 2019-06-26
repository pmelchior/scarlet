import autograd.numpy as np

from .component import Component, ComponentTree
from . import operator
from . import update
from .interpolation import get_projection_slices

import logging

logger = logging.getLogger("scarlet.source")


class SourceInitError(Exception):
    """Error during source initialization
    """
    pass


def get_pixel_sed(sky_coord, observation):
    """Get the SED at `position` in `img`

    Parameters
    ----------
    sky_coord: tuple
        Center of the source
    observation: `~scarlet.Observation`
        Observation to extract SED from.

    Returns
    -------
    SED: `~numpy.array`
        SED for a single source
    """

    pixel = observation.frame.get_pixel(sky_coord)
    sed = observation.images[:, pixel[0], pixel[1]].copy()
    if observation.frame.psfs is not None:
        # Account for the PSF in the intensity
        sed /= observation.frame.psfs.max(axis=(1, 2))

    if np.all(sed[-1] <= 0):
        # If the flux in all channels is  <=0,
        # the new sed will be filled with NaN values,
        # which will cause the code to crash later
        msg = "Zero or negative flux at y={0}, x={1}"
        raise SourceInitError(msg.format(*sky_coord))

    return sed


def get_best_fit_seds(morphs, frame, observation):
    """Calculate best fitting SED for multiple components.

    Solves min_A ||img - AS||^2 for the SED matrix A,
    assuming that the images only contain a single source.

    Parameters
    ----------
    morphs: list
        Morphology for each component in the source.
    frame: `scarlet.observation.frame`
        The frame that the model lives in.
    observation: `~scarlet.Observation`
        Observation to extract SEDs from.
    """
    K = len(morphs)
    _morph = morphs.reshape(K, -1)
    images = observation.images
    data = images.reshape(observation.frame.C, -1)
    seds = np.dot(np.linalg.inv(np.dot(_morph, _morph.T)), np.dot(_morph, data.T))
    return seds

def build_detection_coadd(sed, bg_rms, observation, thresh=1):
    """Build a channel weighted coadd to use for source detection

    Parameters
    ----------
    sed: array
        SED at the center of the source.
    bg_rms: array
        Background RMS in each channel in observation.
    observation: `~scarlet.observation.Observation`
        Observation to use for the coadd.
    thresh: `float`
        Multiple of the backround RMS used as a
        flux cutoff.

    Returns
    -------
    detect: array
        2D image created by weighting all of the channels by SED
    bg_cutoff: float
        The minimum value in `detect` to include in detection.
    """
    C = len(sed)
    if np.any(bg_rms <= 0):
        raise ValueError("bg_rms must be greater than zero in all channels")

    weights = np.array([sed[c] / bg_rms[c] ** 2 for c in range(C)])
    jacobian = np.array([sed[c] ** 2 / bg_rms[c] ** 2 for c in range(C)]).sum()
    detect = np.einsum('i,i...', weights, observation.images) / jacobian

    # thresh is multiple above the rms of detect (weighted variance across channels)
    bg_cutoff = thresh * np.sqrt((weights ** 2 * bg_rms ** 2).sum()) / jacobian
    return detect, bg_cutoff


def init_extended_source(sky_coord, frame, observation, bg_rms,
                         thresh=1., symmetric=True, monotonic=True):
    """Initialize the source that is symmetric and monotonic
    See `ExtendedSource` for a description of the parameters
    """
    # determine initial SED from peak position
    sed = get_pixel_sed(sky_coord, observation)  # amplitude is in sed
    if frame.psfs is not None:
        sed = sed * frame.psfs[0].max()

    morph, bg_cutoff = build_detection_coadd(sed, bg_rms, observation, thresh)
    center = frame.get_pixel(sky_coord)

    # Apply the necessary constraints
    if symmetric:
        morph = operator.prox_uncentered_symmetry(morph, 0, center=center, use_soft=False)
    if monotonic:
        # use finite thresh to remove flat bridges
        prox_monotonic = operator.prox_strict_monotonic(morph.shape, use_nearest=False,
                                                        center=center, thresh=.1)
        morph = prox_monotonic(morph, 0).reshape(morph.shape)

    # trim morph to pixels above threshold
    mask = morph > bg_cutoff
    if mask.sum() == 0:
        msg = "No flux above threshold={2} for source at y={0} x={1}"
        raise SourceInitError(msg.format(*sky_coord, bg_cutoff))
    morph[~mask] = 0

    # normalize to unity at peak pixel
    cy, cx = np.array(center).astype(int)
    center_morph = morph[cy, cx]
    morph /= center_morph
    return sed, morph


def init_combined_extended_source(sky_coord, frame, observations, bg_rms, obs_idx=0,
                                  thresh=1., symmetric=True, monotonic=True):
    """Initialize the source that is symmetric and monotonic
    See `ExtendedSource` for a description of the parameters
    """
    try:
        iter(observations)
    except TypeError:
        observations = [observations]

    # determine initial SED from peak position
    # SED in the frame for source detection

    seds = []
    for obs in observations:
        _sed = get_pixel_sed(sky_coord, obs)
        seds.append(_sed)
    sed = np.concatenate(seds).flatten()

    morph, bg_cutoff = build_detection_coadd(seds[obs_idx], bg_rms[obs_idx], observations[obs_idx], thresh)  # amplitude is in sed

    center = frame.get_pixel(sky_coord)

    # Apply the necessary constraints
    if symmetric:
        morph = operator.prox_uncentered_symmetry(morph, 0, center=center, use_soft=False)

    if monotonic:
        # use finite thresh to remove flat bridges
        prox_monotonic = operator.prox_strict_monotonic(morph.shape, use_nearest=False,
                                                        center=center, thresh=.1)
        morph = prox_monotonic(morph, 0).reshape(morph.shape)

    # trim morph to pixels above threshold
    mask = morph > bg_cutoff
    if mask.sum() == 0:
        msg = "No flux above threshold={2} for source at y={0} x={1}"
        raise SourceInitError(msg.format(*center, bg_cutoff))
    morph[~mask] = 0

    # normalize to unity at peak pixel
    cy, cx = center

    center_morph = morph[np.int(cy), np.int(cx)]
    morph /= center_morph
    return sed, morph


def init_multicomponent_source(sky_coord, frame, observation, bg_rms, flux_percentiles=None,
                               thresh=1., symmetric=True, monotonic=True):
    """Initialize multiple components
    See `MultiComponentSource` for a description of the parameters
    """
    if flux_percentiles is None:
        flux_percentiles = [25]
    # Initialize the first component as an extended source
    sed, morph = init_extended_source(sky_coord, frame, observation, bg_rms,
                                      thresh, symmetric, monotonic)
    # create a list of components from base morph by layering them on top of
    # each other so that they sum up to morph
    K = len(flux_percentiles) + 1

    Ny, Nx = morph.shape
    morphs = np.zeros((K, Ny, Nx), dtype=morph.dtype)
    morphs[0, :, :] = morph[:, :]
    max_flux = morph.max()
    percentiles_ = np.sort(flux_percentiles)
    last_thresh = 0
    for k in range(1, K):
        perc = percentiles_[k - 1]
        flux_thresh = perc * max_flux / 100
        mask_ = morph > flux_thresh
        morphs[k - 1][mask_] = flux_thresh - last_thresh
        morphs[k][mask_] = morph[mask_] - flux_thresh
        last_thresh = flux_thresh

    # renormalize morphs: initially Smax
    for k in range(K):
        morphs[k] /= morphs[k].max()

    # optimal SEDs given the morphologies, assuming img only has that source
    seds = get_best_fit_seds(morphs, frame, observation)

    return seds, morphs


class PointSource(Component):
    def __init__(self, frame, sky_coord, observation, symmetric=False, monotonic=True,
                 center_step=5, delay_thresh=10, **component_kwargs):
        """Extended source intialized with a single pixel

        Point sources are initialized with the SED of the center pixel,
        and the morphology of a single pixel (the center) turned on.
        While the source can have any `constraints`, the default constraints are
        symmetry and monotonicity.

        Parameters
        ----------
        frame: `~scarlet.Frame`
            The frame that the model lives in.
        sky_coord: tuple
            Center of the source
        observation: list of `~scarlet.Observation`
            Observation to initialize this source
        symmetric: bool
            Whether or not the object is forced to be symmetric
        monotonic: bool
            Whether or not the object is forced to be monotonically decreasing
        center_step: int
            Number of steps to skip between centering routines
        delay_thresh: int
            Number of steps to skip before turning on thresholding.
            This is useful for point sources because it allows them to grow
            slightly before removing pixels with low significance.
        component_kwargs: dict
            Keyword arguments to pass to the component initialization.
        """
        # this ignores any broadening from the PSFs ...
        C, Ny, Nx = frame.shape
        morph = np.zeros((Ny, Nx), observation.images.dtype)
        pixel = frame.get_pixel(sky_coord)
        if frame.psfs is None:
            # Use a single pixel if there is no target PSF
            morph[pixel] = 1
        else:
            # A point source is a function of the target PSF
            assert len(frame.psfs[0].shape) == 2
            py, px = pixel
            sy, sx = (np.array(frame.psfs[0].shape) - 1) // 2
            cy, cx = (np.array(morph.shape) - 1) // 2
            yx0 = int(py - cy - sy), int(px - cx - sx)
            bb, ibb, _ = get_projection_slices(frame.psfs[0], morph.shape, yx0)
            morph[bb] = frame.psfs[0][ibb]

        self.pixel_center = pixel
        pixel = observation.frame.get_pixel(sky_coord)
        sed = observation.images[:, pixel[0], pixel[1]].copy()
        if observation.frame.psfs is not None:
            # Account for the PSF in the intensity
            sed /= observation.frame.psfs.max(axis=(1, 2))

        super().__init__(frame, sed, morph, **component_kwargs)
        self.symmetric = symmetric
        self.monotonic = monotonic
        self.center_step = center_step
        self.delay_thresh = delay_thresh
        self.update()

    def update(self):
        """Default update parameters for an ExtendedSource

        This method can be overwritten if a different set of constraints
        or update functions is desired.
        """
        if self._parent is None:
            it = 0
        else:
            it = self._parent.it
        # Update the central pixel location (pixel_center)
        update.fit_pixel_center(self)
        if it > self.delay_thresh:
            update.threshold(self)

        if self.symmetric:
            # Translate to the centered frame
            update.translation(self, 1)
            # make the morphology perfectly symmetric
            update.symmetric(self, strength=1)
            # Translate back to the model frame
            update.translation(self, -1)

        if self.monotonic:
            # make the morphology monotonically decreasing
            if hasattr(self, "bboxes") and "thresh" in self.bboxes:
                update.monotonic(self, self.pixel_center, bbox=self.bboxes["thresh"])
            else:
                update.monotonic(self, self.pixel_center)

        update.positive(self)  # Make the SED and morph non-negative
        update.normalized(self)  # Use MORPH_MAX normalization
        return self


class ExtendedSource(PointSource):
    def __init__(self, frame, sky_coord, observation, bg_rms, thresh=1,
                 symmetric=False, monotonic=True, center_step=5, delay_thresh=10, **component_kwargs):
        """Extended source intialized to match a set of observations

        Parameters
        ----------
        frame: `~scarlet.Frame`
            The frame that the model lives in.
        sky_coord: tuple
            Center of the source
        observation: `~scarlet.observation.Observation`
            Observation to initialize this source.
        bg_rms: array
            Background RMS in each channel in observation.
        thresh: `float`
            Multiple of the backround RMS used as a
            flux cutoff for morphology initialization.
        symmetric: `bool`
            Whether or not to enforce symmetry.
        monotonic: `bool`
            Whether or not to make the object monotonically decrease
            in flux from the center.
        component_kwargs: dict
            Keyword arguments to pass to the component initialization.
        """
        self.symmetric = symmetric
        self.monotonic = monotonic
        self.coords = sky_coord
        center = frame.get_pixel(sky_coord)
        self.pixel_center = center
        self.center_step = center_step
        self.delay_thresh = delay_thresh

        sed, morph = init_extended_source(sky_coord, frame, observation, bg_rms,
                                          thresh, True, monotonic)

        Component.__init__(self, frame, sed, morph, **component_kwargs)
        self.update()


class CombinedExtendedSource(PointSource):
    def __init__(self, frame, sky_coord, observations, bg_rms, obs_idx=0, thresh=1,
                 symmetric=False, monotonic=True, center_step=5, delay_thresh=0, **component_kwargs):
        """Extended source intialized to match a set of observations

        Parameters
        ----------
        frame: `~scarlet.Frame`
            The frame that the model lives in.
        sky_coord: tuple
            Center of the source
        observations: list of `~scarlet.Observation`
            Observations to initialize this source.
        bg_rms: array
            Background RMS in each channel in observation.
        obs_idx: int
            Index of the observation in `observations` to use for
            initializing the morphology.
        thresh: `float`
            Multiple of the backround RMS used as a
            flux cutoff for morphology initialization.
        symmetric: `bool`
            Whether or not to enforce symmetry.
        monotonic: `bool`
            Whether or not to make the object monotonically decrease
            in flux from the center.
        component_kwargs: dict
            Keyword arguments to pass to the component initialization.
        """
        self.symmetric = symmetric
        self.monotonic = monotonic
        self.coords = sky_coord
        center = frame.get_pixel(sky_coord)
        self.pixel_center = center
        self.center_step = center_step
        self.delay_thresh = delay_thresh

        sed, morph = init_combined_extended_source(sky_coord, frame, observations, bg_rms, obs_idx,
                                                   thresh, True, monotonic)

        Component.__init__(self, frame, sed, morph, **component_kwargs)


class MultiComponentSource(ComponentTree):
    """Extended source with multiple components layered vertically.

    Uses `~scarlet.source.ExtendedSource` to define the overall morphology,
    then erodes the outer footprint until it reaches the specified size percentile.
    For the narrower footprint, it evaluates the mean value at the perimeter and
    sets the inside to the perimeter value, creating a flat distribution inside.
    The subsequent component(s) is/are set to the difference between the flattened
    and the overall morphology.
    The SED for all components is calculated as the best fit of the multi-component
    morphology to the multi-channel image in the region of the source.
    """
    def __init__(self, frame, sky_coord, observation, bg_rms, thresh=1, flux_percentiles=None,
                 symmetric=True, monotonic=True, **component_kwargs):
        """Create multi-component extended source.

        Parameters
        ----------
        frame: `~scarlet.Frame`
            The frame that the model lives in.
        sky_coord: tuple
            Center of the source
        observation: `~scarlet.Observation`
            Observation to initialize this source.
        bg_rms: array
            Background RMS in each channel in observation.
        thresh: `float`
            Multiple of the backround RMS used as a
            flux cutoff for morphology initialization.
        flux_percentiles: list
            The flux percentile of each component. If `flux_percentiles` is `None`
            then `flux_percentiles=[25]`, a single component with 25% of the flux
            as the primary source.
        symmetric: `bool`
            Whether or not to enforce symmetry.
        monotonic: `bool`
            Whether or not to make the object monotonically decrease
            in flux from the center.
        component_kwargs: dict
            Keyword arguments to pass to the component initialization.
        """
        seds, morphs = init_multicomponent_source(sky_coord, frame, observation, bg_rms, flux_percentiles,
                                                  thresh, symmetric, monotonic)

        class MultiComponent(Component):
            def __init__(self, frame, sed, morph, symmetric, monotonic, **kwargs):
                self.symmetric = symmetric
                self.monotonic = monotonic
                self.pixel_center = frame.get_pixel(sky_coord)
                super().__init__(frame, sed, morph, **kwargs)

            def update(self):
                if self.symmetric:
                    update.symmetric_fit_center(self)  # update the center position
                    center = self.coords
                    update.symmetric(self, center)  # make the morph perfectly symmetric
                elif self.monotonic:
                    update.fit_pixel_center(self)
                    center = self.pixel_center
                if self.monotonic:
                    update.monotonic(self, center)  # make the morph monotonically decreasing
                update.positive(self)  # Make the SED and morph non-negative
                update.normalized(self, type='morph_max')
                return self

        components = [
            MultiComponent(frame, seds[k], morphs[k], symmetric, monotonic, **component_kwargs)
            for k in range(len(seds))
        ]
        super().__init__(components)

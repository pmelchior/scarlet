import numpy as np

from .component import Component, ComponentTree
from . import operator
from . import update

import logging
logger = logging.getLogger("scarlet.source")


class SourceInitError(Exception):
    """Error during source initialization
    """
    pass


def get_pixel_sed(sky_coord, scene, observations):
    """Get the SED at `position` in `img`

    Parameters
    ----------
    sky_coord: tuple
        Center of the source
    scene: `scarlet.observation.Scene`
        The scene that the model lives in.
    observations: list of `~scarlet.observation.Observation`
        Observations to extract SED from.

    Returns
    -------
    SED: `~numpy.array`
        SED for a single source
    """
    sed = np.zeros(scene.B, dtype=observations[0].images.dtype)
    band = 0
    for obs in observations:
        pixel = obs.get_pixel(sky_coord)
        sed[band:band+obs.B] = obs.images[:, pixel[0], pixel[1]]
        band += obs.B
    if np.all(sed <= 0):
        # If the flux in all bands is  <=0,
        # the new sed will be filled with NaN values,
        # which will cause the code to crash later
        msg = "Zero or negative flux at y={0}, x={1}"
        raise SourceInitError(msg.format(*sky_coord))
    return sed


def get_best_fit_seds(morphs, scene, observations):
    """Calculate best fitting SED for multiple components.

    Solves min_A ||img - AS||^2 for the SED matrix A,
    assuming that the images only contain a single source.

    Parameters
    ----------
    morphs: list
        Morphology for each component in the source.
    scene: `scarlet.observation.Scene`
        The scene that the model lives in.
    observations: list of `~scarlet.observation.Observation`
        Observations to extract SEDs from.
    """
    K = len(morphs)
    seds = np.zeros((K, scene.B), dtype=observations[0].images.dtype)
    band = 0
    _morph = morphs.reshape(K, -1)
    for obs in observations:
        images = obs.get_scene(scene).detach().numpy()
        data = images.reshape(obs.B, -1)
        sed = np.dot(np.linalg.inv(np.dot(_morph, _morph.T)), np.dot(_morph, data.T))
        seds[:, band:band+obs.B] = sed
    return seds

def build_detection_coadd(sed, bg_rms, observation, scene, thresh=1):
    """Build a band weighted coadd to use for source detection

    Parameters
    ----------
    sed: array
        SED at the center of the source.
    bg_rms: array
        Background RMS in each band in observation.
    observation: `~scarlet.observation.Observation`
        Observation to use for the coadd.
    scene: `scarlet.observation.Scene`
        The scene that the model lives in.
    thresh: `float`
        Multiple of the backround RMS used as a
        flux cutoff.

    Returns
    -------
    detect: array
        2D image created by weighting all of the bands by SED
    bg_cutoff: float
        The minimum value in `detect` to include in detection.
    """
    B = observation.B
    images = observation.get_scene(scene)
    weights = np.array([sed[b]/bg_rms[b]**2 for b in range(B)])
    jacobian = np.array([sed[b]**2/bg_rms[b]**2 for b in range(B)]).sum()
    detect = np.einsum('i,i...', weights, images) / jacobian

    # thresh is multiple above the rms of detect (weighted variance across bands)
    bg_cutoff = thresh * np.sqrt((weights**2 * bg_rms**2).sum()) / jacobian
    return detect, bg_cutoff


def init_extended_source(sky_coord, scene, observations, bg_rms, obs_idx=0,
                         thresh=1., symmetric=True, monotonic=True):
    """Initialize the source that is symmetric and monotonic
    See `ExtendedSource` for a description of the parameters
    """
    try:
        iter(observations)
    except TypeError:
        observations = [observations]
    # determine initial SED from peak position
    sed = get_pixel_sed(sky_coord, scene, observations)  # amplitude is in sed
    morph, bg_cutoff = build_detection_coadd(sed, bg_rms, observations[obs_idx], scene, thresh)
    center = scene.get_pixel(sky_coord)

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
    cy, cx = center
    center_morph = morph[cy, cx]
    morph /= center_morph
    return sed, morph


def init_multicomponent_source(sky_coord, scene, observations, bg_rms, flux_percentiles=None, obs_idx=0,
                               thresh=1., symmetric=True, monotonic=True):
    """Initialize multiple components
    See `MultiComponentSource` for a description of the parameters
    """
    try:
        iter(observations)
    except TypeError:
        observations = [observations]

    if flux_percentiles is None:
        flux_percentiles = [25]
    # Initialize the first component as an extended source
    sed, morph = init_extended_source(sky_coord, scene, observations, bg_rms, obs_idx,
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
        perc = percentiles_[k-1]
        flux_thresh = perc*max_flux/100
        mask_ = morph > flux_thresh
        morphs[k-1][mask_] = flux_thresh - last_thresh
        morphs[k][mask_] = morph[mask_] - flux_thresh
        last_thresh = flux_thresh

    # renormalize morphs: initially Smax
    for k in range(K):
        morphs[k] /= morphs[k].max()

    # optimal SEDs given the morphologies, assuming img only has that source
    seds = get_best_fit_seds(morphs, scene, observations)

    return seds, morphs


class PointSource(Component):
    """Extended source intialized with a single pixel

    Point sources are initialized with the SED of the center pixel,
    and the morphology of a single pixel (the center) turned on.
    While the source can have any `constraints`, the default constraints are
    symmetry and monotonicity.

    Parameters
    ----------
    sky_coord: tuple
        Center of the source
    scene: `scarlet.observation.Scene`
        The scene that the model lives in.
    observations: list of `~scarlet.observation.Observation`
        Observations to extract SED from.
    component_kwargs: dict
        Keyword arguments to pass to the component initialization.
    """
    def __init__(self, sky_coord, scene, observations, symmetric=True, monotonic=True, **component_kwargs):
        try:
            iter(observations)
        except TypeError:
            observations = [observations]
        # this ignores any broadening from the PSFs ...
        B, Ny, Nx = scene.shape
        morph = np.zeros((Ny, Nx), observations[0].images.dtype)
        pixel = scene.get_pixel(sky_coord)
        morph[pixel] = 1
        self.pixel_center = pixel

        sed = np.zeros((B,), observations[0].images.dtype)
        b0 = 0
        for obs in observations:
            pixel = obs.get_pixel(sky_coord)
            sed[b0:b0+obs.B] = obs.images[:, pixel[0], pixel[1]]
            b0 += obs.B

        super().__init__(sed, morph, **component_kwargs)
        self.symmetric = symmetric
        self.monotonic = monotonic

    def update(self):
        """Default update parameters for an ExtendedSource

        This method can be overwritten if a different set of constraints
        or update functions is desired.
        """
        if self.symmetric:
            # TODO implement improved symmetry method
            # update the center position
            #update.symmetric_fit_center(self)
            # make the morphology perfectly symmetric
            #update.symmetric(self, self.float_center)
            update.fit_pixel_center(self)
            update.symmetric(self, self.pixel_center)
        elif self.monotonic:
            # update the center position
            update.fit_pixel_center(self)
        if self.monotonic:
            # make the morphology monotonically decreasing
            update.monotonic(self, self.pixel_center)
        update.positive(self)  # Make the SED and morph non-negative
        update.normalized(self, type='morph_max')
        return self


class ExtendedSource(PointSource):
    """Extended source intialized to match a set of observations

    Parameters
    ----------
    sky_coord: tuple
        Center of the source
    scene: `scarlet.observation.Scene`
        The scene that the model lives in.
    observations: list of `~scarlet.observation.Observation`
        Observations to extract SED from.
    bg_rms: array
        Background RMS in each band in observation.
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
    def __init__(self, sky_coord, scene, observations, bg_rms, obs_idx=0, thresh=1,
                 symmetric=True, monotonic=True, **component_kwargs):
        self.symmetric = symmetric
        self.monotonic = monotonic
        self.coords = sky_coord
        center = scene.get_pixel(sky_coord)
        self.pixel_center = center

        sed, morph = init_extended_source(sky_coord, scene, observations, bg_rms, obs_idx,
                                          thresh, symmetric, monotonic)

        Component.__init__(self, sed, morph, **component_kwargs)


class MultiComponentSource(ComponentTree):
    """Create an extended source with multiple components layered vertically.
    Uses `~scarlet.source.ExtendedSource` to define the overall morphology,
    then erodes the outer footprint until it reaches the specified size percentile.
    For the narrower footprint, it evaluates the mean value at the perimeter and
    sets the inside to the perimeter value, creating a flat distribution inside.
    The subsequent component(s) is/are set to the difference between the flattened
    and the overall morphology.
    The SED for all components is calculated as the best fit of the multi-component
    morphology to the multi-band image in the region of the source.

    Parameters
    ----------
    flux_percentiles: list
        The flux percentile of each component. If `flux_percentiles` is `None`
        then `flux_percentiles=[25]`, a single component with 25% of the flux
        as the primary source.

    See `ExtendedSource` for a description of the parameters
    """
    def __init__(self, sky_coord, scene, observations, bg_rms, flux_percentiles=None, obs_idx=0, thresh=1,
                 symmetric=True, monotonic=True, **component_kwargs):
        seds, morphs = init_multicomponent_source(sky_coord, scene, observations, bg_rms, flux_percentiles,
                                                  obs_idx, thresh, symmetric, monotonic)

        class MultiComponent(Component):
            def __init__(self, sed, morph, symmetric, monotonic, **kwargs):
                self.symmetric = symmetric
                self.monotonic = monotonic
                super().__init__(sed, morph, **kwargs)

            def update(self):
                if self.symmetric:
                    update.symmetric_fit_center(self.morph)  # update the center position
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
            MultiComponent(seds[k], morphs[k], symmetric, monotonic, **component_kwargs)
            for k in range(len(seds))
        ]
        super().__init__(components)

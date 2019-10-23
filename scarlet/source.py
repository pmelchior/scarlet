import autograd.numpy as np
import logging, proxmin
logger = logging.getLogger("scarlet.source")

from . import operator
from . import update
from .component import *
from .interpolation import get_projection_slices


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
        The frame of the model
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
        morph = operator.prox_uncentered_symmetry(morph, 0, center=center, algorithm="sdss")
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
        morph = operator.prox_uncentered_symmetry(morph, 0, center=center, algorithm="sdss")

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

default_step = lambda X, it: 0.1*X.mean(axis=0)

class RandomSource(FactorizedComponent):
    """Sources with uniform random morphology.

    For cases with no well-defined spatial shape, this source initializes
    a uniform random field and (optionally) matches the SED to match a given
    observation.
    """
    def __init__(self, frame, observation=None):
        """Source intialized with a single pixel

        Parameters
        ----------
        frame: `~scarlet.Frame`
            The frame of the model
        observation: list of `~scarlet.Observation`
            Observation to initialize the SED of the source
        """
        C, Ny, Nx = frame.shape
        morph = np.random.rand(Ny, Nx)

        if observation is None:
            sed = np.random.rand(C)
        else:
            sed = get_best_fit_seds(morph[None], frame, observation)[0]

        sed = Parameter(sed, name="sed", step=default_step, constraint=proxmin.operators.prox_plus)
        morph = Parameter(morph, name="morph", step=default_step, constraint=proxmin.operators.prox_plus)

        super().__init__(frame, sed, morph)


def gauss_func(yc, xc, sigma=1, x=None, y=None):
    amplitude = 1/(np.pi**2*sigma**2)
    return amplitude * np.exp(-(yc-y)**2/(2*sigma**2))[:,None] * np.exp(-(xc-x)**2/(2*sigma**2))[None,:]

class PointSource(FunctionComponent):
    """Source intialized with a single pixel

    Point sources are initialized with the SED of the center pixel,
    and the morphology taken from `frame.psfs`, centered at `sky_coord`.
    """
    def __init__(self, frame, sky_coord, observation, func):
        """Source intialized with a single pixel

        Parameters
        ----------
        frame: `~scarlet.Frame`
            The frame of the model
        sky_coord: tuple
            Center of the source
        observation: list of `~scarlet.Observation`
            Observation to initialize this source
        """
        C, Ny, Nx = frame.shape
        self.pixel_center = np.array(frame.get_pixel(sky_coord), dtype=np.float)

        pixel = observation.frame.get_pixel(sky_coord)
        sed = observation.images[:, pixel[0], pixel[1]].copy()
        if observation.frame.psfs is not None:
            # Account for the PSF in the intensity
            sed /= observation.frame.psfs.max(axis=(1, 2))

        sed = Parameter(sed, name="sed", step=default_step, constraint=proxmin.operators.prox_plus)
        center = Parameter(self.pixel_center, step=lambda *X, it: 1e-3)

        super().__init__(frame, sed, center, func)


class ExtendedSource(PointSource):
    def __init__(self, frame, sky_coord, observation, bg_rms, thresh=1,
                 symmetric=True, monotonic=True, center_step=5, delay_thresh=10):
        """Extended source intialized to match a set of observations

        Parameters
        ----------
        frame: `~scarlet.Frame`
            The frame of the model
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
        sed = Parameter(sed, name="sed")
        morph = Parameter(morph, name="morph")
        Component.__init__(self, frame, sed, morph)
        self.update()


class CombinedExtendedSource(PointSource):
    def __init__(self, frame, sky_coord, observations, bg_rms, obs_idx=0, thresh=1,
                 symmetric=False, monotonic=True, center_step=5, delay_thresh=0):
        """Extended source intialized to match a set of observations

        Parameters
        ----------
        frame: `~scarlet.Frame`
            The frame of the model
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
        sed = Parameter(sed, name="sed")
        morph = Parameter(morph, name="morph")
        Component.__init__(self, frame, sed, morph)


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
                 symmetric=True, monotonic=True):
        """Create multi-component extended source.

        Parameters
        ----------
        frame: `~scarlet.Frame`
            The frame of the model
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
        """
        seds, morphs = init_multicomponent_source(sky_coord, frame, observation, bg_rms, flux_percentiles,
                                                  thresh, symmetric, monotonic)

        class MultiComponent(Component):
            def __init__(self, frame, sed, morph, symmetric, monotonic):
                self.symmetric = symmetric
                self.monotonic = monotonic
                self.pixel_center = frame.get_pixel(sky_coord)
                sed = Parameter(sed, name="sed")
                morph = Parameter(morph, name="morph")
                super().__init__(frame, sed, morph)

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
            MultiComponent(frame, seds[k], morphs[k], symmetric, monotonic)
            for k in range(len(seds))
        ]
        super().__init__(components)

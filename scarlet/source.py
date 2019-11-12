import proxmin
from .constraint import *
from .component import *
from .bbox import *
from . import operator
from . import measurement

# make sure that import * above doesn't import its own stock numpy
import autograd.numpy as np

import logging
logger = logging.getLogger("scarlet.source")


def get_pixel_sed(sky_coord, observation):
    """Get the SED at `sky_coord` in `observation`

    Parameters
    ----------
    sky_coord: tuple
        Position in the observation
    observation: `~scarlet.Observation`
        Observation to extract SED from.

    Returns
    -------
    SED: `~numpy.array`
    """

    pixel = observation.frame.get_pixel(sky_coord)
    sed = observation.images[:, pixel[0], pixel[1]].copy()
    return sed


def get_psf_sed(sky_coord, observation, frame):
    """Get SED for a point source at `sky_coord` in `observation`

    Identical to `get_pixel_sed`, but corrects for the different
    peak values of the observed seds to approximately correct for PSF
    width variations between channels.

    Parameters
    ----------
    sky_coord: tuple
        Position in the observation
    observation: `~scarlet.Observation`
        Observation to extract SED from.
    frame: `~scarlet.Frame`
        Frame of the model

    Returns
    -------
    SED: `~numpy.array`
    """
    sed = get_pixel_sed(sky_coord, observation)

    # approx. correct PSF width variations from SED by normalizing heights
    if observation.frame.psf is not None:
        # Account for the PSF in the intensity
        sed /= observation.frame.psf.image.max(axis=(1, 2))

    if frame.psf is not None:
        sed = sed * frame.psf.image[0].max()

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

    Returns
    -------
    SED: `~numpy.array`
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

    positive = [ c for c in range(C) if sed[c] > 0 ]
    positive_img = [observation.images[c] for c in positive]
    positive_bgrms = np.array([bg_rms[c] for c in positive])
    weights = np.array([sed[c] / bg_rms[c] ** 2 for c in positive])
    jacobian = np.array([sed[c] ** 2 / bg_rms[c] ** 2 for c in positive]).sum()
    detect = np.einsum('i,i...', weights, positive_img) / jacobian

    # thresh is multiple above the rms of detect (weighted variance across channels)
    bg_cutoff = thresh * np.sqrt((weights ** 2 * positive_bgrms ** 2).sum()) / jacobian
    return detect, bg_cutoff


def init_extended_source(sky_coord, frame, observations, obs_idx=0,
                                  thresh=1, symmetric=True, monotonic=True):
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
        _sed = get_psf_sed(sky_coord, obs, frame)
        seds.append(_sed)
    sed = np.concatenate(seds).flatten()

    if np.all(sed <= 0):
        # If the flux in all channels is  <=0,
        msg = "Zero or negative SED {} at y={}, x={}".format(sed, *sky_coord)
        logger.warning(msg)

    # which observation to use for detection and morphology
    obs_ = observations[obs_idx]
    try:
        bg_rms = np.array([ 1 / np.sqrt(w[w > 0].mean()) for w in obs_.weights])
    except:
        raise AttributeError("Observation.weights missing! Please set inverse variance weights")
    morph, bg_cutoff = build_detection_coadd(seds[obs_idx], bg_rms, obs_, thresh)

    # Apply the necessary constraints
    center = frame.get_pixel(sky_coord)
    if symmetric:
        morph = operator.prox_uncentered_symmetry(morph, 0, center=center, algorithm="sdss")

    if monotonic:
        # use finite thresh to remove flat bridges
        prox_monotonic = operator.prox_strict_monotonic(morph.shape, use_nearest=False,
                                                        center=center, thresh=0.1)
        morph = prox_monotonic(morph, 0).reshape(morph.shape)

    # trim morph to pixels above threshold
    mask = morph > bg_cutoff
    if mask.sum() == 0:
        msg = "No flux above threshold={2} for source at y={0} x={1}".format(*center, bg_cutoff)
        logger.warning(msg)
    morph[~mask] = 0

    # normalize to unity at peak pixel
    cy, cx = np.array(center).astype(int)
    center_morph = morph[cy, cx]
    morph /= center_morph

    return sed, morph


def init_multicomponent_source(sky_coord, frame, observations, obs_idx=0, flux_percentiles=None,
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
    sed, morph = init_extended_source(sky_coord, frame, observations, obs_idx=obs_idx,
                                      thresh=thresh, symmetric=symmetric, monotonic=monotonic)
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
        if np.all(morphs[k] <= 0):
            msg = "Zero or negative morphology for component {} at y={}, x={}"
            logger.warning(msg.format(k, *skycoords))
        morphs[k] /= morphs[k].max()

    # optimal SEDs given the morphologies, assuming img only has that source
    seds = get_best_fit_seds(morphs, frame, observations[obs_idx])

    for k in range(K):
        if np.all(seds[k] <= 0):
            # If the flux in all channels is  <=0,
            # the new sed will be filled with NaN values,
            # which will cause the code to crash later
            msg = "Zero or negative SED {} for component {} at y={}, x={}".format(seds[k], k, *sky_coord)
            logger.warning(msg)

    return seds, morphs


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

        constraint = PositivityConstraint()
        sed = Parameter(sed, name="sed", step=default_step, constraint=constraint)
        morph = Parameter(morph, name="morph", step=default_step, constraint=constraint)

        super().__init__(frame, sed, morph)



class PointSource(FunctionComponent):
    """Source intialized with a single pixel

    Point sources are initialized with the SED of the center pixel,
    and the morphology taken from `frame.psfs`, centered at `sky_coord`.
    """
    def __init__(self, frame, sky_coord, observations):
        """Source intialized with a single pixel

        Parameters
        ----------
        frame: `~scarlet.Frame`
            The frame of the model
        sky_coord: tuple
            Center of the source
        observations: instance or list of `~scarlet.Observation`
            Observation(s) to initialize this source
        """
        C, Ny, Nx = frame.shape
        self.center = np.array(frame.get_pixel(sky_coord), dtype='float')

        # initialize SED from sky_coord
        try:
            iter(observations)
        except TypeError:
            observations = [observations]

        # determine initial SED from peak position
        # SED in the frame for source detection
        seds = []
        for obs in observations:
            _sed = get_psf_sed(sky_coord, obs, frame)
            seds.append(_sed)
        sed = np.concatenate(seds).flatten()

        if np.any(sed <= 0):
            # If the flux in all channels is  <=0,
            # the new sed will be filled with NaN values,
            # which will cause the code to crash later
            msg = "Zero or negative SED {} at y={}, x={}".format(sed, *sky_coord)
            if np.all(sed <= 0):
                logger.warning(msg)
            else:
                logger.info(msg)

        # set up parameters
        sed = Parameter(sed, name="sed", step=partial(relative_step, factor=1e-1), constraint=PositivityConstraint())
        center = Parameter(self.center, name="center", step=1e-1)

        _psf_wrapper = lambda *parameters: frame.psf.__call__(*parameters)[0]

        super().__init__(frame, sed, center, _psf_wrapper)



class ExtendedSource(FactorizedComponent):
    def __init__(self, frame, sky_coord, observations, obs_idx=0, thresh=3,
                 symmetric=True, monotonic=True, shifting=False):
        """Extended source intialized to match a set of observations

        Parameters
        ----------
        frame: `~scarlet.Frame`
            The frame of the model
        sky_coord: tuple
            Center of the source
        observations: instance or list of `~scarlet.observation.Observation`
            Observation(s) to initialize this source.
        obs_idx: int
            Index of the observation in `observations` to
            initialize the morphology.
        thresh: `float`
            Multiple of the backround RMS used as a
            flux cutoff for morphology initialization.
        symmetric: `bool`
            Whether or not to enforce symmetry.
        monotonic: `bool`
            Whether or not to make the object monotonically decrease
            in flux from the center.
        shifting: `bool`
            Whether or not a subpixel shift is added as optimization parameter
        """
        self.symmetric = symmetric
        self.monotonic = monotonic
        center = np.array(frame.get_pixel(sky_coord), dtype='float')
        self.pixel_center = tuple(np.round(center).astype('int'))

        if shifting:
            shift = Parameter(center - self.pixel_center, name="shift", step=1e-1)
        else:
            shift = None

        # initialize from observation
        sed, morph = init_extended_source(sky_coord, frame, observations,
            obs_idx=obs_idx,
            thresh=thresh,
            symmetric=True,
            monotonic=monotonic)

        sed = Parameter(sed, name="sed", step=partial(relative_step, factor=1e-2), constraint=PositivityConstraint())

        # trim the morphology
        bbox = Box.from_data(morph, min_value=0)
        size = 2 * max((self.pixel_center[0] - bbox.bottom, bbox.top - self.pixel_center[0],
                        self.pixel_center[1] - bbox.left, bbox.right - self.pixel_center[1]))
        boxsize = 8
        while boxsize < size:
            boxsize *= 2
        bottom = self.pixel_center[0] - boxsize // 2
        top = self.pixel_center[0] + boxsize // 2
        left = self.pixel_center[1] - boxsize // 2
        right = self.pixel_center[1] + boxsize // 2
        bbox = Box.from_bounds(bottom, top, left, right)
        morph = bbox.image_to_box(morph)

        constraints = []
        if monotonic:
            # most astronomical sources are monotonically decreasing
            # from their center
            constraints.append(MonotonicityConstraint())
        if symmetric:
            # have 2-fold rotation symmetry around their center ...
            constraints.append(SymmetryConstraint())

        constraints += [
            # ... and are positive emitters
            PositivityConstraint(),
            # prevent a weak source from disappearing entirely
            #CenterOnConstraint(),
            # break degeneracies between sed and morphology
            NormalizationConstraint("max")
        ]
        morph_constraint = ConstraintChain(*constraints)

        morph = Parameter(morph, name="morph", step=1e-2, constraint=morph_constraint)

        super().__init__(frame, sed, morph, bbox=bbox, shift=shift)

    @property
    def center(self):
        if len(self.parameters) == 3:
            return self.pixel_center + self.shift
        else:
            return self.pixel_center

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

    def __init__(self, frame, sky_coord, observations, obs_idx=0, thresh=0.1, flux_percentiles=None,
                 symmetric=True, monotonic=True, shifting=False):
        """Create multi-component extended source.

        Parameters
        ----------
        frame: `~scarlet.Frame`
            The frame of the model
        sky_coord: tuple
            Center of the source
        observations: instance or list of `~scarlet.observation.Observation`
            Observation(s) to initialize this source.
        obs_idx: int
            Index of the observation in `observations` to
            initialize the morphology.
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
        shifting: `bool`
            Whether or not a subpixel shift is added as optimization parameter
        """
        self.symmetric = symmetric
        self.monotonic = monotonic
        self.coords = sky_coord
        center = np.array(frame.get_pixel(sky_coord), dtype='float')
        self.pixel_center = tuple(np.round(center).astype('int'))

        if shifting:
            shift = Parameter(center - self.pixel_center, name="shift", step=1e-1)
        else:
            shift = None

        # initialize from observation
        seds, morphs = init_multicomponent_source(sky_coord, frame, observations,
            obs_idx=obs_idx,
            flux_percentiles=flux_percentiles,
            thresh=thresh,
            symmetric=symmetric,
            monotonic=monotonic)

        # trim the morphology
        bbox = Box.from_data(morphs[0], min_value=0)
        size = 2 * max((self.pixel_center[0] - bbox.bottom, bbox.top - self.pixel_center[0],
                        self.pixel_center[1] - bbox.left, bbox.right - self.pixel_center[1]))
        boxsize = 8
        while boxsize < size:
            boxsize *= 2
        bottom = self.pixel_center[0] - boxsize // 2
        top = self.pixel_center[0] + boxsize // 2
        left = self.pixel_center[1] - boxsize // 2
        right = self.pixel_center[1] + boxsize // 2
        bbox = Box.from_bounds(bottom, top, left, right)
        morphs_ = []
        for k in range(len(morphs)):
            morphs_.append(bbox.image_to_box(morphs[k]))

        constraints = []
        if monotonic:
            # most astronomical sources are monotonically decreasing
            # from their center
            constraints.append(MonotonicityConstraint())
        if symmetric:
            # have 2-fold rotation symmetry around their center ...
            constraints.append(SymmetryConstraint())
        constraints += [
            # ... and are positive emitters
            PositivityConstraint(),
            # prevent a weak source from disappearing entirely
            CenterOnConstraint(),
            # break degeneracies between sed and morphology
            NormalizationConstraint("max")
        ]
        morph_constraint = ConstraintChain(*constraints)

        components = []
        for k in range(len(seds)):
            sed = Parameter(seds[k], name="sed", step=partial(relative_step, factor=1e-1), constraint=PositivityConstraint())
            morph = Parameter(morphs_[k], name="morph", step=1e-2, constraint=morph_constraint)
            components.append(FactorizedComponent(frame, sed, morph, bbox=bbox, shift=shift))
            components[-1].pixel_center = self.pixel_center
        super().__init__(components)

    @property
    def shift(self):
        c = self.components[0]
        return c.shift

    @property
    def center(self):
        c = self.components[0]
        if len(c.parameters) == 3:
            return self.pixel_center + c.shift
        else:
            return self.pixel_center

from .source import *
from .wavelet import Starlet, mad_wavelet
from .observation import Observation, LowResObservation
from .interpolation import interpolate_observation
from . import operator
from .bbox import Box

import numpy as np

import logging

logger = logging.getLogger("scarlet.initialisation")


def get_best_fit_seds(morphs, images):
    """Calculate best fitting SED for multiple components.

    Solves min_A ||img - AS||^2 for the SED matrix A,
    assuming that the images only contain a single source.

    Parameters
    ----------
    morphs: list
        Morphology for each component in the source.
    frame: `scarlet.observation.frame`
        The frame of the model
    images: array
        Observation to extract SEDs from.

    Returns
    -------
    SED: `~numpy.array`
    """
    K = len(morphs)
    _morph = morphs.reshape(K, -1)
    if isinstance(images, np.ndarray):
        images_ = images
    elif isinstance(images, Observation):
        images_ = images.images
    elif hasattr(images, "__iter__") and all(
        tuple(obs.frame == images[0].frame for obs in images)
    ):
        # all observations need to have the same frame for this mapping to work
        images_ = np.stack(tuple(obs.images for obs in images), axis=0)

    data = images_.reshape(images_.shape[0], -1)
    seds = np.dot(np.linalg.inv(np.dot(_morph, _morph.T)), np.dot(_morph, data.T))
    return seds


def get_pixel_sed(sky_coord, observations):
    """Get the SED at `sky_coord` in `observation`

    Parameters
    ----------
    sky_coord: tuple
        Position in the observation
    observations: `~scarlet.Observation`
        Observation to extract SED from.

    Returns
    -------
    SED: `~numpy.array`
    """

    if not hasattr(observations, "__iter__"):
        observations = (observations,)

    # determine initial SED from peak position
    # SED in the frame for source detection
    seds = []
    for obs in observations:
        pixel = obs.frame.get_pixel(sky_coord)
        sed = obs.images[:, pixel[0], pixel[1]].copy()
        seds.append(sed)
    sed = np.concatenate(seds).reshape(-1)

    if np.any(sed <= 0):
        # If the flux in all channels is  <=0,
        # the new sed will be filled with NaN values,
        # which will cause the code to crash later
        msg = "Zero or negative SED {} at y={}, x={}".format(sed, *sky_coord)
        if np.all(sed <= 0):
            logger.warning(msg)
        else:
            logger.info(msg)

    return sed


def get_psf_sed(sky_coord, observations, model_frame):
    """Get SED for a point source at `sky_coord` in `observation`

    Identical to `get_pixel_sed`, but corrects for the different
    peak values of the observed seds to approximately correct for PSF
    width variations between channels.

    Parameters
    ----------
    sky_coord: tuple
        Position in the observation
    observations: `~scarlet.Observation`
        Observation to extract SED from.
    model_frame: `~scarlet.Frame`
        Frame of the model

    Returns
    -------
    SED: `~numpy.array`
    """

    if not hasattr(observations, "__iter__"):
        observations = (observations,)

    seds = []
    for obs in observations:
        sed = get_pixel_sed(sky_coord, obs)

        if type(obs) is LowResObservation:
            normalization = "sum"
        else:
            normalization = "max"

        # approx. correct PSF width variations from SED by normalizing heights
        if normalization is "sum":
            if obs._diff_kernels is not None:
                sed /= obs._diff_kernels.image.sum(axis=(-2, -1)) * obs.h ** 2
        else:
            if obs.frame.psf is not None:
                # Account for the PSF in the intensity
                sed /= obs.frame.psf.image.max(axis=(-2, -1))

            if model_frame.psf is not None:
                sed *= model_frame.psf.image[0].max()

        seds.append(sed)

    sed = np.concatenate(seds).reshape(-1)
    return sed


def trim_morphology(sky_coord, frame, morph, bg_cutoff, thresh):
    # trim morph to pixels above threshold
    mask = morph > bg_cutoff * thresh
    boxsize = 16
    pixel_center = frame.get_pixel(sky_coord)
    if mask.sum() > 0:
        morph[~mask] = 0

        # normalize to unity at peak pixel
        center_morph = morph[pixel_center[0], pixel_center[1]]
        morph /= center_morph

        # find fitting bbox
        bbox = Box.from_data(morph, min_value=0)
        if bbox.contains(pixel_center):
            size = 2 * max(
                (
                    pixel_center[0] - bbox.start[-2],
                    bbox.stop[0] - pixel_center[-2],
                    pixel_center[1] - bbox.start[-1],
                    bbox.stop[1] - pixel_center[-1],
                )
            )
            while boxsize < size:
                boxsize *= 2
    else:
        msg = "No flux above threshold for source at y={0} x={1}".format(*pixel_center)
        logger.warning(msg)

    # define bbox and trim to bbox
    bottom = pixel_center[0] - boxsize // 2
    top = pixel_center[0] + boxsize // 2
    left = pixel_center[1] - boxsize // 2
    right = pixel_center[1] + boxsize // 2
    bbox = Box.from_bounds((bottom, top), (left, right))
    morph = bbox.extract_from(morph)
    bbox_3d = Box.from_bounds((0, frame.C), (bottom, top), (left, right))
    return morph, bbox_3d


def init_extended_source(
    sky_coord,
    frame,
    observations,
    coadd=None,
    bg_cutoff=None,
    thresh=1,
    symmetric=True,
    monotonic="flat",
    min_grad=0.1,
):
    """Initialize the source that is symmetric and monotonic
    See `ExtendedSource` for a description of the parameters
    """
    if not hasattr(observations, "__iter__"):
        observations = (observations,)

    # determine initial SED from peak position
    # SED in the frame for source detection
    seds = [get_psf_sed(sky_coord, obs, frame) for obs in observations]
    sed = np.concatenate(seds).reshape(-1)

    if coadd is None:
        # which observation to use for detection and morphology
        try:
            bg_rmses = np.array(
                [
                    [1 / np.sqrt(w[w > 0].mean()) for w in obs.weights]
                    for obs in observations
                ]
            )
        except:
            raise AttributeError(
                "Observation.weights missing! Please set inverse variance weights"
            )
        coadd, bg_cutoff = build_sed_coadd(seds, bg_rmses, observations)
    else:
        if bg_cutoff is None:
            raise AttributeError(
                "background cutoff missing! Please set argument bg_cutoff"
            )
    # Apply the necessary constraints
    center = frame.get_pixel(sky_coord)
    if symmetric:
        morph = operator.prox_uncentered_symmetry(
            coadd.copy(),
            0,
            center=center,
            algorithm="sdss",  # *1 is to artificially pass a variable that is not coadd
        )
    else:
        morph = coadd
    if monotonic:
        if monotonic is True:
            monotonic = "angle"
        # use finite thresh to remove flat bridges
        prox_monotonic = operator.prox_weighted_monotonic(
            morph.shape, neighbor_weight=monotonic, center=center, min_gradient=min_grad
        )
        morph = prox_monotonic(morph, 0).reshape(morph.shape)

    morph, bbox = trim_morphology(sky_coord, frame, morph, bg_cutoff, thresh)
    return sed, morph, bbox


def init_starlet_source(
    sky_coord,
    model_frame,
    observations,
    coadd=None,
    bg_cutoff=None,
    thresh=1,
    symmetric=True,
    monotonic="flat",
    min_grad=0.1,
    starlet_thresh=5,
):

    # initialize as extended from observation
    if not hasattr(observations, "__iter__"):
        observations = (observations,)

    sed, morph, bbox = init_extended_source(
        sky_coord,
        model_frame,
        observations,
        coadd=coadd,
        bg_cutoff=bg_cutoff,
        thresh=thresh,
        symmetric=True,
        monotonic=True,
        min_grad=min_grad,
    )

    noise = []
    for obs in observations:
        noise += [
            mad_wavelet(obs.images)
            * np.sqrt(np.sum(obs._diff_kernels.image ** 2, axis=(-2, -1)))
        ]
    noise = np.concatenate(noise)

    # Threshold in units of noise on the coadd
    thresh = starlet_thresh * np.sqrt(np.sum((sed * noise) ** 2))
    return sed, morph, bbox, thresh


def init_multicomponent_source(
    sky_coord,
    frame,
    observations,
    coadd=None,
    bg_cutoff=None,
    flux_percentiles=None,
    thresh=1,
    symmetric=True,
    monotonic="flat",
    min_grad=0.1,
    obs_ref=None,
):
    """Initialize multiple components
    See `MultiComponentSource` for a description of the parameters
    """
    try:
        iter(observations)
    except TypeError:
        observations = [observations]

    if obs_ref is None:
        if len(observations) == 1:
            obs_ref = observations[0]
        else:
            # The observation that lives in the same plane as the frame
            loc = np.where([type(obs) is Observation for obs in observations])
            # If more than one element is an `Observation`, then pick the first one as a reference (arbitrary)
            obs_ref = observations[loc[0]]

    if flux_percentiles is None:
        flux_percentiles = [25]

    # Initialize the first component as an extended source
    sed, morph, bbox = init_extended_source(
        sky_coord,
        frame,
        observations,
        coadd=coadd,
        bg_cutoff=bg_cutoff,
        thresh=thresh,
        symmetric=symmetric,
        monotonic=monotonic,
        min_grad=min_grad,
    )
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
            logger.warning(msg.format(k, *sky_coord))
        morphs[k] /= morphs[k].max()

    # optimal SEDs given the morphologies, assuming img only has that source
    boxed_img = bbox.extract_from(obs_ref.images)
    seds = get_best_fit_seds(morphs, boxed_img)

    for k in range(K):
        if np.all(seds[k] <= 0):
            # If the flux in all channels is  <=0,
            # the new sed will be filled with NaN values,
            # which will cause the code to crash later
            msg = "Zero or negative SED {} for component {} at y={}, x={}".format(
                seds[k], k, *sky_coord
            )
            logger.warning(msg)

    return seds, morphs, bbox


def build_sed_coadd(seds, bg_rmses, observations, obs_ref=None):
    """Build a channel weighted coadd to use for source detection
    Parameters
    ----------
    sed: array
        SED at the center of the source.
    bg_rms: array
        Background RMS in each channel in observation.
    observations: list of `~scarlet.observation.Observation`
        Observations to use for the coadd.
    obs_ref: `scarlet.Observation`
        observation to use as a reference frame.
        If set to None, the first (or only if applicable) element with type `Observation` is used.
    Returns
    -------
    detect: array
        2D image created by weighting all of the channels by SED
    bg_cutoff: float
        The minimum value in `detect` to include in detection.
    """
    if not hasattr(observations, "__iter__"):
        observations = (observations,)

    if len(observations) == 1:
        obs_ref = observations[0]

    # The observation that lives in the same plane as the frame
    if obs_ref is None:
        loc = np.where([type(obs) is Observation for obs in observations])
        obs_ref = observations[np.int(loc[0])]
    else:
        # The observation that lives in the same plane as the frame
        assert type(obs_ref) is not LowResObservation, (
            f"Reference observation should not be a `LowResObservation`. The observation, {obs_ref} "
            f"provided refers to an observation of type: {type(obs_ref)}"
        )

    positive_img = []
    positive_bgrms = []
    weights = []
    jacobian_args = []
    for i, obs in enumerate(observations):
        sed = seds[i]
        try:
            iter(sed)
        except TypeError:
            sed = [sed]
        C = len(sed)
        bg_rms = bg_rmses[i]
        try:
            iter(bg_rms)
        except TypeError:
            bg_rms = [bg_rms]
        if np.any(np.array(bg_rms) <= 0):
            raise ValueError("bg_rms must be greater than zero in all channels")

        positive = [c for c in range(C) if sed[c] > 0]
        if type(obs) is not LowResObservation:
            positive_img += [obs.images[c] for c in positive]
        else:
            positive_img += [
                interpolate_observation(obs, obs_ref.frame)[c] for c in positive
            ]
        positive_bgrms += [bg_rms[c] for c in positive]
        weights += [sed[c] / bg_rms[c] ** 2 for c in positive]
        jacobian_args += [sed[c] ** 2 / bg_rms[c] ** 2 for c in positive]

    detect = np.einsum("i,i...", np.array(weights), positive_img) / np.sum(
        jacobian_args
    )

    # thresh is multiple above the rms of detect (weighted variance across channels)
    bg_cutoff = np.sqrt(
        (np.array(weights) ** 2 * np.array(positive_bgrms) ** 2).sum()
    ) / np.sum(jacobian_args)
    return detect, bg_cutoff


def build_initialization_coadd(observations, filtered_coadd=False, obs_idx=None):
    """Build a channel weighted coadd to use for source detection

    For `LowResObservation`, images are interpolated to a reference frame

    Parameters
    ----------
    observations: `~scarlet.observation.Observation`
        Observation to use for the coadd.
    filtered_coadd: `bool`
        if set to True, images are filtered using wavelet filtering before interpolation/coadding
    obs_idx: `int`
        index of the observation in observations to use as a reference frame.
        If set to None, the first element with type `Observation` is used.

    Returns
    -------
    detect: array
        2D image created by weighting all of the channels by SED
    bg_cutoff: float
        The minimum value in `detect` to include in detection.
    """
    try:
        iter(observations)
    except TypeError:
        observations = [observations]

    if obs_idx is None:
        loc = np.where([type(obs) is Observation for obs in observations])
        obs_ref = observations[loc[0][0]]
    else:
        # The observation that lives in the same plane as the frame
        assert type(observations[obs_idx]) is Observation, (
            f"Reference observation should be an `Observation`. The observation index, {obs_idx} "
            f"provided refers to an observation of type: {type(observations[obs_idx])}"
        )
        # If more than one element is an `Observation`, then pick the first one as a reference (arbitrary)
        obs_ref = observations[obs_idx]

    coadd = 0
    jacobian = 0
    weights = 0
    for obs in observations:
        try:
            weights = np.array([w[w > 0].mean() for w in obs.weights])
        except:
            raise AttributeError(
                "Observation.weights missing! Please set inverse variance weights"
            )

        if obs is obs_ref:
            if filtered_coadd is True:
                star = Starlet(obs.images)
                # Sarlet filtering at 5 sigma
                star.filter()
                # Sets the last starlet scale to 0 to remove the wings of the profile introduced by psfs
                star.coefficients[:, -1, :, :] = 0
                # Positivity
                star.coefficients[star.coefficients < 0] = 0
                images = star.image
            else:
                images = obs.images
        else:
            # interpolate low-res to reference resolution
            images = interpolate_observation(
                obs, obs_ref.frame, wave_filter=filtered_coadd
            )
        if filtered_coadd is True:
            coadd += np.sum(
                images / np.sum(images, axis=(-2, -1))[:, None, None], axis=0
            )
        else:
            # Weighted coadd
            coadd += (images * weights[:, None, None]).sum(axis=(0))
            jacobian += weights.sum()

    if filtered_coadd is True:
        coadd /= np.max(coadd)
        bg_cutoff = 0.01
        return coadd, bg_cutoff
    coadd /= jacobian
    # thresh is multiple above the rms of detect (weighted variance across channels)
    bg_cutoff = np.sqrt((weights ** 2).sum()) / jacobian
    return coadd, bg_cutoff

import numpy as np

import logging

from . import operator
from .bbox import Box
from .constraint import CenterOnConstraint
from .interpolation import interpolate_observation
from .observation import Observation, LowResObservation
from .wavelet import Starlet, mad_wavelet


logger = logging.getLogger("scarlet.initialisation")


def get_best_fit_spectra(morphs, images):
    """Calculate best fitting spectra for multiple components.

    Solves min_A ||img - AS||^2 for the SED matrix A,
    assuming that the images only contain a single source.

    Parameters
    ----------
    morphs: list
        Morphology for each component in the source.
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


def get_pixel_spectrum(sky_coord, observations, correct_psf=False):
    """Get the spectrum at `sky_coord` in `observation`.

    Yields the spectrum of a single-pixel source with flux 1 in every channel,
    concatenated for all observations. If `correct_psf`, it homogenizes the PSFs of the
    observations, which yields the correct spectrum for a point source.

    Parameters
    ----------
    sky_coord: tuple
        Position in the observation
    observations: instance or list of `~scarlet.Observation`
        Observation to extract SED from.
    correct_psf: bool
        If PSF shape variations in the observations should be corrected.

    Returns
    -------
    spectrum: `~numpy.array`
    """

    if not hasattr(observations, "__iter__"):
        observations = (observations,)

    spectra = []
    for obs in observations:
        pixel = obs.frame.get_pixel(sky_coord)
        index = np.round(pixel).astype(np.int)
        spectrum = obs.images[:, index[0], index[1]].copy()

        if obs.frame.psf is not None and correct_psf:
            # image of point source in observed = obs.frame.psf
            psf_model = obs.frame.psf.get_model()
            psf_center = psf_model.max(axis=(1, 2))
            # best fit solution for the model amplitude of the center pixel
            # to yield to PSF center: (spectrum * psf_center) / psf_center**2
            # or shorter:
            spectrum /= psf_center

        spectra.append(spectrum)

    spectrum = np.concatenate(spectra).reshape(-1)

    if np.any(spectrum <= 0):
        # If the flux in all channels is  <=0,
        # the new sed will be filled with NaN values,
        # which will cause the code to crash later
        msg = "Zero or negative spectrum {} at y={}, x={}".format(spectrum, *sky_coord)
        if np.all(spectrum <= 0):
            logger.warning(msg)
        else:
            logger.info(msg)

    return spectrum


def get_psf_spectrum(sky_coord, observations):
    """Get spectrum for a point source at `sky_coord` in `observation`

    Equivalent to point source photometry for isolated sources. For extended source,
    this will underestimate the actual source flux in every channel. In case of crowding,
    the resulting photometry is likely contaminated by neighbors.

    Yields the spectrum of a PSF-homogenized source of flux 1 in every channel,
    concatenated for all observations.

    Parameters
    ----------
    sky_coord: tuple
        Position in the observation
    observations: instance or list of `~scarlet.Observation`
        Observation to extract SED from.

    Returns
    -------
    spectrum: `~numpy.array`
    """

    # assert normalization in ["max", "sum"]

    if not hasattr(observations, "__iter__"):
        observations = (observations,)

    spectra = []
    for obs in observations:

        pixel = obs.frame.get_pixel(sky_coord)
        index = np.round(pixel).astype(np.int)

        psf = obs.frame.psf.get_model()
        bbox = obs.frame.psf.bbox + (0, *index)
        img = bbox.extract_from(obs.images)

        # img now 0 outside of observation, psf is not:
        # restrict both to observed pixels to avoid truncation effects
        mask = img[0] > 0
        psf = psf[:, mask]  # flattens array in last two axes
        img = img[:, mask]

        spectrum = (img * psf).sum(axis=1) / (psf * psf).sum(axis=1)
        spectra.append(spectrum)

    spectrum = np.concatenate(spectra).reshape(-1)

    if np.any(spectrum <= 0):
        # If the flux in all channels is  <=0,
        # the new sed will be filled with NaN values,
        # which will cause the code to crash later
        msg = "Zero or negative spectrum {} at y={}, x={}".format(spectrum, *sky_coord)
        if np.all(spectrum <= 0):
            logger.warning(msg)
        else:
            logger.info(msg)

    return spectrum


def trim_morphology(center_index, morph, bg_thresh):
    # trim morph to pixels above threshold
    mask = morph > bg_thresh
    morph[~mask] = 0

    # find fitting bbox
    boxsize = 15
    bbox = Box.from_data(morph, min_value=0)
    if bbox.contains(center_index):
        size = 2 * max(
            (
                center_index[0] - bbox.start[-2],
                bbox.stop[0] - center_index[-2],
                center_index[1] - bbox.start[-1],
                bbox.stop[1] - center_index[-1],
            )
        )
        while boxsize < size:
            boxsize += 16  # keep box sizes quite small

    # define bbox and trim to bbox
    bottom = center_index[0] - boxsize // 2
    top = center_index[0] + boxsize // 2 + 1
    left = center_index[1] - boxsize // 2
    right = center_index[1] + boxsize // 2 + 1
    bbox = Box.from_bounds((bottom, top), (left, right))
    morph = bbox.extract_from(morph)
    bbox = Box.from_bounds((bottom, top), (left, right))
    return morph, bbox


def init_extended_source(
    sky_coord,
    frame,
    observations,
    coadd=None,
    coadd_rms=None,
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

    if coadd is None:
        # determine initial SED from peak position
        # don't correct for PSF variation: emphasize sharper bands
        spectra = [
            get_pixel_spectrum(sky_coord, obs, correct_psf=False)
            for obs in observations
        ]

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
        coadd, bg_rms = build_sed_coadd(spectra, bg_rmses, observations)
    else:
        if coadd_rms is None:
            raise AttributeError(
                "background cutoff missing! Please set argument bg_cutoff"
            )
        coadd = coadd.copy()  # will be reused by other sources
        bg_rms = coadd_rms

    # Apply the necessary constraints
    center = frame.get_pixel(sky_coord)
    center_index = np.round(center).astype(np.int)
    if symmetric:
        morph = operator.prox_uncentered_symmetry(
            coadd,
            0,
            center=center_index,
            algorithm="sdss",  # *1 is to artificially pass a variable that is not coadd
        )
    else:
        morph = coadd
    if monotonic:
        if monotonic is True:
            monotonic = "angle"
        # use finite thresh to remove flat bridges
        prox_monotonic = operator.prox_weighted_monotonic(
            morph.shape,
            neighbor_weight=monotonic,
            center=center_index,
            min_gradient=min_grad,
        )
        morph = prox_monotonic(morph, 0).reshape(morph.shape)

    # truncate morph at thresh * bg_rms
    threshold = bg_rms * thresh
    morph, bbox = trim_morphology(center_index, morph, threshold)
    bbox = frame.bbox[0] @ bbox

    # get PSF-corrected center pixel spectrum
    spectrum = get_pixel_spectrum(sky_coord, observations, correct_psf=True)

    # normalize to unity at peak pixel for the imposed normalization
    if morph.sum() > 0:
        morph /= morph.max()

        # since the spectrum assumes a point source:
        # determine the optimal amplitude for matching morph and the model psf
        # TODO: morph is still convolved with the observed PSF, but we compute
        # amplitude correction as if it were not..
        if frame.psf is not None:
            psf = frame.psf.get_model()

            shape = (psf.shape[0], *morph.shape)
            bbox_ = Box(
                shape,
                origin=(
                    psf.shape[0] - shape[0],
                    psf.shape[1] // 2 - shape[1] // 2,
                    psf.shape[2] // 2 - shape[2] // 2,
                ),
            )
            psf = bbox_.extract_from(psf)

            # if model PSF is constant across bands (as it should) then factor
            # is constant as well
            factor = (morph[None, :, :] * psf).sum(axis=(1, 2)) / (psf * psf).sum(
                axis=(1, 2)
            )

            # correct amplitude from point source to this morph
            spectrum /= factor

    else:
        morph = CenterOnConstraint()(morph, 0)
        msg = "No flux in morphology model for source at y={0} x={1}".format(*sky_coord)
        logger.warning(msg)

    return spectrum, morph, bbox


def init_starlet_source(
    sky_coord,
    model_frame,
    observations,
    coadd=None,
    coadd_rms=None,
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
        coadd_rms=coadd_rms,
        thresh=thresh,
        symmetric=symmetric,
        monotonic=monotonic,
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
    coadd_rms=None,
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
        coadd_rms=coadd_rms,
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
    spectra = get_best_fit_spectra(morphs, boxed_img)

    for k in range(K):
        if np.all(spectra[k] <= 0):
            # If the flux in all channels is  <=0,
            # the new sed will be filled with NaN values,
            # which will cause the code to crash later
            msg = "Zero or negative spectrum {} for component {} at y={}, x={}".format(
                spectra[k], k, *sky_coord
            )
            logger.warning(msg)

    # avoid using the same box for multiple components
    boxes = tuple(bbox.copy() for k in range(K))

    # # define minimal boxes (NOTE: dangerous due to box truncation)
    # morphs_ = []
    # boxes = []
    # threshold = 0
    # for k in range(K):
    #     morph, bbox = trim_morphology(sky_coord, frame, morphs[k], threshold)
    #     morphs_.append(morph)
    #     boxes.append(bbox)
    # morphs = morphs_

    return spectra, morphs, boxes


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

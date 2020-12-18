import numpy as np

import logging

from . import operator
from .bbox import Box, overlapped_slices
from .cache import Cache
from .constraint import CenterOnConstraint
from .interpolation import interpolate_observation
from .observation import Observation, LowResObservation
from .wavelet import Starlet, mad_wavelet
from . import fft
from . import measure


logger = logging.getLogger("scarlet.initialisation")


def get_best_fit_spectrum(morph, images):
    """Calculate best fitting spectra for one or multiple morphologies.

    Solves min_A ||img - AS||^2 for the spectrum matrix A,
    assuming that the images only contain a single source.

    Parameters
    ----------
    morph: array or list thereof
        Morphology for each component in the source
    images: array
        images to get the spectrum amplitude from

    Returns
    -------
    spectrum: `~numpy.array`
    """

    if isinstance(morph, (list, tuple)) or (
        isinstance(morph, np.ndarray) and len(morph.shape) == 3
    ):
        morphs = morph
    else:
        morphs = (morph,)

    K = len(morphs)
    C = images.shape[0]
    im = images.reshape(C, -1)

    if K == 1:
        morph = morphs[0].reshape(-1)
        return np.dot(im, morph) / np.dot(morph, morph)
    else:
        morph = np.array(morphs).reshape(K, -1)
        return np.dot(np.linalg.inv(np.dot(morph, morph.T)), np.dot(morph, im.T))


def get_pixel_spectrum(sky_coord, observations, correct_psf=False, models=None):
    """Get the spectrum at `sky_coord` in `observation`.

    Yields the spectrum of a single-pixel source with flux 1 in every channel,
    concatenated for all observations.

    If `correct_psf`, it homogenizes the PSFs of the observations, which yields the
    correct spectrum for a flux=1 point source.

    If `model` is set, it reads of the value of the model at `sky_coord` and yields the
    spectrum for that model.

    Parameters
    ----------
    sky_coord: tuple
        Position in the observation
    observations: instance or list of `~scarlet.Observation`
        Observation to extract SED from.
    correct_psf: bool
        If PSF shape variations in the observations should be corrected.
    models: instance or list of arrays
        Rendered models for this source in every observation

    Returns
    -------
    spectrum: `~numpy.array` or list thereof
    """
    if models is not None:
        assert correct_psf is False

    if not hasattr(observations, "__iter__"):
        single = True
        observations = (observations,)
        models = (models,)
    else:
        if models is not None:
            assert len(models) == len(observations)
        else:
            models = (None,) * len(observations)
        single = False

    spectra = []
    for obs, model in zip(observations, models):
        pixel = obs.frame.get_pixel(sky_coord)
        index = np.round(pixel).astype(np.int)
        spectrum = obs.images[:, index[0], index[1]].copy()

        if obs.frame.psf is not None:
            # correct spectrum for PSF-induced change in peak pixel intensity
            psf_model = obs.frame.psf.get_model()._data
            psf_peak = psf_model.max(axis=(1, 2))
            spectrum /= psf_peak
        elif model is not None:
            model_value = model[:, index[0], index[1]].copy()
            spectrum /= model_value

        spectra.append(spectrum)

        if np.any(spectrum <= 0):
            # If the flux in all channels is  <=0,
            # the new sed will be filled with NaN values,
            # which will cause the code to crash later
            msg = f"Zero or negative spectrum {spectrum} at {sky_coord}"
            if np.all(spectrum <= 0):
                logger.warning(msg)
            else:
                logger.info(msg)

    if single:
        return spectra[0]
    return spectra


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
    spectrum: ~numpy.array` or list thereof
    """

    if not hasattr(observations, "__iter__"):
        single = True
        observations = (observations,)
    else:
        single = False

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

        # amplitude of img when projected onto psf
        # i.e. factor to multiply psf with to get img (if img looked like psf)
        spectrum = (img * psf).sum(axis=1) / (psf * psf).sum(axis=1)
        spectra.append(spectrum)

        if np.any(spectrum <= 0):
            # If the flux in all channels is  <=0,
            # the new sed will be filled with NaN values,
            # which will cause the code to crash later
            msg = f"Zero or negative spectrum {spectrum} at {sky_coord}"
            if np.all(spectrum <= 0):
                logger.warning(msg)
            else:
                logger.info(msg)

    if single:
        return spectra[0]
    return spectra


def get_minimal_boxsize(size, min_size=21, increment=10):
    boxsize = min_size
    while boxsize < size:
        boxsize += increment  # keep box sizes quite small
    return boxsize


def trim_morphology(center_index, morph, bg_thresh=0):
    # trim morph to pixels above threshold
    mask = morph > bg_thresh
    morph[~mask] = 0

    bbox = Box.from_data(morph, min_value=0)

    # find fitting bbox
    if bbox.contains(center_index):
        size = 2 * max(
            (
                center_index[0] - bbox.start[-2],
                bbox.stop[0] - center_index[-2],
                center_index[1] - bbox.start[-1],
                bbox.stop[1] - center_index[-1],
            )
        )
    else:
        size = 0

    # define new box and cut morphology accordingly
    boxsize = get_minimal_boxsize(size)
    bottom = center_index[0] - boxsize // 2
    top = center_index[0] + boxsize // 2 + 1
    left = center_index[1] - boxsize // 2
    right = center_index[1] + boxsize // 2 + 1
    bbox = Box.from_bounds((bottom, top), (left, right))
    morph = bbox.extract_from(morph)
    return morph, bbox


def init_compact_source(sky_coord, frame):
    """Initialize a source just like `init_extended_morphology`,
    but with the morphology of a point source.
    """

    # position in frame coordinates
    center = frame.get_pixel(sky_coord)
    center_index = np.round(center).astype(np.int)

    # morphology initialized as a point source
    morph_ = frame.psf.get_model().mean(axis=0)
    origin = (
        center_index[0] - (morph_.shape[0] // 2),
        center_index[1] - (morph_.shape[1] // 2),
    )
    bbox_ = Box(morph_.shape, origin=origin)

    # adjust box size to conform with extended sources
    size = max(morph_.shape)
    boxsize = get_minimal_boxsize(size)
    morph = np.zeros((boxsize, boxsize))
    origin = (
        center_index[0] - (morph.shape[0] // 2),
        center_index[1] - (morph.shape[1] // 2),
    )
    bbox = Box(morph.shape, origin=origin)

    slices = overlapped_slices(bbox, bbox_)
    morph[slices[0]] = morph_[slices[1]]

    # apply max normalization
    morph /= morph.max()

    return morph, bbox


def init_extended_morphology(
    sky_coord,
    frame,
    detect,
    detect_std,
    thresh=1,
    symmetric=True,
    monotonic="flat",
    min_grad=0.1,
):
    """Initialize the source that is symmetric and monotonic
    See `ExtendedSource` for a description of the parameters
    """

    # position in frame coordinates
    center = frame.get_pixel(sky_coord)
    center_index = np.round(center).astype(np.int)

    # Copy detect if reused for other sources
    im = detect.copy()

    # Apply the necessary constraints
    if symmetric:
        im = operator.prox_uncentered_symmetry(
            im,
            0,
            center=center_index,
            algorithm="sdss",  # *1 is to artificially pass a variable that is not coadd
        )
    if monotonic:
        if monotonic is True:
            monotonic = "angle"
        # use finite thresh to remove flat bridges
        prox_monotonic = operator.prox_weighted_monotonic(
            im.shape,
            neighbor_weight=monotonic,
            center=center_index,
            min_gradient=min_grad,
        )
        im = prox_monotonic(im, 0).reshape(im.shape)

    # truncate morph at thresh * bg_rms
    threshold = detect_std * thresh
    morph, bbox = trim_morphology(center_index, im, bg_thresh=threshold)

    # normalize to unity at peak pixel for the imposed normalization
    if morph.sum() > 0:
        morph /= morph.max()
    else:
        morph = CenterOnConstraint(tiny=1)(morph, 0)
        msg = f"No flux in morphology model for source at {sky_coord}"
        logger.warning(msg)

    return morph, bbox


def init_multicomponent_morphology(
    sky_coord,
    frame,
    detect,
    detect_std,
    flux_percentiles,
    thresh=1,
    symmetric=True,
    monotonic="flat",
    min_grad=0,
):
    """Initialize multiple components
    See `MultiComponentSource` for a description of the parameters
    """

    # Initialize the whole component as an extended source
    morph, bbox = init_extended_morphology(
        sky_coord,
        frame,
        detect,
        detect_std,
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
            msg = f"Zero or negative morphology for component {k} at {sky_coord}"
            logger.warning(msg)
        morphs[k] /= morphs[k].max()

    # avoid using the same box for multiple components
    boxes = tuple(bbox.copy() for k in range(K))
    return morphs, boxes


def build_initialization_image(observations, spectra=None, prerender=False):
    """Build a spectrum-weighted image from all observations.

    Parameters
    ----------
    observations: list of `~scarlet.observation.Observation`
        Every Observation with a `image_prerender` attribute will contribute to the
        detection image, according to the noise level of that prerender image
    spectra: list of array
        for every observation: spectrum at the center of the source
        If not set, returns the detection image in all channels, instead of averaging.

    Returns
    -------
    image: array
        image created by weighting all of the channels by SED
    std: float
        the effective noise standard deviation of `image`
    """

    if not hasattr(observations, "__iter__"):
        observations = (observations,)
        if spectra is not None:
            spectra = (spectra,)

    # only use cache for the unweighted images
    # otherwise we have one per source!
    name = "build_initialization_image"
    key = tuple(observations), prerender
    if spectra is None:
        try:
            return Cache.check(name, key)
        except KeyError:
            pass

    model_frame = observations[0].model_frame
    detect = np.zeros(model_frame.shape, dtype=model_frame.dtype)
    var = np.zeros(model_frame.shape, dtype=model_frame.dtype)
    for i, obs in enumerate(observations):

        if prerender:
            # only use observations that have a prerender image
            if obs.prerender_images is None:
                continue
            images = obs.prerender_images
            bg_rms = obs.prerender_sigma
        else:
            images = obs.images
            bg_rms = np.mean(obs.noise_rms, axis=(1, 2))

        if spectra is None:
            spectrum = weights = 1
        else:
            spectrum = spectra[i][:, None, None]
            weights = spectrum / (bg_rms ** 2)[:, None, None]

        try:
            obs.map_channels(detect)[obs._slices_for_model] += (
                weights * images[obs._slices_for_images]
            )
            obs.map_channels(var)[obs._slices_for_model] += spectrum * weights
        # LowResObservation cannot be sliced into the model frame, continue without it
        except ValueError:
            continue

    if spectra is not None:
        detect = detect.sum(axis=0)
        var = var.sum(axis=0)

    # only save the unweighted one
    if spectra is None:
        Cache.set(name, key, (detect, np.sqrt(var)))

    return detect, np.sqrt(var)


def init_all_sources(
    frame,
    centers,
    observations,
    thresh=1,
    max_components=1,
    min_snr=30,
    edge_distance=None,
    shifting=False,
    fallback=True,
    prerender=False,
    silent=False,
    set_spectra=True,
):
    """Initialize all sources in a blend

    Any sources which cannot be initialized are returned as a `skipped`
    index, the index needed to reinsert them into a catalog to preserve
    their index in the output catalog.

    See `~init_sources` for a description of the arguments

    Parameters
    ----------
    centers : list of tuples
        `(y, x)` center location for each source in sky coordinates
    silent: bool
        If set to True, will prevent exceptions from being thrown abd register the
        source index in a list of skipped sources.
    set_spectra: bool
        If set to True, will solve for the best spectra of all sources given the
        observations. See `set_spectra_to_match` for details.

    Returns
    -------
    sources: list
        List of intialized sources, where each source derives from the
        `~scarlet.Component` class.
    skipped: list
        This list contains sources that failed to initialize with `silent` = True
    """
    if not hasattr(observations, "__iter__"):
        observations = (observations,)

    # Only deblend sources that can be initialized
    sources = []
    skipped = []
    for k, center in enumerate(centers):
        try:
            source = init_source(
                frame,
                center,
                observations,
                thresh=thresh,
                max_components=max_components,
                min_snr=min_snr,
                edge_distance=edge_distance,
                shifting=shifting,
                fallback=fallback,
                prerender=prerender,
            )
            sources.append(source)
        except Exception as e:
            msg = f"Failed to initialize source {k}"
            logger.warning(msg)
            if silent:
                skipped.append(k)
            else:
                raise e

    if set_spectra:
        set_spectra_to_match(sources, observations)

    return sources, skipped


def init_source(
    frame,
    center,
    observations,
    thresh=1,
    max_components=1,
    min_snr=30,
    edge_distance=None,
    shifting=False,
    fallback=True,
    prerender=False,
):
    """Initialize a Source

    The user can specify the number of desired components
    for the modeled source. If scarlet cannot initialize a
    model with the desired number of components it continues
    to attempt initialization of one fewer component until
    it finds a model that can be initialized.
    It is possible that scarlet will be unable to initialize a
    source with the desired number of components, for example
    a two component source might have degenerate components,
    a single component source might not have enough signal in
    the joint coadd (all bands combined together into
    single signal-to-noise weighted image for initialization)
    to initialize, and a true spurious detection will not have
    enough signal to initialize as a point source.
    If all of the models fail, including a `PointSource` model,
    then this source is skipped.

    Parameters
    ----------
    frame : `~scarlet.Frame`
        The model frame for the scene
    center : `tuple` of `float``
        `(y, x)` location for the center of the source.
    observations : instance or list of `~scarlet.Observation`
        The `Observation` that contains the images, weights, and PSF
        used to generate the model.
    thresh : `float`
        Fraction of the background to use as a threshold for
        each pixel in the initialization
    max_components : int
        The maximum number of components in a source.
        If `fallback` is `True` then when
        a source fails to initialize with `max_components` it
        will continue to subtract one from the number of components
        until it reaches zero (which fits a `CompactExtendedSource`).
        If a point source cannot be fit then the source is skipped.
    edge_distance : int
        The distance from the edge of the image to consider
        a source an edge source. For example if `edge_distance=3`
        then any source within 3 pixels of the edge will be
        considered to have edge flux.
        If `edge_distance` is `None` then the edge check is ignored.
    shifting : bool
        Whether or not to fit the position of a source.
        This is an expensive operation and is typically only used when
        a source is on the edge of the detector.
    fallback : bool
        Whether to reduce the number of components
        if the model cannot be initialized with `max_components`.
        This is unlikely to be used in production
        but can be useful for troubleshooting when an error can cause
        a particular source class to fail every time.
    """
    from .source import ExtendedSource

    if not hasattr(observations, "__iter__"):
        observations = (observations,)

    source_shifting = shifting
    while max_components >= 0:
        try:
            if max_components > 0:
                source = ExtendedSource(
                    frame,
                    center,
                    observations,
                    thresh=thresh,
                    shifting=source_shifting,
                    K=max_components,
                    prerender=prerender,
                )
            else:
                source = ExtendedSource(
                    frame, center, observations, shifting=source_shifting, compact=True
                )

            # test if parameters are fine, otherwise throw ArithmeticError
            source.check_parameters()

            # make sure to have enough SNR for every component
            if max_components > 1:
                components = source.children
            else:
                components = (source,)
            if max_components > 0 and any(
                [
                    measure.snr(component, observations, prerender=prerender) < min_snr
                    for component in components
                ]
            ):
                raise ArithmeticError("Insufficient SNR")

        except ArithmeticError as e:

            if fallback:
                msg = f"Could not initialize source at {center} with {max_components} components: {e}"
                logger.info(msg)
                max_components -= 1
                continue
            else:
                raise e

        # The LSST DM detection algorithm implemented in meas_algorithms
        # does not place sources within the edge mask
        # (roughly 5 pixels from the edge). This results in poor
        # deblending of the edge source, which for bright sources
        # may ruin an entire blend. So we reinitialize edge sources
        # to allow for shifting and return the result.
        if source_shifting is False and hasEdgeFlux(source, edge_distance):
            source_shifting = True
            continue

        return source


def hasEdgeFlux(source, edge_distance=1):
    """hasEdgeFlux

    Determine whether or not a source has flux within `edge_distance`
    of the edge.

    Parameters
    ----------
    source : `scarlet.Component`
        The source to check for edge flux
    edge_distance : int
        The distance from the edge of the image to consider
        a source an edge source. For example if `edge_distance=3`
        then any source within 3 pixels of the edge will be
        considered to have edge flux.
        If `edge_distance` is `None` then the edge check is ignored.

    Returns
    -------
    isEdge: `bool`
        Whether or not the source has flux on the edge.
    """
    if edge_distance is None:
        return False

    assert edge_distance > 0

    # Use the first band that has a non-zero SED
    flux = measure.flux(source)
    band = np.min(np.where(flux > 0)[0])
    model = source.get_model()[band]
    for edge in range(edge_distance):
        if (
            np.any(model[edge - 1] > 0)
            or np.any(model[-edge] > 0)
            or np.any(model[:, edge - 1] > 0)
            or np.any(model[:, -edge] > 0)
        ):
            return True
    return False


def set_spectra_to_match(sources, observations):

    from .component import FactorizedComponent, CombinedComponent

    if not hasattr(observations, "__iter__"):
        observations = (observations,)
    model_frame = observations[0].model_frame

    for obs in observations:

        # extract model for every component
        morphs = []
        parameters = []
        for src in sources:
            if isinstance(src, CombinedComponent):
                components = src.children
            else:
                components = (src,)
            for c in components:
                if isinstance(c, FactorizedComponent):
                    p = c.parameters[0]
                    if not p.fixed:
                        obs.map_channels(p)[:] = 1
                        parameters.append(p)
                        model_ = obs.render(c.get_model(frame=model_frame))
                        morphs.append(model_)

        morphs = np.array(morphs)
        K = len(morphs)

        images = obs.images
        weights = obs.weights
        C = obs.frame.C

        # independent channels, no mixing
        spectra = np.empty((K, C))
        for c in range(C):
            im = images[c].reshape(-1)
            w = weights[c].reshape(-1)
            m = morphs[:, c, :, :].reshape(K, -1)
            covar = np.linalg.inv((m * w[None, :]) @ m.T)
            spectra[:, c] = covar @ m @ (im * w)

        for p, spectrum in zip(parameters, spectra):
            obs.map_channels(p)[:] = spectrum

    # enforce constraints
    for p in parameters:
        p[:] = p.constraint(p, 0)

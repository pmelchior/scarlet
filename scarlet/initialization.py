import numpy as np
import logging

from .bbox import Box
from .cache import Cache
from .renderer import NullRenderer, ConvolutionRenderer


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
        pixel = obs.get_pixel(sky_coord)
        index = np.round(pixel).astype("int")
        spectrum = obs.data[:, index[0], index[1]].copy()

        if correct_psf and obs.psf is not None:
            # correct spectrum for PSF-induced change in peak pixel intensity
            psf_model = obs.psf.get_model()
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


def get_psf_spectrum(sky_coord, observations, compute_snr=False):
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
        Observation to extract the spectrum from.
    compute_snr: bool
        Whether the compute the SNR of a PSF at `sky_coord`

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
    if compute_snr:
        snr_num, snr_denom = [], []

    for i, obs in enumerate(observations):
        pixel = obs.get_pixel(sky_coord)
        index = np.round(pixel).astype("int")

        psf = obs.psf.get_model()
        bbox = obs.psf.bbox + (0, *index)
        img = bbox.extract_from(obs.data)
        noise_rms = obs.noise_rms
        # masked array doesn't survive extract_from
        noise = bbox.extract_from(noise_rms)
        noise_mask = bbox.extract_from(noise_rms.mask)

        spectra.append([])
        # apply mask: needs to be done separately in each channel
        for c in range(obs.C):
            # outside of observation or masked pixels have noise_mask = 0
            mask = ~(noise_mask[c])
            psf_ = psf[c, mask]
            img_ = img[c, mask]

            # amplitude of img when projected onto psf
            # i.e. factor to multiply psf with to get img (if img looked like psf)
            img_psf = img_ @ psf_
            spectrum = img_psf / (psf_ @ psf_)
            spectra[i].append(spectrum)

            if compute_snr:
                noise_ = noise[c, mask]
                snr_num.append(img_psf)
                snr_denom.append((psf_ * noise_ ** 2) @ psf_)

        spectra[i] = np.array(spectra[i])

        if np.any(spectra[i] <= 0):
            # If the flux in all channels is  <=0,
            # the new sed will be filled with NaN values,
            # which will cause the code to crash later
            msg = f"Zero or negative spectrum {spectra[i]} at {sky_coord}"
            if np.all(spectra[i] <= 0):
                logger.warning(msg)
            else:
                logger.info(msg)

    if single:
        spectra = spectra[0]

    if compute_snr:
        snr = np.sum(snr_num) / np.sqrt(np.sum(snr_denom))
        return spectra, snr
    return spectra


def get_minimal_boxsize(size, min_size=21, increment=10):
    boxsize = min_size
    while boxsize < size:
        boxsize += increment  # keep box sizes quite small
    return boxsize


def trim_morphology(center_index, morph, bg_thresh=0, boxsize=None):
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
    if boxsize is None:
        boxsize = get_minimal_boxsize(size)

    bottom = center_index[0] - boxsize // 2
    top = center_index[0] + boxsize // 2 + 1
    left = center_index[1] - boxsize // 2
    right = center_index[1] + boxsize // 2 + 1
    bbox = Box.from_bounds((bottom, top), (left, right))
    morph = bbox.extract_from(morph)
    return morph, bbox


def build_initialization_image(observations, spectra=None):
    """Build a spectrum-weighted image from all observations.

    Parameters
    ----------
    observations: list of `~scarlet.observation.Observation`
        Every observation with a suitable renderer will contribute to the initialization image, according to the noise level of its data
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

    model_frame = observations[0].model_frame

    # check if detection images are stored in obs[0]
    # stoing in an obs avoids using the cache (see issue 256)
    if not hasattr(observations[0], "_detect"):
        # if not, map every obs and variance onto the model frame
        detect, var = [], []
        for i, obs in enumerate(observations):

            # only works on unrotated simple frames
            if not isinstance(obs.renderer, (NullRenderer, ConvolutionRenderer)):
                continue

            detect_ = np.zeros(model_frame.shape, dtype=model_frame.dtype)
            var_ = np.zeros(model_frame.shape, dtype=model_frame.dtype)
            data_slice, model_slice = obs.renderer.slices
            obs.renderer.map_channels(detect_)[model_slice] += obs.data[data_slice]
            obs.renderer.map_channels(var_)[model_slice] += (
                obs.noise_rms[data_slice]
            ) ** 2
            detect.append(detect_)
            var.append(var_)
        detect = np.array(detect)  # L x C x Ny x Nx
        var = np.array(var)  # L x C x Ny x Nx
        observations[0]._detect = (detect, var)

    _detect, _var = observations[0]._detect

    # get multi-channel image for spectrum matching
    if spectra is None:
        nonzero = (_var > 0).sum(axis=0)
        detect = _detect.sum(axis=0) / nonzero
        var = _var.sum(axis=0) / nonzero
    else:
        # spectrum SNR weighted combination of all observations
        spectrum = []
        for i, obs in enumerate(observations):
            if not isinstance(obs.renderer, (NullRenderer, ConvolutionRenderer)):
                continue
            spectrum_ = np.zeros(model_frame.C)
            obs.renderer.map_channels(spectrum_)[:] = spectra[i]
            spectrum.append(spectrum_)
        spectrum = np.stack(spectrum, axis=0)[:, :, None, None]  # L x C x Ny x Nx
        weight = spectrum / _var
        detect = (weight * _detect).sum(axis=(0, 1))
        var = (spectrum * weight).sum(axis=(0, 1))

    return detect, np.sqrt(var)


def init_all_sources(
    frame,
    centers,
    observations,
    thresh=1,
    max_components=1,
    min_components=1,
    min_snr=50,
    shifting=False,
    resizing=True,
    boxsize=None,
    fallback=True,
    silent=False,
    set_spectra=True,
):
    """Initialize all sources in a blend

    Seeks to initialize sources at the sky positions of `centers` with multiple
    components of type `ExtendedSource`. If each component has sufficient SNR, the
    model will be kept, otherwise one component is removed and the source reinitialized.
    If a source cannot be initialized, its index is returned in `skipped`.

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
                min_components=min_components,
                min_snr=min_snr,
                shifting=shifting,
                resizing=resizing,
                boxsize=boxsize,
                fallback=fallback,
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
    min_components=1,
    min_snr=50,
    shifting=False,
    resizing=True,
    boxsize=None,
    fallback=True,
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
    If all of the models fail, including a `CompactExtendedSource` model,
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
    min_components : int
        The minimum number of components in a source.
        Only relevent for `fallback=True`.
    min_snr: float
        Mininmum SNR per component to accept the source.
    shifting : bool
        Whether or not to fit the position of a source.
        This is an expensive operation and is typically only used when
        a source is on the edge of the detector.
    resizing : bool
        Whether or not to change the size of the source box.
    boxsize: int or None
        Spatial size of the source box
    fallback : bool
        Whether to reduce the number of components
        if the model cannot be initialized with `max_components`.
        Fallback = False is unlikely to be used in production
        but can be useful for troubleshooting when an error can cause
        a particular source class to fail every time.
    prerender: bool
        Whether to initialize the source with pre-rendered observations.
        This is an experimental feature, which may be removed in the future.
    """
    from .source import ExtendedSource

    if not hasattr(observations, "__iter__"):
        observations = (observations,)

    if fallback:
        _, psf_snr = get_psf_spectrum(center, observations, compute_snr=True)
        max_components = np.min(
            [
                max_components,
                np.max([min_components, np.floor(psf_snr / min_snr).astype("int")]),
            ]
        )

    while max_components >= 0:
        try:
            if max_components > 0:
                source = ExtendedSource(
                    frame,
                    center,
                    observations,
                    thresh=thresh,
                    shifting=shifting,
                    resizing=resizing,
                    boxsize=boxsize,
                    K=max_components,
                )
            else:
                source = ExtendedSource(
                    frame,
                    center,
                    observations,
                    shifting=shifting,
                    resizing=resizing,
                    boxsize=boxsize,
                    compact=True,
                )

            # test if parameters are fine, otherwise throw ArithmeticError
            source.check_parameters()

        except ArithmeticError as e:
            if fallback:
                msg = f"Could not initialize source at {center} with {max_components} components: {e}"
                logger.info(msg)
                max_components -= 1
                continue
            else:
                raise e

        return source


def set_spectra_to_match(sources, observations):
    """Sets the spectra of any `FactorizedComponent` to match the Observations.

    Computes the best-fit amplitude of the rendered model of the sources in every
    channel of every observation as a linear inverse problem.

    Parameters
    ----------
    sources: list of sources
        Only `FactorizedComponent` or `CombinedComponent` will be considered
    observations: `scarlet.Observation` or list thereof
    """

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
                        obs.renderer.map_channels(p)[:] = 1
                        parameters.append(p)
                        model_ = obs.render(c.get_model(frame=model_frame))
                        morphs.append(model_)

        morphs = np.array(morphs)
        K = len(morphs)

        images = obs.data
        weights = obs.weights
        C = obs.C

        # independent channels, no mixing
        # solve the linear inverse problem of the amplitudes in every channel
        # given all the rendered morphologies
        # spectrum = (M^T Sigma^-1 M)^-1 M^T Sigma^-1 * im
        spectra = np.zeros((K, C))
        for c in range(C):
            im = images[c].reshape(-1)
            w = weights[c].reshape(-1)
            m = morphs[:, c, :, :].reshape(K, -1)
            mw = m * w[None, :]
            # check if all components have nonzero flux in c.
            # because of convolutions, flux can be outside of the box,
            # so we need to compare weighted flu with unweighted flux,
            # which is the same (up to a constant) for constant weights
            # so we check if *most* of the flux is from pixels with non-zero weight
            nonzero = np.sum(mw, axis=1) / np.sum(m, axis=1) / np.mean(w) > 0.1
            nonzero = np.flatnonzero(nonzero)
            if len(nonzero) == K:
                covar = np.linalg.inv(mw @ m.T)
                spectra[:, c] = covar @ m @ (im * w)
            else:
                covar = np.linalg.inv(mw[nonzero] @ m[nonzero].T)
                spectra[nonzero, c] = covar @ m[nonzero] @ (im * w)

        for p, spectrum in zip(parameters, spectra):
            obs.renderer.map_channels(p)[:] = spectrum

    # enforce constraints
    for p in parameters:
        if p.constraint is not None:
            p[:] = p.constraint(p, 0)

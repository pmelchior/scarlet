from functools import partial
import logging
import numpy as np

from ..operator import prox_monotonic_mask, prox_uncentered_symmetry, prox_weighted_monotonic
from ..detect import bounds_to_bbox, get_detect_wavelets
from ..bbox import Box, overlapped_slices
from ..initialization import trim_morphology
from ..parameter import relative_step
from .measure import calculate_snr
from .models import LiteSource, LiteFactorizedComponent, LiteComponent
from .parameters import AdaproxParameter, FistaParameter
from .utils import project_morph_to_center, insert_image


logger = logging.getLogger("scarlet.lite.initialization")


def get_min_psf(psfs, thresh=0.01):
    """Extract the significant portion of the PSF

    This function compares the PSF in each band and
    finds the minimum box needed to contain all pixels
    in the PSF model that differ by more than `thresh`
    in any two bands. The result is that all pixels
    outside of

    Parameters
    ----------
    psfs: `numpy.ndarray`
        The full 3D (bands, height, width) PSF model.
    thresh: `float`
        The minimal difference between two PSFs to be
        considered significant.
    make_circle: `bool`
        Whether or not to make the output PSFs
        circular. If `make_circle` is `False` then
        the square PSFs are returned.

    Returns
    -------
    psfs: `numpy.ndarray`
        The extracted PSFs.
    """
    # The radius of the PSF in the X and Y directions
    py = psfs.shape[1] // 2
    px = psfs.shape[2] // 2

    # Get the radial coordinates of each pixel
    X = np.arange(psfs.shape[-1])
    Y = np.arange(psfs.shape[-2])
    X, Y = np.meshgrid(X, Y)
    R = np.sqrt((X-px)**2 + (Y-py)**2)

    max_radius = 0
    for p1 in range(len(psfs)-1):
        for p2 in range(p1+1,len(psfs)):
            # Calculate the difference between the PSFs
            psf1 = psfs[p1]
            psf2 = psfs[p2]
            diff = (psf1 - psf2)/np.max([psf1, psf2])
            # keep all pixels greater than the threshold
            significant = np.abs(diff) > thresh
            # extract the radius for all of the significant pixels
            radius = int(np.max(R * significant))
            # Update the maximum radius (if necessary)
            if radius > max_radius:
                max_radius = radius
    # Create the slices to extract the PSF
    dy = py - max_radius
    dx = px - max_radius
    if dy > 0:
        sy = slice(dy, -dy)
    else:
        sy = slice(None)
    if dx > 0:
        sx = slice(dx, -dx)
    else:
        sx = slice(None)
    return psfs[:, sy, sx].copy()


def init_monotonic_morph(detect, center, full_box, grow=0, normalize=True,
        use_mask=True, thresh=0):
    """Initialize a morphology for a monotonic source

    Parameters
    ----------
    detect: `numpy.ndarray`
        The 2D detection image contained in `full_box`.
    center: `tuple` of `int`
        The center of the monotonic source.
    full_box: `scarlet.bbox.Box`
        The bounding box of `detect`.
    grow: `int`
        The number of pixels to grow the morphology in each direction.
        This can be useful if initializing a source with a kernel that
        is known to be narrower than the expected value of the source.
    normalize: `bool`
        Whether or not to normalize the morphology.
    tresh: `float`
        Background threshold value to use for truncating the morphology.

    Returns
    -------
    bbox: `scarlet.bbox.Box`
        The bounding box of the morphology.
    morph: `numpy.ndarray`
        The initialized morphology.
    """
    if use_mask:
        _, morph, bounds = prox_monotonic_mask(detect, 0, center, max_iter=0)
        bbox = bounds_to_bbox(bounds)
        if bbox.shape == (1, 1) and morph[bbox.slices][0,0] == 0:
            return bbox, None

        if grow is not None and grow > 0:
            bbox = bbox.grow(grow)
        morph, bbox = project_morph_to_center(morph, center, bbox, full_box)
    else:
        prox_monotonic = prox_weighted_monotonic(
            detect.shape,
            neighbor_weight="angle",
            center=center,
            min_gradient=0,
        )

        morph = prox_monotonic(detect, 0).reshape(detect.shape)

        # truncate morph at thresh * bg_rms
        morph, bbox = trim_morphology(center, morph, bg_thresh=thresh)
        if np.max(morph) == 0:
            return Box((0,0,0)), None

    if normalize:
        morph /= np.max(morph)
    return bbox, morph


def multifit_seds(observation, morphs, boxes):
    """Fit the seds f multiple components simultaneously

    Parameters
    ----------
    observation: `scarlet.let.LiteObservation`
        The class containing the observation data.
    morphs: `list` of `numpy.ndarray`
        The morphology of each component.
    boxes: `list` of `scarlet.bbox.Box`
        The bounding box for each morph.

    Returns
    -------
    seds: `list` of `numpy.ndarray`
        The SED for each component, in the same order as `morphs` and `boxes`.
    """
    if len(morphs) != len(boxes):
        msg = (f"morphs and boxes should have the same number of parameters, "
               f"got {len(morphs)} and {len(boxes)} respectively")
        raise ValueError(msg)
    bands = observation.images.shape[0]
    dtype = observation.images.dtype

    if len(morphs) != len(boxes):
        msg = f"morphs and boxes must be the same length, got {len(morphs)} and {len(boxes)}"
        raise Exception(msg)
    spec_box = observation.bbox[0]
    full_box = boxes[0]
    for box in boxes[1:]:
        full_box |= box
    full_box = spec_box @ full_box
    img = insert_image(full_box, observation.bbox, observation.images)

    morph_images = np.zeros((bands, len(morphs), img[0].size), dtype=dtype)
    for idx, (morph, bbox) in enumerate(zip(morphs, boxes)):
        _img = insert_image(full_box, spec_box @ bbox, morph[None, :, :])
        morph_images[:, idx] = observation.convolve(_img).reshape(bands,-1)

    seds = np.zeros((len(morphs), bands), dtype=dtype)

    for b in range(bands):
        A = np.vstack(morph_images[b]).T
        seds[:, b] = np.linalg.lstsq(A, img[b].flatten(), rcond=None)[0]
    seds[seds < 0] = 0
    return seds


def init_main_parameters(detect, center, observation,
        convolved=None, use_mask=False, thresh=0.5):
    """Initialize parameters using the same general algorithm as scarlet main

    This is currently up to date as of commit `6619736`, but might get out of
    date if the ain initialization changes, but even now there are slight
    differences that have very little effect on the overal initialization.

    Parameters
    ----------
    detect: `numpy.ndarray`
        The monochromatic detection image (usually a chi^2 coadd,
        possibly weighted by the SED of the source being detected).
    center: `tuple` of `int`
        The location of the center of the source to detect in the full image.
    observation: `scarlet.lite.LiteObservation`
        The observation that is being modeled.
    convolved: `numpy.ndarray`
        The convolved image in each band. Since the morphology of each source
        is close to the input images, this is a good approximation of the
        convolved morphologies and gives an SED within 1% without having
        to convolve the morphology of each source separately.
        If `convolved` is `None` then the result is accurate to
        machine precision.
    use_mask: `bool`
        Whether to use the monotonic mask constraint for initialization or
        the weighted monotonicity constraint.
    thresh: `float`
        The fraction of the `noise_rms` used to trim the morphology.
    """
    _detect = prox_uncentered_symmetry(detect.copy(), 0, center, "sdss")
    thresh = np.mean(observation.noise_rms) * thresh

    bbox, morph = init_monotonic_morph(
        _detect,
        center,
        observation.bbox[1:],
        grow=0,
        normalize=False,
        use_mask=use_mask,
        thresh=thresh,
    )

    if morph is None:
        return bbox, None, None

    sed_center = (slice(None), center[0], center[1])
    images = observation.images

    if convolved is None:
        # Convolve the morphology to get the exact SED to match the image,
        # accurate to machine precision
        _morph = insert_image(observation.bbox[1:], bbox, morph)
        convolved = observation.convolve(np.repeat(_morph[None, :, :], images.shape[0], axis=0), mode="real")
    sed = images[sed_center] / convolved[sed_center]
    sed[sed<0] = 0
    morph_max = np.max(morph)
    sed *= morph_max
    morph /= morph_max
    return bbox, morph, sed


def init_adaprox_component(center, bbox, sed, morph, observation, factor=10, bg_thresh=None,
                            max_prox_iter=1):
    """Initialize sed and morph as parameters optimized using adaprox

    Parameters
    ----------
    center: `tuple` of `int`
        The center of the component.
    sed: `numpy.ndarray`
        The SED of the component.
    morph: `numpy.ndarray`
        The morphology of the component.
    bbox: `scarlet.bbox.Box`
        The bounding box of the component.
    observation: `scarlet.lite.LiteObservation`
        The observation that is being modeled.
    factor: `float`
        The factor of the noise RMS to use as a threshold, where the minimum
        SED allowed is `obsevation.noise_rms/factor`.

    Returns
    -------
    component: `scarlet.lite.LiteFactorizedComponent`
        The component created using the input parameters.
    """
    sed = AdaproxParameter(
            sed,
            step=partial(relative_step, factor=1e-2, minimum=observation.noise_rms/factor),
            max_prox_iter=max_prox_iter,
        )
    morph = AdaproxParameter(morph, step=1e-2, max_prox_iter=max_prox_iter)
    component = LiteFactorizedComponent(
        sed, morph, center, bbox, observation.bbox, observation.noise_rms, bg_thresh=bg_thresh,
    )
    return component


def init_fista_component(center, bbox, sed, morph, observation, bg_thresh=None):
    """Initialize sed and morph as parameters optimized using FISTA PGM

    Parameters
    ----------
    center: `tuple` of `int`
        The center of the component.
    sed: `numpy.ndarray`
        The SED of the component.
    morph: `numpy.ndarray`
        The morphology of the component.
    bbox: `scarlet.bbox.Box`
        The bounding box of the component.
    observation: `scarlet.lite.LiteObservation`
        The observation that is being modeled.
    bg_thresh: `float`
        The factor of the noise RMS to use for clipping sources

    Returns
    -------
    component: `scarlet.lite.LiteFactorizedComponent`
        The component created using the input parameters.
    """
    slices = overlapped_slices(bbox, observation.bbox)
    _weights = observation.weights[slices[1]]
    step = 2*np.mean(_weights[_weights > 0])
    _sed = FistaParameter(sed, step=1/step)
    _morph = FistaParameter(morph, step=1/step)
    component = LiteFactorizedComponent(
        _sed, _morph, center, bbox, observation.bbox, observation.noise_rms, bg_thresh=bg_thresh,
    )
    return component


def init_all_sources_main(observation, centers, detect=None,
        min_snr=50, use_mask=False, percentile=25, thresh=0.5):
    """Initialize all of the sources in a blend into factrized components

    This function uses a set of algorithms to give similar results to the
    algorithms in scarlet main to give nearly identical resulting sed and
    morphology arrays without creating all of the intermediate scarlet
    objects.

    Parameters
    ----------
    observation: `scarlet.lite.LiteObservation`
        The observation containing the blend
    detect: `numpy.ndarray`
        The array that contains a 2D image used for detection.
    centers: `list` of `tuple`
        The coordinates of all the peak locations to use for initializing sources.
    min_snr: `float`
        The minimum SNR required per component.
        So a 2-component source requires at least `2*min_snr` while sources
        with SNR < `min_snr` will be initialized with the PSF.
    use_mask: `bool`
        Whether to use the monotonic mask or weighted monotonicity for
        initialization.
    percentile: `float`
        The percentage of the overall central flux to attribute to the disk.
    thresh: `float`
        The threshold used to trim the morphology

    Returns
    -------
    sources: `list` of `scarlet.lite.LiteSource`
        The list of sources in the blend.
        This includes null sources that have no components.
    """
    if detect is None:
        detect = np.sum(observation.images/(observation.noise_rms**2)[:, None, None], axis=0)
    convolved = observation.convolve(np.repeat(detect[None, :, :], observation.shape[0], axis=0), mode="real")
    model_psf = observation.model_psf[0]
    convolved_psf = observation.convolve(
        np.repeat(observation.model_psf,
        observation.images.shape[0], axis=0),
        mode="real"
    )
    py = model_psf.shape[0]//2
    px = model_psf.shape[1]//2
    psf_sed = convolved_psf[:, py, px]

    sources = []
    for center in centers:
        snr = np.floor(calculate_snr(observation.images, observation.variance, observation.psfs, center))
        component_snr = snr / min_snr

        bbox, morph, sed = bbox, morph, sed = init_main_parameters(
            detect, center, observation, convolved, use_mask, thresh)

        if morph is None:
            sed_center = (slice(None), center[0], center[1])
            sed = observation.images[sed_center] / psf_sed
            sed[sed<0] = 0
            morph = model_psf.copy()
            morph = morph/np.max(morph)
            bbox = Box(model_psf.shape, origin=(center[0]-py, center[1]-px))

        elif component_snr >= 2:
            bulge_morph = morph.copy()
            disk_morph = morph
            flux_thresh = percentile / 100
            mask = disk_morph > flux_thresh
            disk_morph[mask] = flux_thresh
            bulge_morph -= flux_thresh
            bulge_morph[bulge_morph < 0] = 0

            if bulge_morph is None or disk_morph is None:
                if bulge_morph is None:
                    if disk_morph is None:
                        return None
                    morph = disk_morph
                else:
                    morph = bulge_morph
                # One of the components was null,
                # so initialize as a single component
                components = [LiteComponent(center, observation.bbox @ bbox, sed, morph)]
            else:
                bulge_morph /= np.max(bulge_morph)
                disk_morph /= np.max(disk_morph)

                bulge_sed, disk_sed = multifit_seds(observation, [bulge_morph, disk_morph], [bbox, bbox])

                components = [
                    LiteComponent(center, observation.bbox[0] @ bbox, bulge_sed, bulge_morph),
                    LiteComponent(center, observation.bbox[0] @ bbox, disk_sed, disk_morph),
                ]
        else:
            components = [LiteComponent(center, observation.bbox[0] @ bbox, sed, morph)]

        source = LiteSource(components, observation.dtype)
        sources.append(source)
    return sources


class WaveletInitParameters:
    """Parameters used to initialize all sources with wavelet detections

    There are a large number of parameters that are universal for all of the
    sources being initialized from the same set of wavelet coefficients.
    To simplify the API those parameters are all initialized by this class
    and passed to `init_wavelet_source` for each source.
    """
    def __init__(self, observation,
            bulge_slice=slice(None,2), disk_slice=slice(2, -1),
            bulge_grow=5, disk_grow=5, use_psf=True, scales=5, wavelets=None):
        """Initialize the parameters.

        See `init_all_sources_wavelets` for a description of the parameters.
        """
        if wavelets is None:
            wavelets = get_detect_wavelets(observation.images, observation.variance, scales=scales)
        wavelets[wavelets<0] = 0
        # The detection coadd for single component sources
        detectlets = np.sum(wavelets[:-1], axis=0)
        # The detection coadd for the bulge
        bulgelets = np.sum(wavelets[bulge_slice], axis=0)
        # The detection coadd for the disk
        disklets = np.sum(wavelets[disk_slice], axis=0)

        # useful extracted parameters
        images = observation.images
        model_psf = observation.model_psf[0]

        # The convolve image, used to initialize the SED
        convolved = observation.convolve(
            np.repeat(detectlets[None, :, :],
            observation.shape[0], axis=0),
            mode="real"
        )
        convolved_psf = observation.convolve(
            np.repeat(model_psf[None, :, :],
            observation.images.shape[0], axis=0),
            mode="real"
        )
        py = observation.model_psf.shape[1]//2
        px = observation.model_psf.shape[2]//2
        psf_sed = convolved_psf[:, py, px]

        self.observation = observation
        self.images = images
        self.convolved = convolved
        self.detectlets = detectlets
        self.bulgelets = bulgelets
        self.disklets = disklets
        self.bulge_grow = bulge_grow
        self.disk_grow = disk_grow
        self.psf_sed = psf_sed
        self.py = py
        self.px = px
        self.use_psf = use_psf


def init_wavelet_source(center, nbr_components, init):
    """Initialize a single source with wavelet coefficients

    Parameters
    ----------
    center: `tupel` of `int`
        The location of the source in the full image.
    nbr_components: `int`
        The number of components of the source.
        If `nbr_components >= 2` then initialization with 2 components
        is attempted. If this fails, or if ` 2 > nbr_components >= 1`
        then initialization with 1 component is attempted.
        Otherwise the source is initialized with the PSF.
    init: `WaveletInitParameters`
        Parameters used to initialize all sources.

    Returns
    -------
    source: `scarlet.lite.LiteSource`
    """
    observation = init.observation
    model_psf = observation.model_psf[0]
    sed_center = (slice(None), center[0], center[1])

    if nbr_components < 1 and init.use_psf or init.detectlets[center[0], center[1]] <= 0:
        sed = init.images[sed_center] / init.psf_sed
        sed[sed<0] = 0
        morph = model_psf.copy()
        morph = morph/np.max(morph)
        bbox = Box(model_psf.shape, origin=(center[0]-init.py, center[1]-init.px))

        component = LiteComponent(center, observation.bbox[0] @ bbox, sed, morph)
        source = LiteSource([component], observation.dtype)
    elif nbr_components < 2:
        bbox, morph = init_monotonic_morph(init.detectlets, center, observation.bbox[1:], init.disk_grow)
        if morph is None or np.max(morph) <= 0:
            return LiteSource([], observation.dtype)

        sed = init.images[sed_center] / init.convolved[sed_center]
        sed[sed<0] = 0
        morph = morph/np.max(morph)

        component = LiteComponent(center, observation.bbox[0] @ bbox, sed, morph)
        source = LiteSource([component], observation.dtype)
    else:
        bulge_box, bulge_morph = init_monotonic_morph(
            init.bulgelets, center, observation.bbox[1:], init.bulge_grow)
        disk_box, disk_morph = init_monotonic_morph(
            init.disklets, center, observation.bbox[1:], init.disk_grow)

        if bulge_morph is None or disk_morph is None:
            if bulge_morph is None:
                if disk_morph is None:
                    return None
                morph = disk_morph
                bbox = disk_box
            else:
                morph = bulge_morph
                bbox = bulge_box
            # One of the components was null,
            # so initialize as a single component
            return init_wavelet_source(center, 1, init)
        else:
            bulge_sed, disk_sed = multifit_seds(
                observation, [bulge_morph, disk_morph], [bulge_box, disk_box])

            components = []
            if np.sum(bulge_sed != 0):
                components.append(
                    LiteComponent(center, observation.bbox[0] @ bulge_box, bulge_sed, bulge_morph))
            else:
                logger.debug("cut bulge")
            if np.sum(disk_sed) != 0:
                components.append(
                    LiteComponent(center, observation.bbox[0] @ disk_box, disk_sed, disk_morph))
            else:
                logger.debug("cut disk")

            source = LiteSource(components, observation.dtype)
    return source


def init_all_sources_wavelets(observation, centers, min_snr=50, bulge_grow=5, disk_grow=5,
        use_psf=True, bulge_slice=slice(None,2), disk_slice=slice(2, -1), scales=5, wavelets=None):
    """Initialize all sources using wavelet detection images.

    This does not initialize the SED and morpholgy parameters, so
    `parameterize_source` must still be run to select a parameterization
    (optimizer) that `LiteBlend` requires for fitting.

    Parameters
    ----------
    observation: `scarlet.lite.LiteObservation`
        The multiband observation of the blend.
    centers: `list` of `tuple`
        Peak locations for all of the sources to attempt to initialize.
    wavelets: `numpy.ndarray`
        The array of wavelet coefficients `(scale, y, x)` used for detection.
    bulge_slice, disk_slice: `slice`
        The slice used to select the wavelet scales used for the bulge/disk.
    bulge_grow, disk_grow: `int`
        The number of pixels to grow the bounding box of the bulge/disk
        to leave extra room for growth in the first few iterations.
    use_psf: `bool`
        Whether or not to use the PSF for single component sources.
        If `use_psf` is `False` then only sources with low signal at all scales
        are initialized with the PSF morphology.
    min_snr: `float`
        Minimum signal to noise for each component. So if `min_snr=50`,
        a source must have SNR > 50 to be initialized with one component
        and SNR > 100 for 2 components.

    Returns
    -------
    sources: `list` of `scarlet.lite.LiteSource`
        The sources that have been initialized.
    """
    init = WaveletInitParameters(
        observation, bulge_slice, disk_slice, bulge_grow, disk_grow, use_psf, scales, wavelets)
    sources = []
    for center in centers:
        snr = np.floor(calculate_snr(observation.images, observation.variance, observation.psfs, center))
        component_snr = snr / min_snr
        source = init_wavelet_source(center, component_snr, init)
        sources.append(source)
    return sources


def parameterize_sources(sources, observation, parameterization):
    """Convert the parameters in a list of sources in scarlet lite parameters

    Parameters
    ----------
    sources: `list` of `scarlet.lite.LiteSource`
        The sources to parameterize.
    observation: `scarlet.lite.LiteObservation`
        The observation that is being fit.
    parameterization: `Callable`
        The function that is used to convert the SED and morphology into
        parameters that update by a given optimization algorithm.
        The function must take `center`, `sed`, `morph`, `bbox`, and
        `observation` as parameters.
        For an example of a valid `parameterization` function see
        `init_adaprox_component`.

    Returns
    -------
    sources: `list` of `scarlet.lite.LiteSource`
        The input list of sources with their components updated.
    """
    new_sources = []
    for src in sources:
        components = []
        for c in src.components:
            # Copy all of the parameters in case the same initialization
            # variables are wrapped by different parameterizations
            component = parameterization(
                center=tuple([coord for coord in c.center]),
                sed=c.sed.copy(),
                morph=c.morph.copy(),
                bbox=c.bbox.copy(),
                observation=observation
            )
            components.append(component)
        new_sources.append(LiteSource(components, src.dtype))
    return new_sources

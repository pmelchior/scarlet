import autograd.numpy as np
from functools import partial

from .bbox import Box
from .component import Component, CombinedComponent, FactorizedComponent
from .constraint import PositivityConstraint
from .initialization import (
    get_best_fit_spectrum,
    get_pixel_spectrum,
    init_compact_source,
    init_extended_morphology,
    init_multicomponent_morphology,
    build_initialization_image,
)
from .morphology import (
    ImageMorphology,
    PointSourceMorphology,
    StarletMorphology,
    ExtendedSourceMorphology,
)
from .parameter import Parameter, relative_step
from .spectrum import TabulatedSpectrum


class NullSource(Component):
    """Source that does nothing"""

    def __init__(self, model_frame):
        """
        Initialize a NullSource

        Parameters
        ----------
        model_frame: `~scarlet.Frame`
            The frame of the model
        """
        super().__init__(model_frame)

    def get_model(self, *parameters, frame=None):
        """Get the model for this component.

        Parameters
        ----------
        parameters: tuple of optimimzation parameters

        frame: `~scarlet.frame.Frame`
            Frame to project the model into. If `frame` is `None`
            then the model contained in `bbox` is returned.

        Returns
        -------
        model: array
            (Channels, Height, Width) image of the model
        """
        model = np.zeros(self.frame.shape)
        # project the model into frame (if necessary)
        if frame is not None:
            model = self.model_to_frame(frame, model)
        return model


class RandomSource(FactorizedComponent):
    """Sources with uniform random morphology and sed.

    For cases with no well-defined spatial shape, this source initializes
    a uniform random field and (optionally) matches the SED to match a given
    observation.
    """

    def __init__(self, model_frame, observations=None):
        """Source intialized as random field.

        Parameters
        ----------
        model_frame: `~scarlet.Frame`
            The frame of the model
        observations: instance or list of `~scarlet.Observation`
            Observation to initialize the SED of the source
        """
        C, Ny, Nx = model_frame.bbox.shape
        image = np.random.rand(Ny, Nx)
        morphology = ImageMorphology(image)

        if observations is None:
            spectrum = np.random.rand(C)
        else:
            spectrum = get_best_fit_spectrum(image[None], observations)[0]

        # default is step=1e-2, using larger steps here becaus SED is probably uncertain
        spectrum = Parameter(
            spectrum,
            name="spectrum",
            step=partial(relative_step, factor=1e-1),
            constraint=PositivityConstraint(),
        )
        spectrum = TabulatedSpectrum(spectrum)

        super().__init__(model_frame, spectrum, morphology)


class PointSource(FactorizedComponent):
    """Point-Source model

    Point sources modeled as `model_frame.psfs`, centered at `sky_coord`.
    Their SEDs are taken from `observations` at the center pixel.
    """

    def __init__(self, model_frame, sky_coord, observations):
        """Source intialized with a single pixel

        Parameters
        ----------
        model_frame: `~scarlet.Frame`
            The frame of the full model
        sky_coord: tuple
            Center of the source
        observations: instance or list of `~scarlet.Observation`
            Observation(s) to initialize this source
        """
        if not hasattr(observations, "__iter__"):
            observations = (observations,)

        center = model_frame.get_pixel(sky_coord)
        morphology = PointSourceMorphology(model_frame, center)

        # get spectrum from peak pixel, correct for PSF
        spectra = get_pixel_spectrum(sky_coord, observations, correct_psf=True)
        spectrum = np.concatenate(spectra, axis=0)
        spectrum = TabulatedSpectrum(model_frame, spectrum)

        super().__init__(model_frame, spectrum, morphology)

        # retain center as attribute
        self.center = morphology.center


class CompactExtendedSource(FactorizedComponent):
    def __init__(
        self, model_frame, sky_coord, observations, shifting=False,
    ):
        """Compact extended source model

        The model is initialized from `observations` with a point-source morphology
        and a spectrum from its peak pixel.

        During optimization it enforces positivitiy for spectrum and morphology,
        as well as monotonicity of the morphology.

        Parameters
        ----------
        model_frame: `~scarlet.Frame`
            The frame of the full model
        sky_coord: tuple
            Center of the source
        observations: instance or list of `~scarlet.observation.Observation`
            Observation(s) to initialize this source.
        shifting: `bool`
            Whether or not a subpixel shift is added as optimization parameter
        """
        if not hasattr(observations, "__iter__"):
            observations = (observations,)

        # initialize morphology from model_frame psf
        morph, bbox = init_compact_source(sky_coord, model_frame)
        center = model_frame.get_pixel(sky_coord)
        morphology = ExtendedSourceMorphology(
            model_frame,
            center,
            morph,
            bbox=bbox,
            monotonic="angle",
            symmetric=False,
            min_grad=0,
            shifting=shifting,
        )

        # get spectrum from peak pixel, correct for PSF
        spectra = get_pixel_spectrum(sky_coord, observations, correct_psf=True)
        spectrum = np.concatenate(spectra, axis=0)
        spectrum /= morph.sum()
        noise_rms = np.array(
            [np.mean(obs.noise_rms, axis=(1, 2)) for obs in observations]
        )
        spectrum = TabulatedSpectrum(model_frame, spectrum, min_step=noise_rms)

        # set up model with its parameters
        super().__init__(model_frame, spectrum, morphology)

        # retain center as attribute
        self.center = morphology.center


class SingleExtendedSource(FactorizedComponent):
    def __init__(
        self,
        model_frame,
        sky_coord,
        observations,
        thresh=1.0,
        shifting=False,
        prerender=False,
    ):
        """Extended source model

        The model is initialized from `observations` with a symmetric and
        monotonic profile and a spectrum from its peak pixel.

        During optimization it enforces positivitiy for spectrum and morphology,
        as well as monotonicity of the morphology.

        Parameters
        ----------
        model_frame: `~scarlet.Frame`
            The frame of the full model
        sky_coord: tuple
            Center of the source
        observations: instance or list of `~scarlet.observation.Observation`
            Observation(s) to initialize this source.
        thresh: `float`
            Multiple of the backround RMS used as a
            flux cutoff for morphology initialization.
        compact: `bool`
            Initialize with the shape of a point source
        shifting: `bool`
            Whether or not a subpixel shift is added as optimization parameter
        """
        if not hasattr(observations, "__iter__"):
            observations = (observations,)

        # get center pixel spectrum
        # this is from convolved image: weighs higher emission *and* narrow PSF
        spectra = get_pixel_spectrum(sky_coord, observations)

        # initialize morphology
        # compute optimal SNR coadd for detection
        image, std = build_initialization_image(
            observations, spectra=spectra, prerender=prerender
        )
        # make monotonic morphology, trimmed to box with pixels above std
        morph, bbox = init_extended_morphology(
            sky_coord,
            model_frame,
            image,
            std,
            thresh=thresh,
            symmetric=True,
            monotonic="flat",
            min_grad=0,
        )

        center = model_frame.get_pixel(sky_coord)
        morphology = ExtendedSourceMorphology(
            model_frame,
            center,
            morph,
            bbox=bbox,
            monotonic="angle",
            symmetric=False,
            min_grad=0,
            shifting=shifting,
        )

        # find best-fit spectra for morph from init coadd
        # assumes img only has that source in region of the box
        detect_all, std_all = build_initialization_image(
            observations, prerender=prerender
        )
        box_3D = Box((model_frame.C,)) @ bbox
        boxed_detect = box_3D.extract_from(detect_all)
        spectrum = get_best_fit_spectrum((morph,), boxed_detect)
        noise_rms = np.concatenate(
            [np.array(np.mean(obs.noise_rms, axis=(1, 2))) for obs in observations]
        )
        spectrum = TabulatedSpectrum(model_frame, spectrum, min_step=noise_rms)

        # set up model with its parameters
        super().__init__(model_frame, spectrum, morphology)

        # retain center as attribute
        self.center = morphology.center


class StarletSource(FactorizedComponent):
    """Source intialized with starlet coefficients

    Sources are initialized with the SED of the center pixel,
    and the morphologies are initialised as `~scarlet.ExtendedSource`
    and transformed into starlet coefficients.
    """

    def __init__(
        self,
        model_frame,
        sky_coord,
        observations,
        thresh=1.0,
        min_grad=0.1,
        starlet_thresh=5e-3,
        prerender=False,
    ):
        """Extended source intialized to match a set of observations

        Parameters
        ----------
        model_frame: `~scarlet.Frame`
            The frame of the model
        sky_coord: tuple
            Center of the source
        observations: instance or list of `~scarlet.observation.Observation`
            Observation(s) to initialize this source.
        thresh: `float`
            Multiple of the backround RMS used as a
            flux cutoff for morphology initialization.
        """

        if not hasattr(observations, "__iter__"):
            observations = (observations,)

        # get center pixel spectrum
        # this is from convolved image: weighs higher emission *and* narrow PSF
        spectra = get_pixel_spectrum(sky_coord, observations)

        # initialize morphology
        image, std = build_initialization_image(
            observations, spectra=spectra, prerender=prerender
        )
        # make monotonic morphology, trimmed to box with pixels above std
        morph, bbox = init_extended_morphology(
            sky_coord,
            model_frame,
            image,
            std,
            thresh=thresh,
            symmetric=True,
            monotonic="flat",
            min_grad=0,
        )

        # transform to starlets
        morphology = StarletMorphology(
            model_frame, morph, bbox=bbox, threshold=starlet_thresh
        )

        # find best-fit spectra for morph from unweighted deconvolution coadd
        # assumes img only has that source in region of the box
        morph = morphology.get_model()
        detect_all, std_all = build_initialization_image(
            observations, prerender=prerender
        )
        box_3D = Box((model_frame.C,)) @ bbox
        boxed_detect = box_3D.extract_from(detect_all)
        spectrum = get_best_fit_spectrum((morph,), boxed_detect)
        noise_rms = np.concatenate(
            [np.array(np.mean(obs.noise_rms, axis=(1, 2))) for obs in observations]
        )
        spectrum = TabulatedSpectrum(model_frame, spectrum, min_step=noise_rms)

        super().__init__(model_frame, spectrum, morphology)


class MultiExtendedSource(CombinedComponent):
    """Extended source with multiple components layered vertically

    Uses `~scarlet.ExtendedSource` to define the overall morphology,
    then erodes the outer footprint until it reaches the specified size percentile.
    For the narrower footprint, it evaluates the mean value at the perimeter and
    sets the inside to the perimeter value, creating a flat distribution inside.
    The subsequent component(s) is/are set to the difference between the flattened
    and the overall morphology.
    The SED for all components is calculated as the best fit of the multi-component
    morphology to the multi-channel image in the region of the source.
    """

    def __init__(
        self,
        model_frame,
        sky_coord,
        observations,
        K=2,
        flux_percentiles=None,
        thresh=1.0,
        shifting=False,
        prerender=False,
    ):
        """Create multi-component extended source.

        Parameters
        ----------
        model_frame: `~scarlet.Frame`
            The frame of the full model
        sky_coord: tuple
            Center of the source
        observations: instance or list of `~scarlet.observation.Observation`
            Observation(s) to initialize this source.
        K: int
            Number of stacked components
        flux_percentiles: list
            Flux percentile of each component as the transition point between components.
            If pixel value is below the first precentile, it becomes part of the
            outermost component. If it is above, the percentile value will be subtracted
            and the remainder attributed to the next component.
            If `flux_percentiles` is `None` then `flux_percentiles=[25,]`.
        thresh: `float`
            Multiple of the backround RMS used as a
            flux cutoff for morphology initialization.
        shifting: `bool`
            Whether or not a subpixel shift is added as optimization parameter
        """

        if flux_percentiles is None:
            flux_percentiles = (25,)
        assert K == len(flux_percentiles) + 1

        # initialize from observation
        if not hasattr(observations, "__iter__"):
            observations = (observations,)

        # get center pixel spectrum
        # this is from convolved image: weighs higher emission *and* narrow PSF
        spectra = get_pixel_spectrum(sky_coord, observations)
        image, std = build_initialization_image(
            observations, spectra=spectra, prerender=prerender
        )
        morphs, boxes = init_multicomponent_morphology(
            sky_coord,
            model_frame,
            image,
            std,
            flux_percentiles,
            thresh=thresh,
            symmetric=True,
            monotonic="flat",
            min_grad=0,
        )

        # find best-fit spectra for each of morph from unweighted deconvolution coadd
        # assumes img only has that source in region of the box
        detect_all, std_all = build_initialization_image(
            observations, prerender=prerender
        )
        box_3D = Box((model_frame.C,)) @ boxes[0]
        boxed_detect = box_3D.extract_from(detect_all)
        spectra = get_best_fit_spectrum(morphs, boxed_detect)
        noise_rms = np.concatenate(
            [np.array(np.mean(obs.noise_rms, axis=(1, 2))) for obs in observations]
        )

        # create one component for each spectrum and morphology
        components = []
        center = model_frame.get_pixel(sky_coord)
        for k in range(K):

            spectrum = TabulatedSpectrum(model_frame, spectra[k], noise_rms=noise_rms)

            morphology = ExtendedSourceMorphology(
                model_frame,
                center,
                morphs[k],
                bbox=boxes[k],
                monotonic="angle",
                symmetric=False,
                min_grad=0,
                shifting=shifting,
            )
            self.center = morphology.center
            component = FactorizedComponent(model_frame, spectrum, morphology)
            components.append(component)

        super().__init__(components)


def append_docs_from(other_func):
    def doc(func):
        func.__doc__ = func.__doc__ + "\n\n" + other_func.__doc__
        return func

    return doc


# factory two switch between single and multi-ExtendedSource
@append_docs_from(MultiExtendedSource.__init__)
def ExtendedSource(
    model_frame,
    sky_coord,
    observations,
    K=1,
    flux_percentiles=None,
    thresh=1.0,
    compact=False,
    shifting=False,
    prerender=False,
):
    """Create extended sources with either a single component or multiple components.

    If `K== 1`, a single instance of `SingleExtendedSource` is returned, otherwise
    and instance of `MultiExtendedSource` is returned.
    """

    if compact:
        return CompactExtendedSource(
            model_frame, sky_coord, observations, shifting=shifting,
        )
    if K == 1:
        return SingleExtendedSource(
            model_frame,
            sky_coord,
            observations,
            thresh=thresh,
            shifting=shifting,
            prerender=prerender,
        )
    else:
        return MultiExtendedSource(
            model_frame,
            sky_coord,
            observations,
            K=K,
            flux_percentiles=flux_percentiles,
            thresh=thresh,
            shifting=shifting,
            prerender=prerender,
        )

from functools import partial

from .initialization import *
from .spectrum import *
from .morphology import *
from .constraint import *
from .parameter import Parameter, relative_step
from .component import Component, FactorizedComponent, CombinedComponent
from .bbox import Box

# make sure that import * above doesn't import its own stock numpy
import autograd.numpy as np


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
            spectrum = get_best_fit_seds(image[None], observations)[0]

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

    Their SEDs are either taken from `observations` at the center pixel or specified
    by `sed` and/or `sed_func`. In the former case, `sed` is a parameter array with
    the same size of `model_frame.C`, in the latter case it is the parameters of the
    SED model function.
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
        spectrum = get_psf_sed(sky_coord, observations, model_frame)
        spectrum = TabulatedSpectrum(model_frame, spectrum)

        center = np.array(model_frame.get_pixel(sky_coord), dtype="float")
        morphology = PointSourceMorphology(model_frame, center)
        self.center = morphology.center
        super().__init__(model_frame, spectrum, morphology)


class StarletSource(FactorizedComponent):
    """Source intialized with starlet coefficients.

    Sources are initialized with the SED of the center pixel,
    and the morphologies are initialised as ExtendedSources
    and transformed into starlet coefficients.
    """

    def __init__(
        self,
        model_frame,
        sky_coord,
        observations,
        coadd=None,
        bg_cutoff=None,
        thresh=1.0,
        min_grad=0.1,
        starlet_thresh=5,
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

        # initialize from observation
        sed, morph, bbox, thresh = init_starlet_source(
            sky_coord,
            model_frame,
            observations,
            coadd,
            bg_cutoff,
            thresh=thresh,
            symmetric=True,
            monotonic="angle",
            min_grad=min_grad,
            starlet_thresh=starlet_thresh,
        )
        spectrum = TabulatedSpectrum(model_frame, sed, bbox=bbox[0])
        morphology = StarletMorphology(
            model_frame, morph, bbox=bbox[1:], threshold=thresh
        )
        super().__init__(model_frame, spectrum, morphology)


class ExtendedSource(FactorizedComponent):
    def __init__(
        self,
        model_frame,
        sky_coord,
        observations,
        coadd=None,
        bg_cutoff=None,
        thresh=1.0,
        monotonic="flat",
        symmetric=False,
        shifting=False,
        min_grad=0.1,
    ):
        """Extended source intialized to match a set of observations

        Parameters
        ----------
        model_frame: `~scarlet.Frame`
            The frame of the full model
        sky_coord: tuple
            Center of the source
        observations: instance or list of `~scarlet.observation.Observation`
            Observation(s) to initialize this source.
        coadd: `numpy.ndarray`
            The coaddition of all images across observations.
        bg_cutoff: float
            flux cutoff for morphology initialization.
        thresh: `float`
            Multiple of the backround RMS used as a
            flux cutoff for morphology initialization.
        monotonic: ['flat', 'angle', 'nearest'] or None
            Which version of monotonic decrease in flux from the center to enforce
        symmetric: `bool`
            Whether or not to enforce symmetry.
        shifting: `bool`
            Whether or not a subpixel shift is added as optimization parameter
        """
        # initialize from observation
        sed, morph, bbox = init_extended_source(
            sky_coord,
            model_frame,
            observations,
            coadd,
            bg_cutoff,
            thresh=thresh,
            symmetric=True,
            monotonic="flat",
            min_grad=min_grad,
        )
        spectrum = TabulatedSpectrum(model_frame, sed, bbox=bbox[0])

        center = np.array(model_frame.get_pixel(sky_coord), dtype="float")
        morphology = ExtendedSourceMorphology(
            model_frame,
            center,
            morph,
            bbox=bbox[1:],
            monotonic=monotonic,
            symmetric=symmetric,
            min_grad=min_grad,
            shifting=shifting,
        )
        self.center = morphology.center
        super().__init__(model_frame, spectrum, morphology)


class MultiComponentSource(CombinedComponent):
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

    def __init__(
        self,
        model_frame,
        sky_coord,
        observations,
        coadd=None,
        bg_cutoff=None,
        thresh=1.0,
        flux_percentiles=None,
        symmetric=False,
        monotonic="flat",
        shifting=False,
        min_grad=0.1,
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
        obs_idx: int
            Index of the observation in `observations` to
            initialize the morphology.
        thresh: `float`
            Multiple of the backround RMS used as a
            flux cutoff for morphology initialization.
        flux_percentiles: list
            The flux percentile of each component. If `flux_percentiles` is `None`
            then `flux_percentiles=[25,]`, a single component with 25% of the flux
            as the primary source.
        symmetric: `bool`
            Whether or not to enforce symmetry.
        monotonic: ['flat', 'angle', 'nearest'] or None
            Which version of monotonic decrease in flux from the center to enforce
        shifting: `bool`
            Whether or not a subpixel shift is added as optimization parameter
        """

        # initialize from observation
        seds, morphs, bbox = init_multicomponent_source(
            sky_coord,
            model_frame,
            observations,
            coadd=coadd,
            bg_cutoff=bg_cutoff,
            flux_percentiles=flux_percentiles,
            thresh=thresh,
            symmetric=True,
            monotonic=True,
            min_grad=min_grad,
        )

        if flux_percentiles is None:
            flux_percentiles = (25,)
        K = len(flux_percentiles) + 1
        center = np.array(model_frame.get_pixel(sky_coord), dtype="float")

        components = []
        for k in range(K):

            # higher tolerance on SED: construct parameter explicitly with larger step
            sed = Parameter(
                seds[k],
                name="spectrum",
                step=partial(relative_step, factor=1e-1),
                constraint=PositivityConstraint(),
            )
            spectrum = TabulatedSpectrum(model_frame, sed)

            morphology = ExtendedSourceMorphology(
                model_frame,
                center,
                morphs[k],
                bbox=bbox[1:],
                monotonic=monotonic,
                symmetric=symmetric,
                min_grad=min_grad,
                shifting=shifting,
            )
            component = FactorizedComponent(model_frame, spectrum, morphology)
            components.append(component)

        super().__init__(components)

    @property
    def center(self):
        c = self.__getitem__(0)
        return c.center

from functools import partial

from .initialization import *
from .constraint import (
    PositivityConstraint,
    MonotonicityConstraint,
    SymmetryConstraint,
    L0Constraint,
)
from .constraint import NormalizationConstraint, ConstraintChain, CenterOnConstraint
from .parameter import Parameter, relative_step
from .component import ComponentTree, Factor, FactorizedComponent
from .bbox import Box
from .wavelet import Starlet
from .psf import PSF
from . import fft
from . import interpolation

# make sure that import * above doesn't import its own stock numpy
import autograd.numpy as np


class Spectrum(Factor):
    pass


class TabulatedSpectrum(Spectrum):
    def __init__(self, spectrum, bbox=None):
        if isinstance(spectrum, Parameter):
            assert spectrum.name == "spectrum"
        else:
            constraint = PositivityConstraint()
            step = partial(relative_step, factor=1e-2)
            spectrum = Parameter(
                spectrum, name="spectrum", step=step, constraint=constraint
            )
        self.spectrum = spectrum

        if bbox is None:
            self._bbox = Box(spectrum.shape)
        else:
            assert bbox.shape == spectrum.shape
            self._bbox = bbox

        super().__init__(self.spectrum)

    @property
    def bbox(self):
        return self._bbox

    def get_model(self, *parameters):
        spectrum = self.spectrum
        for p in parameters:
            if p._value.name == "spectrum":
                spectrum = p
        return spectrum


class Morphology(Factor):
    @property
    def center(self):
        return None


class ImageMorphology(Morphology):
    def __init__(self, image, bbox=None, shift=None):
        if isinstance(image, Parameter):
            assert image.name == "image"
        else:
            constraint = PositivityConstraint()
            image = Parameter(
                image, name="image", step=relative_step, constraint=constraint
            )
        self.image = image

        if bbox is None:
            self._bbox = Box(image.shape)
        else:
            assert bbox.shape == image.shape
            self._bbox = bbox

        if shift is None:
            parameters = (image,)
        else:
            assert shift.shape == (2,)
            if isinstance(shift, Parameter):
                assert shift.name == "shift"
            else:
                shift = Parameter(shift, name="shift", step=1e-1)
            parameters = (image, shift)
            # fft helpers
            padding = 10
            self.fft_shape = fft._get_fft_shape(image, image, padding=padding)
            self.shifter_y, self.shifter_x = interpolation.mk_shifter(self.fft_shape)
        self.shift = shift

        super().__init__(*parameters)

    @property
    def bbox(self):
        return self._bbox

    def get_model(self, *parameters):
        image, shift = self.image, self.shift

        # if params are set they are not Parameters, but autograd ArrayBoxes
        # need to access the wrapped class with _value
        for p in parameters:
            if p._value.name == "image":
                image = p
            elif p._value.name == "shift":
                shift = p

        return self._shift_image(shift, image)

    def _shift_image(self, shift, image):
        if shift is not None:
            X = fft.Fourier(image)
            X_fft = X.fft(self.fft_shape, (0, 1))

            # Apply shift in Fourier
            result_fft = (
                X_fft
                * np.exp(self.shifter_y[:, None] * shift[0])
                * np.exp(self.shifter_x[None, :] * shift[1])
            )

            X = fft.Fourier.from_fft(result_fft, self.fft_shape, X.shape, [0, 1])
            return np.real(X.image)
        return image


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


class PointSourceMorphology(Morphology):
    def __init__(self, center, psf):
        assert isinstance(psf, PSF)
        self.psf = psf

        # morph parameters is simply 2D center
        self._center = Parameter(center, name="center", step=1e-1)
        super().__init__(self._center)

        # define bbox
        pixel_center = tuple(np.round(center).astype("int"))
        bottom = pixel_center[0] - psf.shape[1] // 2
        top = pixel_center[0] + psf.shape[1] // 2
        left = pixel_center[1] - psf.shape[2] // 2
        right = pixel_center[1] + psf.shape[2] // 2
        self._bbox = Box.from_bounds((bottom, top), (left, right))

    @property
    def bbox(self):
        return self._bbox

    @property
    def center(self):
        return self._center._data

    def get_model(self, *parameters):
        center = self._center
        for p in parameters:
            if p._value.name == "center":
                center = p

        spec_bbox = Box((0,))
        return self.psf(*center, bbox=spec_bbox @ self._bbox)[0]


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
        frame: `~scarlet.Frame`
            The frame of the full model
        sky_coord: tuple
            Center of the source
        observations: instance or list of `~scarlet.Observation`
            Observation(s) to initialize this source
        """
        spectrum = get_psf_sed(sky_coord, observations, model_frame)
        spectrum = TabulatedSpectrum(spectrum)

        center = np.array(model_frame.get_pixel(sky_coord), dtype="float")
        morphology = PointSourceMorphology(center, model_frame.psf)

        super().__init__(model_frame, spectrum, morphology)


class StarletMorphology(Morphology):
    def __init__(self, image, bbox=None, threshold=0):

        if bbox is None:
            self._bbox = Box(image.shape)
        else:
            assert bbox.shape == image.shape
            self._bbox = bbox

        # Starlet transform of morphologies (n1,n2) with 4 dimensions: (1,lvl,n1,n2), lvl = wavelet scales
        self.transform = Starlet(image)
        # The starlet transform is the model
        coeffs = self.transform.coefficients
        # wavelet-scale norm
        starlet_norm = self.transform.norm
        # One threshold per wavelet scale: thresh*norm
        thresh_array = np.zeros(coeffs.shape) + threshold
        thresh_array = (
            thresh_array * np.array([starlet_norm])[..., np.newaxis, np.newaxis]
        )
        # We don't threshold the last scale
        thresh_array[:, -1, :, :] = 0

        constraint = ConstraintChain(L0Constraint(thresh_array), PositivityConstraint())

        coeffs = Parameter(coeffs, name="coeffs", step=1e-2, constraint=constraint)
        self.coeffs = coeffs
        super().__init__(coeffs)

    @property
    def bbox(self):
        return self._bbox

    def get_model(self, *parameters):
        """ Takes the inverse transform of parameters as starlet coefficients.

        """
        coeffs = self.coeffs
        for p in parameters:
            if p._value.name == "coeffs":
                coeffs = p
        return Starlet(coefficients=coeffs).image[0]


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
            monotonic="flat",
            min_grad=min_grad,
            starlet_thresh=starlet_thresh,
        )
        spectrum = TabulatedSpectrum(sed)
        morphology = StarletMorphology(morph, bbox[1:], thresh)
        super().__init__(model_frame, spectrum, morphology)

        # since we use the starlet for localized source: it has a center
        self._center = np.array(model_frame.get_pixel(sky_coord), dtype="float")

    @property
    def center(self):
        return self._center


class ExtendedSourceMorphology(ImageMorphology):
    def __init__(
        self,
        center,
        image,
        bbox,
        monotonic="flat",
        symmetric=False,
        min_grad=0,
        shifting=False,
    ):

        constraints = []
        # backwards compatibility: monotonic was boolean
        if monotonic is True:
            monotonic = "angle"
        elif monotonic is False:
            monotonic = None
        if monotonic is not None:
            # most astronomical sources are monotonically decreasing
            # from their center
            constraints.append(
                MonotonicityConstraint(neighbor_weight=monotonic, min_gradient=min_grad)
            )

        if symmetric:
            # have 2-fold rotation symmetry around their center ...
            constraints.append(SymmetryConstraint())

        constraints += [
            # ... and are positive emitters
            PositivityConstraint(),
            # prevent a weak source from disappearing entirely
            CenterOnConstraint(),
            # break degeneracies between sed and morphology
            NormalizationConstraint("max"),
        ]
        morph_constraint = ConstraintChain(*constraints)
        image = Parameter(image, name="image", step=1e-2, constraint=morph_constraint)

        self.pixel_center = tuple(np.round(center).astype("int"))
        if shifting:
            shift = Parameter(center - self.pixel_center, name="shift", step=1e-1)
        else:
            shift = None

        super().__init__(image, bbox=bbox, shift=shift)

    @property
    def center(self):
        if self.shift is not None:
            return self.pixel_center + self.shift._data
        else:
            return self.pixel_center


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
        spectrum = TabulatedSpectrum(sed)

        center = np.array(model_frame.get_pixel(sky_coord), dtype="float")
        morphology = ExtendedSourceMorphology(
            center,
            morph,
            bbox[1:],
            monotonic=monotonic,
            symmetric=symmetric,
            min_grad=min_grad,
            shifting=shifting,
        )
        super().__init__(model_frame, spectrum, morphology)


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

    def __init__(
        self,
        model_frame,
        sky_coord,
        observations,
        coadd=None,
        bg_cutoff=None,
        thresh=1.0,
        flux_percentiles=[25,],
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
            spectrum = TabulatedSpectrum(sed)

            morphology = ExtendedSourceMorphology(
                center,
                morphs[k],
                bbox[1:],
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
        c = self.components[0]
        return c.center

from functools import partial

from .initialization import *
from .constraint import PositivityConstraint, MonotonicityConstraint, SymmetryConstraint, L0Constraint
from .constraint import NormalizationConstraint, ConstraintChain, CenterOnConstraint
from .parameter import Parameter, relative_step
from .component import ComponentTree, FunctionComponent, FactorizedComponent
from .bbox import Box
from .wavelet import Starlet, mad_wavelet

# make sure that import * above doesn't import its own stock numpy
import autograd.numpy as np

class RandomSource(FactorizedComponent):
    """Sources with uniform random morphology and sed.

    For cases with no well-defined spatial shape, this source initializes
    a uniform random field and (optionally) matches the SED to match a given
    observation.
    """

    def __init__(self, model_frame, observation=None):
        """Source intialized as random field.

        Parameters
        ----------
        frame: `~scarlet.Frame`
            The frame of the model
        observation: list of `~scarlet.Observation`
            Observation to initialize the SED of the source
        """
        C, Ny, Nx = model_frame.shape
        morph = np.random.rand(Ny, Nx)


        if observation is None:
            seds = np.random.rand(C)
        else:
            try:
                iter(observation)
            except TypeError:
                observation = [observation]
            seds = []
            for obs in observation:
                seds.append(get_best_fit_seds(morph[None], obs.images)[0])

        constraint = PositivityConstraint()
        sed = Parameter(seds, name="sed", step=relative_step, constraint=constraint)
        morph = Parameter(
            morph, name="morph", step=relative_step, constraint=constraint
        )

        super().__init__(model_frame, model_frame.bbox, sed, morph)

class PointSource(FunctionComponent):
    """Source intialized with a single pixel

    Point sources are initialized with the SED of the center pixel,
    and the morphology taken from `frame.psfs`, centered at `sky_coord`.
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
        C, Ny, Nx = model_frame.shape
        self.center = np.array(model_frame.get_pixel(sky_coord), dtype="float")

        # initialize SED from sky_coord
        try:
            iter(observations)
        except TypeError:
            observations = [observations]

        # determine initial SED from peak position
        # SED in the frame for source detection
        seds = []
        for obs in observations:
            _sed = get_psf_sed(sky_coord, obs, model_frame)
            seds.append(_sed)
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

        # set up parameters
        sed = Parameter(
            sed,
            name="sed",
            step=partial(relative_step, factor=1e-2),
            constraint=PositivityConstraint(),
        )
        center = Parameter(self.center, name="center", step=1e-1)

        # define bbox
        pixel_center = tuple(np.round(center).astype("int"))
        front, back = 0, C
        bottom = pixel_center[0] - model_frame.psf.shape[1] // 2
        top = pixel_center[0] + model_frame.psf.shape[1] // 2
        left = pixel_center[1] - model_frame.psf.shape[2] // 2
        right = pixel_center[1] + model_frame.psf.shape[2] // 2
        bbox = Box.from_bounds((front, back), (bottom, top), (left, right))

        super().__init__(model_frame, bbox, sed, center, self._psf_wrapper)

    def _psf_wrapper(self, *parameters):
        return self.model_frame.psf.__call__(*parameters, bbox=self.bbox)[0]

class StarletSource(FunctionComponent):
    """Source intialized with starlet coefficients.

    Sources are initialized with the SED of the center pixel,
    and the morphologies are initialised as ExtendedSources
    and transformed into starlet coefficients.
    """
    def __init__(
        self,
        frame,
        sky_coord,
        observations,
        coadd=None,
        bg_cutoff=None,
        thresh=1.0,
        starlet_thresh=5,
        min_grad=0.1,
    ):
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
        shifting: `bool`
            Whether or not a subpixel shift is added as optimization parameter
        """
        center = np.array(frame.get_pixel(sky_coord), dtype="float")
        self.pixel_center = tuple(np.round(center).astype("int"))

        # initialize SED from sky_coord
        try:
            iter(observations)
        except TypeError:
            observations = [observations]

        # initialize from observation
        sed, image_morph, bbox = init_extended_source(
            sky_coord,
            frame,
            observations,
            coadd=coadd,
            bg_cutoff=bg_cutoff,
            thresh=thresh,
            symmetric=True,
            monotonic=True,
            min_grad=min_grad,
        )
        noise =[]
        for obs in observations:
            noise += [mad_wavelet(obs.images) * \
                    np.sqrt(np.sum(obs._diff_kernels.image**2, axis = (-2,-1)))]
        noise = np.concatenate(noise)
        # Threshold in units of noise
        thresh = starlet_thresh * np.sqrt(np.sum((sed*noise) ** 2))

        # Starlet transform of morphologies (n1,n2) with 4 dimensions: (1,lvl,n1,n2), lvl = wavelet scales
        self.transform = Starlet(image_morph)
        #The starlet transform is the model
        morph = self.transform.coefficients
        # wavelet-scale norm
        starlet_norm = self.transform.norm
        #One threshold per wavelet scale: thresh*norm
        thresh_array = np.zeros(morph.shape) + thresh
        thresh_array = thresh_array * np.array([starlet_norm])[..., np.newaxis, np.newaxis]
        # We don't threshold the last scale
        thresh_array[:,-1,:,:] = 0

        sed = Parameter(
            sed,
            name="sed",
            step=partial(relative_step, factor=1e-2),
            constraint=PositivityConstraint(),
        )

        morph_constraint = ConstraintChain(*[L0Constraint(thresh_array), PositivityConstraint()])

        morph = Parameter(morph, name="morph", step=1.e-2, constraint=morph_constraint)

        super().__init__(frame, bbox, sed, morph, self._iuwt)

    @property
    def center(self):
        if len(self.parameters) == 3:
            return self.pixel_center + self.shift
        else:
            return self.pixel_center

    def _iuwt(self, param):
        """ Takes the inverse transform of parameters as starlet coefficients.

        """
        return Starlet(coefficients = param).image[0]

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
        center = np.array(model_frame.get_pixel(sky_coord), dtype="float")
        self.pixel_center = tuple(np.round(center).astype("int"))

        if shifting:
            shift = Parameter(center - self.pixel_center, name="shift", step=1e-1)
        else:
            shift = None

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
            min_grad = min_grad
        )

        sed = Parameter(
            sed,
            name="sed",
            step=partial(relative_step, factor=1e-2),
            constraint=PositivityConstraint(),
        )

        constraints = []

        # backwards compatibility: monotonic was boolean
        if monotonic is True:
            monotonic = "angle"
        elif monotonic is False:
            monotonic = None
        if monotonic is not None:
            # most astronomical sources are monotonically decreasing
            # from their center
            constraints.append(MonotonicityConstraint(neighbor_weight=monotonic, min_gradient=min_grad))

        if symmetric:
            # have 2-fold rotation symmetry around their center ...
            constraints.append(SymmetryConstraint())

        constraints += [
            # ... and are positive emitters
            PositivityConstraint(),
            # prevent a weak source from disappearing entirely
            # CenterOnConstraint(),
            # break degeneracies between sed and morphology
            NormalizationConstraint("max"),
        ]
        morph_constraint = ConstraintChain(*constraints)

        morph = Parameter(morph, name="morph", step=1e-2, constraint=morph_constraint)

        super().__init__(model_frame, bbox, sed, morph, shift=shift)

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
            then `flux_percentiles=[25]`, a single component with 25% of the flux
            as the primary source.
        symmetric: `bool`
            Whether or not to enforce symmetry.
        monotonic: ['flat', 'angle', 'nearest'] or None
            Which version of monotonic decrease in flux from the center to enforce
        shifting: `bool`
            Whether or not a subpixel shift is added as optimization parameter
        """
        self.symmetric = symmetric
        self.monotonic = monotonic
        self.coords = sky_coord
        center = np.array(model_frame.get_pixel(sky_coord), dtype="float")
        pixel_center = tuple(np.round(center).astype("int"))

        if shifting:
            shift = Parameter(center - pixel_center, name="shift", step=1e-1)
        else:
            shift = None

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

        constraints = []

        # backwards compatibility: monotonic was boolean
        if monotonic is True:
            monotonic = "angle"
        elif monotonic is False:
            monotonic = None
        if monotonic is not None:
            # most astronomical sources are monotonically decreasing
            # from their center
            constraints.append(MonotonicityConstraint(neighbor_weight=monotonic, min_gradient=min_grad))

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

        components = []
        for k in range(len(seds)):
            sed = Parameter(
                seds[k],
                name="sed",
                step=partial(relative_step, factor=1e-1),
                constraint=PositivityConstraint(),
            )
            morph = Parameter(
                morphs[k], name="morph", step=1e-2, constraint=morph_constraint
            )
            components.append(
                FactorizedComponent(model_frame, bbox, sed, morph, shift=shift)
            )
            components[-1].pixel_center = pixel_center
        super().__init__(components)

    @property
    def shift(self):
        c = self.components[0]
        return c.shift

    @property
    def center(self):
        c = self.components[0]
        if len(c.parameters) == 3:
            return c.pixel_center + c.shift
        else:
            return c.pixel_center

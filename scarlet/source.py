import autograd.numpy as np
from functools import partial
import logging
import sys
from . import initialization as init
from . import operator
from .bbox import Box, overlapped_slices
from .component import Component, CombinedComponent, FactorizedComponent
from .constraint import CenterOnConstraint, PositivityConstraint
from .morphology import (
    ImageMorphology,
    PointSourceMorphology,
    StarletMorphology,
    ExtendedSourceMorphology,
    GaussianMorphology,
    SpergelMorphology,
)
from .parameter import Parameter, relative_step
from .spectrum import TabulatedSpectrum, StaticSpectrum

logger = logging.getLogger("scarlet.source")


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
            model = self.model_to_box(frame.bbox, model)
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
        morphology = ImageMorphology(model_frame, image)

        if observations is None:
            spectrum = np.random.rand(C)
        else:
            spectrum = init.get_best_fit_spectrum(image[None], observations)[0]

        # default is step=1e-2, using larger steps here becaus SED is probably uncertain
        spectrum = Parameter(
            spectrum,
            name="spectrum",
            step=partial(relative_step, factor=1e-1),
            constraint=PositivityConstraint(),
        )
        spectrum = TabulatedSpectrum(model_frame, spectrum)

        super().__init__(model_frame, spectrum, morphology)


class PointSource(FactorizedComponent):
    """Point-Source model

    Point sources modeled as `model_frame.psfs`, centered at `sky_coord`.
    Their SEDs are taken from `observations` at the center pixel.
    """

    def __init__(self, model_frame, sky_coord, observations, shiftmultiple=1):
        """Source intialized with a single pixel

        Parameters
        ----------
        model_frame: `~scarlet.Frame`
            The frame of the model
        sky_coord: tuple
            Center of the source
        observations: instance or list of `~scarlet.Observation`
            Observation(s) to initialize this source
        """
        if not hasattr(observations, "__iter__"):
            observations = (observations,)

        center = model_frame.get_pixel(sky_coord)
        center = Parameter(center, name="center", step=3e-2*shiftmultiple)
        morphology = PointSourceMorphology(model_frame, center, shiftmultiple=shiftmultiple)

        # get spectrum from peak pixel, correct for PSF
        spectra = init.get_pixel_spectrum(sky_coord, observations, correct_psf=True)
        spectrum = np.concatenate(spectra, axis=0)
        if np.sum(np.isnan(spectrum))>0:
            spectrum[np.isnan(spectrum)]=1e-10
            print('Replacing nans with small number')
         
        
        noise_rms = np.concatenate(
            [np.array(np.mean(obs.noise_rms, axis=(1, 2))) for obs in observations]
        ).reshape(-1)
        spectrum = TabulatedSpectrum(model_frame, spectrum, min_step=noise_rms)

        super().__init__(model_frame, spectrum, morphology)

        # retain center as attribute
        self.center = morphology.center


class GaussianSource(FactorizedComponent):
    """Gassian-shaped source

    Their SEDs are initialized from `observations` at the center pixel.
    """

    def __init__(self, model_frame, sky_coord, sigma, ellipticity, observations):
        """Gassian-shaped source intialized with a single pixel SED

        Parameters
        ----------
        model_frame: `~scarlet.Frame`
            The frame of the model
        sky_coord: tuple
            Center of the source
        sigma: float
            Standard deviation of the Gaussian
        ellipticity: array or None
            Two-component ellipticity (e1,e2)
        observations: instance or list of `~scarlet.Observation`
            Observation(s) to initialize this source
        """
        if not hasattr(observations, "__iter__"):
            observations = (observations,)

        center = model_frame.get_pixel(sky_coord)
        center = Parameter(center, name="center", step=0.01)
        sigma = Parameter(np.array((sigma,)), name="radius", step=relative_step)
        if ellipticity is not None:
            ellipticity = Parameter(ellipticity, name="ellipticity", step=0.01)

        morphology = GaussianMorphology(
            model_frame, center, sigma, ellipticity=ellipticity
        )

        # get spectrum from peak pixel, don't correct for PSF (extended source)
        spectra = init.get_pixel_spectrum(sky_coord, observations, correct_psf=False)
        spectrum = np.concatenate(spectra, axis=0)

        # get peak pixel value from model
        vmax = morphology.f(0)
        spectrum /= vmax

        # noise rms for step sizes
        noise_rms = np.concatenate(
            [np.array(np.mean(obs.noise_rms, axis=(1, 2))) for obs in observations]
        ).reshape(-1)
        noise_rms /= vmax

        # make spectrum model
        spectrum = TabulatedSpectrum(model_frame, spectrum, min_step=noise_rms)

        super().__init__(model_frame, spectrum, morphology)

        # retain center as attribute
        self.center = morphology.center


class SpergelSource(FactorizedComponent):
    """Source based on the Spergel (2010) profile

    Their SEDs are initialized from `observations` at the center pixel.
    """

    def __init__(self, model_frame, sky_coord, nu, rhalf, ellipticity, observations):
        """Spergel (2010) profile source intialized with a single pixel SED

        Parameters
        ----------
        model_frame: `~scarlet.Frame`
            The frame of the model
        sky_coord: tuple
            Center of the source
        nu: float
            Bessel function order.
        rhalf: float
            Half-light radius in frame pixels.
        ellipticity: array or None
            Two-component ellipticity (e1,e2)
        observations: instance or list of `~scarlet.Observation`
            Observation(s) to initialize this source
        """
        if not hasattr(observations, "__iter__"):
            observations = (observations,)

        center = model_frame.get_pixel(sky_coord)
        center = Parameter(center, name="center", step=0.01)
        nu = Parameter(np.array((nu,), dtype="float"), name="nu", step=0.01)
        rstep = partial(relative_step, factor=0.01)
        rhalf = Parameter(np.array((rhalf,), dtype="float"), name="radius", step=rstep)
        if ellipticity is not None:
            ellipticity = Parameter(ellipticity, name="ellipticity", step=0.01)

        morphology = SpergelMorphology(
            model_frame, center, nu, rhalf, ellipticity=ellipticity
        )

        # get spectrum from peak pixel, don't correct for PSF (extended source)
        spectra = init.get_pixel_spectrum(sky_coord, observations, correct_psf=False)
        spectrum = np.concatenate(spectra, axis=0)

        # get peak pixel value from model
        vmax = morphology.f(0)
        spectrum /= vmax

        # noise rms for step sizes
        noise_rms = np.concatenate(
            [np.array(np.mean(obs.noise_rms, axis=(1, 2))) for obs in observations]
        ).reshape(-1)
        noise_rms /= vmax

        # make spectrum model
        spectrum = TabulatedSpectrum(model_frame, spectrum, min_step=noise_rms)

        super().__init__(model_frame, spectrum, morphology)

        # retain center as attribute
        self.center = morphology.center


class CompactExtendedSource(FactorizedComponent):
    def __init__(
        self,
        model_frame,
        sky_coord,
        observations,
        shifting=False,
        resizing=True,
        boxsize=None,
    ):
        """Compact extended source model

        The model is initialized from `observations` with a point-source morphology
        and a spectrum from its peak pixel.

        During optimization it enforces positivitiy for spectrum and morphology,
        as well as monotonicity of the morphology.

        Parameters
        ----------
        model_frame: `~scarlet.Frame`
            The frame of the model
        sky_coord: tuple
            Center of the source
        observations: instance or list of `~scarlet.observation.Observation`
            Observation(s) to initialize this source.
        shifting: `bool`
            Whether or not a subpixel shift is added as optimization parameter
        resizing : bool
            Whether or not to change the size of the source box.
        boxsize: int or None
            Spatial size of the source box
        """
        if not hasattr(observations, "__iter__"):
            observations = (observations,)

        # initialize morphology from model_frame psf
        assert model_frame.psf is not None
        morph, bbox = self.init_morph(model_frame, sky_coord, boxsize=boxsize)
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
            resizing=resizing,
        )

        # get spectrum from peak pixel, correct for PSF
        spectra = init.get_pixel_spectrum(sky_coord, observations, correct_psf=True)
        spectrum = np.concatenate(spectra, axis=0)
        spectrum /= morph.sum()
        noise_rms = np.concatenate(
            [np.array(np.mean(obs.noise_rms, axis=(1, 2))) for obs in observations]
        ).reshape(-1)
        spectrum = TabulatedSpectrum(model_frame, spectrum, min_step=noise_rms)

        # set up model with its parameters
        super().__init__(model_frame, spectrum, morphology)

        # retain center as attribute
        self.center = morphology.center

    @staticmethod
    def init_morph(frame, sky_coord, boxsize=None):
        """Initialize a source just like `init_extended_morphology`,
        but with the morphology of a point source.

        Parameters
        ----------
        frame: `~scarlet.Frame`
            The model frame
        sky_coord: tuple
            Center of the source
        boxsize: int or None
            Size of morph box

        Returns
        -------
        morph, bbox
        """

        # position in frame coordinates
        center = frame.get_pixel(sky_coord)
        center_index = np.round(center).astype("int")

        # morphology initialized as a point source
        morph_ = frame.psf.get_model().mean(axis=0)
        origin = (
            center_index[0] - (morph_.shape[0] // 2),
            center_index[1] - (morph_.shape[1] // 2),
        )
        bbox_ = Box(morph_.shape, origin=origin)

        if boxsize is None:
            size = max(morph_.shape)
            boxsize = init.get_minimal_boxsize(size)

        # adjust box size to conform with extended sources
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



class StaticSource(FactorizedComponent):
    def __init__(
        self,
        model_frame,
        sky_coord,
        observations,
        thresh=1.0,
        shifting=False,
        symmetric=False,
        resizing=True,
        boxsize=None,
        shiftmultiple=1
    ):
        """Static extended source model

        The model is initialized from `observations` with a symmetric and
        monotonic profile and a spectrum from its peak pixel.

        During optimization it enforces positivitiy for spectrum and morphology,
        as well as monotonicity of the morphology.

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
        compact: `bool`
            Initialize with the shape of a point source
        shifting: `bool`
            Whether or not a subpixel shift is added as optimization parameter
        resizing : bool
            Whether or not to change the size of the source box.
        boxsize: int or None
            Spatial size of the source box
        """
        if not hasattr(observations, "__iter__"):
            observations = (observations,)

        # get center pixel spectrum
        # this is from convolved image: weighs higher emission *and* narrow PSF
        spectra = init.get_psf_spectrum(sky_coord, observations)
        # initialize morphology
        # compute optimal SNR coadd for detection 
        bands0,bandind = np.unique([obs.channels[0][0] for obs in observations],return_index=True)
        epochs = np.asarray([obs.channels[0][0] for obs in observations]) 
        bands = epochs[bandind] 
        repeats = np.asarray([np.sum([epochs==b]) for b in bands],dtype=int)
        self.repeats = repeats
        image, std = init.build_initialization_image(observations, spectra=spectra)
        # make monotonic morphology, trimmed to box with pixels above std
        morph, bbox = self.init_morph(
            model_frame,
            sky_coord,
            image,
            std,
            thresh=thresh,
            symmetric=True,
            monotonic="flat",
            min_grad=0,
            boxsize=boxsize,
        )

        center = model_frame.get_pixel(sky_coord)
        morphology = ExtendedSourceMorphology(
            model_frame,
            center,
            morph,
            bbox=bbox,
            monotonic="angle",
            symmetric=symmetric,
            min_grad=0,
            shifting=shifting,
            resizing=resizing,
            shiftmultiple=shiftmultiple
        )
        
        # find best-fit spectra for morph from init coadd
        # assumes img only has that source in region of the box
        rms = [] 
        bandmatch = [] 
        for a,b in enumerate(bands):
            bandsingle=[]
            for i,e in enumerate(epochs):
                if e==b:
                    bandsingle.append(i)
            bandmatch.append(bandsingle)
        
        for bind,b in enumerate(bands):
            rms.append(np.min(np.concatenate(
                [np.array(np.median(obs.noise_rms, axis=(1, 2))) for obs in observations[bandmatch[bind][0]:bandmatch[bind][-1]+1]]
            ).reshape(-1)))
        epochs = [obs.channels[0][0] for obs in observations]
        detect_all, std_all = init.build_initialization_image(observations)
        
        box_3D = Box((model_frame.C,)) @ bbox
        boxed_detect = box_3D.extract_from(detect_all)
        boxed_std_all = box_3D.extract_from(std_all)
        
        spectrum = init.get_best_fit_spectrum((morph,), boxed_detect, boxed_std = boxed_std_all)
        medianspectrum=[np.median(spectrum[b][spectrum[b]>0]) for b in bandmatch]
         
        noise_rms = np.concatenate(
            [np.array(np.mean(obs.noise_rms, axis=(1, 2))) for obs in observations]
        ).reshape(-1)
        
        quiescentspectrum=np.asfarray(medianspectrum)
        spectrum = StaticSpectrum(model_frame, quiescentspectrum, min_step=rms, repeats=self.repeats) 
        # set up model with its parameters
        super().__init__(model_frame, spectrum, morphology)
        # retain center as attribute
        self.center = morphology.center

    @staticmethod
    def init_morph(
        frame,
        sky_coord,
        detect,
        detect_std,
        thresh=1,
        symmetric=True,
        monotonic="flat",
        min_grad=0,
        boxsize=None,
    ):
        """Initialize the source that is symmetric and monotonic
        See `ExtendedSource` for a description of the parameters

        Returns
        -------
        morph, bbox
        """

        # position in frame coordinates
        center = frame.get_pixel(sky_coord)
        center_index = np.round(center).astype("int")

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
        morph, bbox = init.trim_morphology(
            center_index, im, bg_thresh=threshold, boxsize=boxsize
        )

        # normalize to unity at peak pixel for the imposed normalization
        if morph.sum() > 0:
            morph /= morph.max()
        else:
            msg = f"No flux in morphology model for source at {sky_coord}"
            logger.warning(msg)
            morph = CenterOnConstraint(tiny=1)(morph, 0)

        # for very noise inits, there is only 1 or few pixels in the center:
        # pad morph with the shape of the PSF
        if frame.psf is not None:
            psf_morph, _ = CompactExtendedSource.init_morph(
                frame, sky_coord, boxsize=max(bbox.shape)
            )
            morph = np.maximum(morph, psf_morph)

        return morph, bbox


class SingleExtendedSource(FactorizedComponent):
    def __init__(
        self,
        model_frame,
        sky_coord,
        observations,
        thresh=1.0,
        shifting=False,
        resizing=True,
        boxsize=None,
        quiescent=False
    ):
        """Extended source model

        The model is initialized from `observations` with a symmetric and
        monotonic profile and a spectrum from its peak pixel.

        During optimization it enforces positivitiy for spectrum and morphology,
        as well as monotonicity of the morphology.

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
        compact: `bool`
            Initialize with the shape of a point source
        shifting: `bool`
            Whether or not a subpixel shift is added as optimization parameter
        resizing : bool
            Whether or not to change the size of the source box.
        boxsize: int or None
            Spatial size of the source box
        """
        if not hasattr(observations, "__iter__"):
            observations = (observations,)

        # get center pixel spectrum
        # this is from convolved image: weighs higher emission *and* narrow PSF
        spectra = init.get_pixel_spectrum(sky_coord, observations, quiescent = quiescent) 
        # initialize morphology
        # compute optimal SNR coadd for detection

        detect_all, std_all = init.build_initialization_image(observations)
        image, std = init.build_initialization_image(observations, spectra=spectra)
        # make monotonic morphology, trimmed to box with pixels above std
        morph, bbox = self.init_morph(
            model_frame,
            sky_coord,
            image,
            std,
            thresh=thresh,
            symmetric=True,
            monotonic="flat",
            min_grad=0,
            boxsize=boxsize,
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
            resizing=resizing,
        )

        # find best-fit spectra for morph from init coadd
        # assumes img only has that source in region of the box
        
        detect_all, std_all = init.build_initialization_image(observations)
        box_3D = Box((model_frame.C,)) @ bbox
        boxed_detect = box_3D.extract_from(detect_all)
        if not quiescent: 
            spectrum = init.get_best_fit_spectrum((morph,), boxed_detect)
        else:
            spectrum = init.get_best_fit_spectrum((morph,), boxed_detect, quiescent=True, epochs=bands)
        
        if np.sum(np.isnan(spectrum))>0:
            spectrum[np.isnan(spectrum)]=1e-5
            print('Replacing nans with small number')
         
        noise_rms = np.concatenate(
            [np.array(np.mean(obs.noise_rms, axis=(1, 2))) for obs in observations]
        ).reshape(-1)
        spectrum = TabulatedSpectrum(model_frame, spectrum, min_step=noise_rms)#, bands=bands, epochs=epochs)

        # set up model with its parameters
        super().__init__(model_frame, spectrum, morphology)
        

        # retain center as attribute
        self.center = morphology.center

    @staticmethod
    def init_morph(
        frame,
        sky_coord,
        detect,
        detect_std,
        thresh=1,
        symmetric=True,
        monotonic="flat",
        min_grad=0,
        boxsize=None,
    ):
        """Initialize the source that is symmetric and monotonic
        See `ExtendedSource` for a description of the parameters

        Returns
        -------
        morph, bbox
        """

        # position in frame coordinates
        center = frame.get_pixel(sky_coord)
        center_index = np.round(center).astype("int")

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
        morph, bbox = init.trim_morphology(
            center_index, im, bg_thresh=threshold, boxsize=boxsize
        )

        # normalize to unity at peak pixel for the imposed normalization
        if morph.sum() > 0:
            morph /= morph.max()
        else:
            msg = f"No flux in morphology model for source at {sky_coord}"
            logger.warning(msg)
            morph = CenterOnConstraint(tiny=1)(morph, 0)

        # for very noise inits, there is only 1 or few pixels in the center:
        # pad morph with the shape of the PSF
        if frame.psf is not None:
            psf_morph, _ = CompactExtendedSource.init_morph(
                frame, sky_coord, boxsize=max(bbox.shape)
            )
            morph = np.maximum(morph, psf_morph)

        return morph, bbox


class StarletSource(FactorizedComponent):
    """Source with a starlet morphology model

    Sources are initialized as `~scarlet.ExtendedSource`, and the morphology is then
    transformed into starlet coefficients.
    """

    def __init__(
        self,
        model_frame,
        sky_coord=None,
        observations=None,
        spectrum=None,
        thresh=1.0,
        monotonic=False,
        starlet_thresh=5e-3,
        boxsize=None,
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
        spectrum: `numpy.ndarray` or `scarlet.Parameter`
            Initial spectrum, otherwise given by `ExtendedSource` initialization
        monotonic: bool
            Whether to constrain every starlet scale to be monotonic; otherwise they are
            hard-thresholded by `starlet_thresh`.
        starlet_thresh: `float`
            Multiple of the backround RMS used as a
            flux cutoff for starlet threshold (usually between 5 and 3).
        boxsize: int or None
            Spatial size of the source box
        """
        if sky_coord is None:
            source = RandomSource(model_frame,)
        else:
            source = ExtendedSource(
                model_frame, sky_coord, observations, thresh=thresh, boxsize=boxsize
            )

        source = StarletSource.from_source(
            source, monotonic=monotonic, starlet_thresh=starlet_thresh
        )

        if spectrum is not None:
            if isinstance(spectrum, Parameter):
                assert spectrum.name == "spectrum"
            else:
                noise_rms = np.concatenate(
                    [
                        np.array(np.mean(obs.noise_rms, axis=(1, 2)))
                        for obs in observations
                    ]
                ).reshape(-1)
                spectrum = TabulatedSpectrum(model_frame, spectrum, min_step=noise_rms)

            source.children[0] = spectrum

        # still need to init *this* source
        super().__init__(source.frame, *source.children)

    @classmethod
    def from_source(cls, source, monotonic=False, starlet_thresh=5e-3):
        assert isinstance(source, FactorizedComponent)

        frame = source.frame
        spectrum, morphology = source.children
        morph = morphology.get_model()
        bbox = morphology.bbox

        # transform to starlets
        morphology = StarletMorphology(
            frame, morph, bbox=bbox, monotonic=monotonic, threshold=starlet_thresh
        )

        # this trick gets us the proper class while call init on the base class
        obj = cls.__new__(cls)
        super(StarletSource, obj).__init__(frame, spectrum, morphology)
        return obj

class StaticMultiExtendedSource(CombinedComponent):
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
        shiftmultiple=1,
        symmetric=False,
        shifting=False,
        resizing=True,
        boxsize=None,
    ):
        """Create multi-component extended source.

        Parameters
        ----------
        model_frame: `~scarlet.Frame`
            The frame of the model
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
        resizing : bool
            Whether or not to change the size of the source box.
        boxsize: int or None
            Spatial size of the source box
        """

        if flux_percentiles is None:
            flux_percentiles = (25,)
        assert K == len(flux_percentiles) + 1

        # initialize from observation
        if not hasattr(observations, "__iter__"):
            observations = (observations,)

        # start off with regular ExtendedSource
        source = StaticSource(
            model_frame, sky_coord, observations, thresh=thresh, boxsize=boxsize
        )
        _, morphology = source.children
        morphs, boxes = self.init_morphs(morphology, flux_percentiles)
        
        # find best-fit spectra for each of morph from the observations
        # assumes observations only have that one source in region of the box
        bands0,bandind = np.unique([obs.channels[0][0] for obs in observations],return_index=True)  
        epochs = np.asarray([obs.channels[0][0] for obs in observations])
        bands = epochs[bandind]
        repeats = np.asarray([np.sum([epochs==b]) for b in bands],dtype=int) 
        self.repeats = repeats
        bandmatch = [] 
       
        for b in bands:
            bandsingle=[]
            for i,e in enumerate(epochs):
                if e==b:
                    bandsingle.append(i)
            bandmatch.append(bandsingle)
        bandmatch = np.asarray(bandmatch) 
                     
        detect_all, std_all = init.build_initialization_image(observations)
        
        box_3D = Box((model_frame.C,)) @ boxes[0]
        boxed_detect = box_3D.extract_from(detect_all)
        boxed_std_all = box_3D.extract_from(std_all)

        spectrum = init.get_best_fit_spectrum(morphs, boxed_detect, boxed_std = boxed_std_all) 
        spectrum = np.asarray(spectrum)
        medianspectrum=[np.median(spectrum[:,b],axis=1) for b in bandmatch]
        
        noise_rms = np.concatenate(
            [np.array(np.mean(obs.noise_rms, axis=(1, 2))) for obs in observations]
        ).reshape(-1)
        
        rms = [np.median(noise_rms[b]) for b in bandmatch]
        quiescentspectrum=np.asfarray(medianspectrum)
        
        # create one component for each spectrum and morphology
        components = []
        center = model_frame.get_pixel(sky_coord)
        for k in range(K):
            spectrum = StaticSpectrum(model_frame, quiescentspectrum[:,k], min_step=rms, repeats=self.repeats)
            
            morphology = ExtendedSourceMorphology(
                model_frame,
                center,
                morphs[k],
                bbox=boxes[k],
                monotonic="angle",
                symmetric=symmetric,
                min_grad=0,
                shifting=shifting,
                shiftmultiple=1,
                resizing=resizing,
            )
            self.center = morphology.center
            component = FactorizedComponent(model_frame, spectrum, morphology)
            components.append(component)

        super().__init__(components)

    @staticmethod
    def init_morphs(morphology, flux_percentiles):

        morph = morphology.get_model()
        bbox = morphology.bbox

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
        resizing=True,
        boxsize=None,
    ):
        """Create multi-component extended source.

        Parameters
        ----------
        model_frame: `~scarlet.Frame`
            The frame of the model
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
        resizing : bool
            Whether or not to change the size of the source box.
        boxsize: int or None
            Spatial size of the source box
        """

        if flux_percentiles is None:
            flux_percentiles = (25,)
        assert K == len(flux_percentiles) + 1

        # initialize from observation
        if not hasattr(observations, "__iter__"):
            observations = (observations,)

        # start off with regular ExtendedSource
        source = ExtendedSource(
            model_frame, sky_coord, observations, thresh=thresh, boxsize=boxsize
        )
        _, morphology = source.children
        morphs, boxes = self.init_morphs(morphology, flux_percentiles)

        # find best-fit spectra for each of morph from the observations
        # assumes observations only have that one source in region of the box
        detect_all, std_all = init.build_initialization_image(observations)
        box_3D = Box((model_frame.C,)) @ boxes[0]
        boxed_detect = box_3D.extract_from(detect_all)
        spectra = init.get_best_fit_spectrum(morphs, boxed_detect)
        noise_rms = np.concatenate(
            [np.array(np.mean(obs.noise_rms, axis=(1, 2))) for obs in observations]
        ).reshape(-1)

        # create one component for each spectrum and morphology
        components = []
        center = model_frame.get_pixel(sky_coord)
        for k in range(K):

            spectrum = TabulatedSpectrum(
                model_frame, spectra[k], min_step=noise_rms / 10
            )

            morphology = ExtendedSourceMorphology(
                model_frame,
                center,
                morphs[k],
                bbox=boxes[k],
                monotonic="angle",
                symmetric=False,
                min_grad=0,
                shifting=shifting,
                resizing=resizing,
            )
            self.center = morphology.center
            component = FactorizedComponent(model_frame, spectrum, morphology)
            components.append(component)

        super().__init__(components)

    @staticmethod
    def init_morphs(morphology, flux_percentiles):

        morph = morphology.get_model()
        bbox = morphology.bbox

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
    resizing=True,
    boxsize=None,
):
    """Create extended sources with either a single component or multiple components.

    If `K== 1`, a single instance of `SingleExtendedSource` is returned, otherwise
    and instance of `MultiExtendedSource` is returned.
    """

    if compact:
        return CompactExtendedSource(
            model_frame,
            sky_coord,
            observations,
            shifting=shifting,
            resizing=resizing,
            boxsize=boxsize,
        )
    if K == 1:
        return SingleExtendedSource(
            model_frame,
            sky_coord,
            observations,
            thresh=thresh,
            shifting=shifting,
            resizing=resizing,
            boxsize=boxsize,
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
            resizing=resizing,
            boxsize=boxsize,
        )

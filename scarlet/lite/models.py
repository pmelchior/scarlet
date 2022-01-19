# This module contains classes and methods to use the so-called "lite" version of scarlet.
# In this instance "lite" just means scarlet stripped down to the bare essentials needed
# to deblend observations from a single instrument, where all of the observations have
# been resampled to the same pixel grid (but may have differing PSFs).
# Because some of the assumptions and constraints are different, this module is designed
# to run faster and use less memory than the standard scarlet modules while obtaining
# a similar log likelihood.
import numpy as np

from scarlet.lite.measure import weight_sources

from ..bbox import overlapped_slices, Box
from ..renderer import convolve
from .. import fft, interpolation, initialization
from ..constraint import  MonotonicityConstraint
from .utils import insert_image


class LiteComponent:
    """A base component in scarlet lite

    If `sed` and `morph` are arrays and not `LiteParameter`s then the
    component is not `initialized` and must still be initialized by
    another function.

    Parameters
    ----------
    center: `tuple` of `int`
        Location of the center pixel of the component in the full blend.
    bbox: `scarlet.bbox.Box`
        The bounding box for this component
    sed: `numpy.ndarray`
        The array of values for the SED `(bands,)`
    morph: `numpy.ndarray`
        The `(height, wdidth)` array of values for the morphology.
    initialized: `bool`
        Whether or not the component has been initialized.
    bg_thresh: `float`
        Level of the background thresh, required by some parameterizations.
    bg_rms: `float`
        The RMS of the background, required by some parameterizations.
    """
    def __init__(self, center, bbox, sed=None, morph=None, initialized=False,
                 bg_thresh=0.25, bg_rms=0):
        self._center = center
        self._bbox = bbox
        self._sed = sed
        self._morph = morph
        self.initialized = initialized
        self.bg_thresh = bg_thresh
        self.bg_rms = bg_rms

    @property
    def center(self):
        """The central locaation of the peak"""
        return self._center

    @property
    def bbox(self):
        """The bounding box that contains the component in the full image"""
        return self._bbox

    @property
    def sed(self):
        """The array of SED values"""
        return self._sed

    @property
    def morph(self):
        """The array of morphology values"""
        return self._morph

    def resize(self):
        """Test whether or not the component needs to be resized
        """
        # No need to resize if there is no size threshold.
        # To allow box sizing but no thresholding use `bg_thresh=0`.
        if self.bg_thresh is None:
            return False

        morph = self.morph
        size = max(morph.shape)

        # shrink the box? peel the onion
        dist = 0
        while (
            np.all(morph[dist, :] == 0)
            and np.all(morph[-dist, :] == 0)
            and np.all(morph[:, dist] == 0)
            and np.all(morph[:, -dist] == 0)
        ):
            dist += 1

        new_size = initialization.get_minimal_boxsize(size - 2 * dist)
        if new_size < size:
            dist = (size - new_size) // 2
            self.bbox.origin = (self.bbox.origin[0], self.bbox.origin[1]+dist, self.bbox.origin[2]+dist)
            self.bbox.shape = (self.bbox.shape[0], new_size, new_size)
            self._morph.shrink(dist)
            self.slices = overlapped_slices(self.model_bbox, self.bbox)
            return True

        # grow the box?
        model = self.get_model()
        edge_flux = np.array([
            np.sum(model[:, 0]),
            np.sum(model[:, -1]),
            np.sum(model[0, :]),
            np.sum(model[-1, :]),
        ])

        edge_mask = np.array([
            np.sum(model[:, 0] > 0),
            np.sum(model[:, -1] > 0),
            np.sum(model[0, :] > 0),
            np.sum(model[-1, :] > 0),
        ])

        if np.any(edge_flux/edge_mask > self.bg_thresh*self.bg_rms[:, None, None]):
            new_size = initialization.get_minimal_boxsize(size + 1)
            dist = (new_size - size) // 2
            self.bbox.origin = (self.bbox.origin[0], self.bbox.origin[1]-dist, self.bbox.origin[2]-dist)
            self.bbox.shape = (self.bbox.shape[0], new_size, new_size)
            self._morph.grow(self.bbox.shape[1:], dist)
            self.slices = overlapped_slices(self.model_bbox, self.bbox)
            return True
        return False

    def __str__(self):
        return "LiteComponent"

    def __repr__(self):
        return "LiteComponent"


class LiteFactorizedComponent(LiteComponent):
    """Implementation of a `FactorizedComponent` for simplified observations.
    """
    def __init__(self, sed, morph, center, bbox, model_bbox, bg_rms, bg_thresh=0.25, floor=1e-20,
                 fit_center_radius=1):
        """Initialize the component.

        Parameters
        ----------
        sed: `LiteParameter`
            The parameter to store and update the SED.
        morph: `LiteParameter`
            The parameter to store and update the morphology.
        center: `array-like`
            The center `(y,x)` of the source in the full model.
        bbox: `~scarlet.bbox.Box`
            The `Box` in the `model_bbox` that contains the source.
        model_bbox: `~scarlet.bbox.Box`
            The `Box` that contains the model.
            This is simplified from the main scarlet, where the model exists
            in a `frame`, which primarily exists because not all
            observations in main scarlet will use the same set of bands.
        bg_rms: `numpy.array`
            The RMS of the background used to threshold, grow,
            and shrink the component.
        bg_thesh: `float`
            The fraction of the `bg_rms` used in each band determine
            L0 sparsity thresholding.
        floor: `float`
            Minimum value of the SED or center morpjology pixel.
        """
        # Initialize all of the base attributes
        super().__init__(center, bbox, sed, morph, initialized=True, bg_thresh=bg_thresh, bg_rms=bg_rms)
        # Initialize the monotonicity constraint
        self.monotonicity = MonotonicityConstraint(
            neighbor_weight="angle",
            min_gradient=0,
            fit_center_radius=fit_center_radius
        )
        self.floor = floor
        self.model_bbox = model_bbox

        # update the parameters
        self._sed.grad = self.grad_sed
        self._sed.prox = self.prox_sed
        self._morph.grad = self.grad_morph
        self._morph.prox = self.prox_morph
        self.slices = overlapped_slices(model_bbox, bbox)

    @property
    def sed(self):
        """The array of SED values"""
        return self._sed.x

    @property
    def morph(self):
        """The array of morphology values"""
        return self._morph.x

    def get_model(self, bbox=None):
        """Build the model from the SED and morphology"""
        model = self.sed[:, None, None] * self.morph[None, :, :]

        if bbox is not None:
            slices = overlapped_slices(bbox, self.bbox)
            _model = np.zeros(bbox.shape, self.morph.dtype)
            _model[slices[0]] = model[slices[1]]
            model = _model
        return model

    def grad_sed(self, input_grad, sed, morph):
        """Gradient of the SED wrt. the component model"""
        _grad = np.zeros(self.bbox.shape, dtype=self.morph.dtype)
        _grad[self.slices[1]] = input_grad[self.slices[0]]
        return np.einsum("...jk,jk", _grad, morph)

    def grad_morph(self, input_grad, morph, sed):
        """Gradient of the morph wrt. the component model"""
        _grad = np.zeros(self.bbox.shape, dtype=self.morph.dtype)
        _grad[self.slices[1]] = input_grad[self.slices[0]]
        return np.einsum("i,i...", sed, _grad)

    def prox_sed(self, sed, prox_step=0):
        """Apply a prox-like update to the SED"""
        # prevent divergent SED
        sed[sed < self.floor] = self.floor
        return sed

    def prox_morph(self, morph, prox_step=0):
        """Apply a prox-like update to the morphology"""
        # monotonicity
        morph = self.monotonicity(morph, 0)

        if self.bg_thresh is not None:
            bg_thresh = self.bg_rms * self.bg_thresh
            # Enforce background thresholding
            model = self.sed[:, None, None] * morph[None, :, :]
            morph[np.all(model < bg_thresh[:, None, None], axis=0)] = 0
        else:
            # enforce positivity
            morph[morph < 0] = 0

        # prevent divergent morphology
        shape = morph.shape
        center = (shape[0] // 2, shape[1] // 2)
        morph[center] = np.max([morph[center], self.floor])
        # Normalize the morphology
        morph[:] = morph / morph.max()
        return morph

    def update(self, it, input_grad):
        """Update the SED and morphology parameters"""
        # Store the input SED so that the morphology can
        # have a consistent update
        sed = self.sed.copy()
        self._sed.update(it, input_grad, self.morph)
        self._morph.update(it, input_grad, sed)

    def __str__(self):
        return "LiteFactorizedComponent"

    def __repr__(self):
        return "LiteFactorizedComponent"


class LiteSource:
    """A container for components associated with the same astrophysical object

    A source can have a single component, or multiple components, and each can be
    contained in different bounding boxes.
    """
    def __init__(self, components, dtype):
        self.components = components
        self.dtype = dtype
        self._flux = None
        self.flux_bbox = None

    @property
    def n_components(self):
        """The number of components in this source"""
        return len(self.components)

    @property
    def center(self):
        if not self.is_null:
            return self.components[0].center
        return None

    @property
    def is_null(self):
        """True if the source does not have any components"""
        return self.n_components == 0

    @property
    def bbox(self):
        """The minimal bounding box to contain all of this sources components

        Null sources have a bounding box with shape `(0,0,0)`
        """
        if self.n_components == 0:
            return Box((0,0,0))
        bbox = self.components[0].bbox
        for component in self.components[1:]:
            bbox = bbox | component.bbox
        return bbox

    def get_model(self, bbox=None, use_flux=False):
        """Build the model for the source

        This is never called during optimization and is only used
        to generate a model of the source for investigative purposes.
        """
        if self.n_components == 0:
            return 0

        if use_flux:
            # Return the redistributed flux
            # (calculated by scarlet.lite.measure.weight_sources)
            if bbox is None:
                return self.flux
            return insert_image(bbox, self.flux_box, self.flux)

        if bbox is None:
            bbox = self.bbox
        model = np.zeros(bbox.shape, dtype=self.dtype)
        for component in self.components:
            slices = overlapped_slices(bbox, component.bbox)
            model[slices[0]] += component.get_model()[slices[1]]
        return model

    def __str__(self):
        return f"LiteSource<{','.join([str(c) for c in self.components])}>"

    def __repr__(self):
        return f"LiteSource<{len(self.components)}>"


class LiteObservation:
    """A single observation

    This is effectively a combination of the `Observation` and
    `Renderer` class from base scarlet, greatly simplified due
    to the assumptions that the observations are all resampled
    onto the same pixel grid and that the `images` contain all
    of the information for all of the model bands.
    """
    def __init__(self, images, variance, weights, psfs, model_psf=None, noise_rms=None,
                 bbox=None, padding=3, convolution_mode="fft"):
        self.images = images
        self.variance = variance
        self.weights = weights
        # make sure that the images and psfs have the same dtype
        if psfs.dtype != images.dtype:
            psfs = psfs.astype(images.dtype)
        self.psfs = psfs

        assert convolution_mode in ["fft", "real"], "convolution_mode must be either 'fft' or 'real'"
        self.mode = convolution_mode
        if noise_rms is None:
            noise_rms = np.array(np.mean(np.sqrt(variance), axis=(1,2)))
        self.noise_rms = noise_rms

        # Create a difference kernel to convolve the model to the PSF
        # in each band
        self.model_psf = model_psf
        self.padding = padding
        if model_psf is not None:
            self.diff_kernel = fft.match_psf(psfs, model_psf, padding=padding)
            # The gradient of a convolution is another convolution,
            # but with the flipped and transposed kernel.
            diff_img = self.diff_kernel.image
            self.grad_kernel = fft.Fourier(diff_img[:, ::-1, ::-1])
        else:
            self.diff_kernel = self.grad_kernel = None

        if bbox is None:
            self.bbox = Box(images.shape)
        else:
            self.bbox = bbox

    def convolve(self, image, mode=None, grad=False):
        """Convolve the model into the observed seeing in each band.

        Parameters
        ----------
        image: `~numpy.array`
            The image to convolve
        mode: `str`
            The convolution mode to use.
            This should be "real" or "fft" or `None`,
            where `None` will use the default `convolution_mode`
            specified during init.
        grad: `bool`
            Whether this is a backward gradient convolution
            (`grad==True`) or a pure convolution with the PSF.
        """
        if grad:
            kernel = self.grad_kernel
        else:
            kernel = self.diff_kernel

        if kernel is None:
            return image

        if mode is None:
            mode = self.mode
        if mode == "fft":
            result = fft.convolve(
                fft.Fourier(image), kernel, axes=(1, 2),
            ).image
        elif mode == "real":
            result = convolve(image, kernel.image, self.convolution_bounds)
        else:
            raise ValueError(f"mode must be either 'fft' or 'real', got {mode}")
        return result

    def render(self, model):
        """Mirror of `Observation.render to make APIs match
        """
        return self.convolve(model)

    @property
    def data(self):
        """Mirror of `Observation.data` to make APIs match
        """
        return self.images

    @property
    def shape(self):
        """The shape of the iamges, variance, etc."""
        return self.images.shape

    @property
    def n_bands(self):
        """The number of bands in the observation"""
        return self.images.shape[0]

    @property
    def dtype(self):
        """The dtype of the observation is the dtype of the images
        """
        return self.images.dtype

    @property
    def convolution_bounds(self):
        """Build the slices needed for convolution in real space
        """
        if not hasattr(self, "_convolution_bounds"):
            coords = interpolation.get_filter_coords(self.diff_kernel[0])
            self._convolution_bounds = interpolation.get_filter_bounds(
                coords.reshape(-1, 2)
            )
        return self._convolution_bounds

    def __getitem__(self, i):
        """Allow the user to slice the observations with python indexing
        """
        images = self.images[i]
        variance = self.variance[i]
        weights = self.weights[i]
        psfs = self.psfs[i]
        noise_rms = self.noise_rms[i]

        if len(images.shape) == 2:
            images = images[None]
            variance = variance[None]
            weights = weights[None]
            psfs = psfs[None]
            noise_rms = np.array([noise_rms])

        return LiteObservation(
            images,
            variance,
            weights,
            psfs,
            model_psf=self.model_psf,
            noise_rms=noise_rms,
            bbox=self.bbox,
            padding=self.padding,
            convolution_mode=self.mode,
        )


class LiteBlend:
    """A single blend.

    This is effectively a combination of the `Blend`, `Observation`, and `Renderer`
    classes in main scarlet, greatly simplified due to the assumptions that the
    observations are all resampled onto the same pixel grid and that the
    `images` contain all of the information for all of the model bands.

    This is still agnostic to the component type, so new custom classes
    are allowed as long as they posses the `get_model`, `update`, and
    `resize` methods, but all components should be contained in sources.
    The only underlying assumption is that all of the components inserted
    into the model by addition. If the components require a more
    complicated insertion, for example multiplication of a dust lane,
    then a new blend class will need to be created.
    """
    def __init__(self, sources, observation):
        """Initialize the class.

        Parameters
        ----------
        sources: `list` of `scarlet.lite.LiteSource`
            The sources to fit.
        `observation`: `scarlet.lite.LiteObservation`
            The observation that contains the images,
            PSF, etc. that are being fit.
        """
        self.sources = sources
        self.components = []
        for source in sources:
            self.components.extend(source.components)
        self.observation = observation

        # Initialzie the iteration count and loss function
        self.it = 0
        self.loss = []

    @property
    def bbox(self):
        """The bounding box of the entire blend"""
        return self.observation.bbox

    def get_model(self, convolve=False, use_flux=False):
        """Generate a model of the entire blend"""
        model = np.zeros(self.bbox.shape, dtype=self.observation.images.dtype)

        if use_flux:
            for src in self.sources:
                slices = overlapped_slices(self.bbox, src.flux_box)
                model[slices[0]] += src.flux
        else:
            for component in self.components:
                _model = component.get_model()
                model[component.slices[0]] += _model[component.slices[1]]
            if convolve:
                return self.observation.convolve(model)
        return model

    def grad_logL(self):
        """Gradient of the likelihood wrt the unconvolved model"""
        model = self.get_model(convolve=True)
        # Update the loss
        self.loss.append(0.5 * -np.sum(self.observation.weights * (self.observation.images - model)**2))
        # Calculate the gradient wrt the model d(logL)/d(model)
        grad_logL = self.observation.weights * (model - self.observation.images)
        grad_logL = self.observation.convolve(grad_logL, grad=True)
        return grad_logL

    def fit_spectra(self, clip=False):
        """Fit all of the spectra given their current morphologies

        Parameters
        ----------
        clip: `bool`
            Whether or not to clip components that were not
            assigned any flux during the fit.
        """
        from .initialization import multifit_seds

        morphs = [c.morph for c in self.components]
        boxes = [c.bbox[1:] for c in self.components]
        fit_seds = multifit_seds(self.observation, morphs, boxes)
        for idx, component in enumerate(self.components):
            component.sed[:] = fit_seds[idx]
            component.sed[component.sed < 0] = 0

        if clip:
            components = []
            # Remove components with no sed or morphology
            for src in self.sources:
                _components = []
                for c in src.components:
                    if np.any(c.sed) > 0 and np.any(c.morph) > 0:
                        components.append(c)
                        _components.append(c)
                src.components = _components
            self.components = components
        else:
            for src in self.sources:
                for c in src.components:
                    c.prox_sed(c.sed)

        return self

    @property
    def log_likelihood(self, model=None):
        if model is None:
            return np.array(self.loss)
        return 0.5 * -np.sum(self.observation.weights * (self.observation.images - model)**2)

    def fit(self, max_iter, e_rel=1e-4, min_iter=1, resize=10, reweight=True):
        """Fit all of the parameters

        Parameters
        ----------
        max_iter: `int`
            The maximum number of iterations
        e_rel: `float`
            The relative error to use for determining convergence.
        min_iter: `int`
            The minimum number of iterations.
        resize: `int`
            Number of iterations before attempting to resize the
            resizable components. If `resize` is `None` then
            no resizing is ever attempted.
        """
        it = self.it
        while it < max_iter:
            # Calculate the gradient wrt the on-convolved model
            grad_logL = self.grad_logL()
            # Update each component given the current gradient
            for component in self.components:
                component.update(it, grad_logL)
            # Check to see if any components need to be resized
            if resize is not None and it > 0 and it % resize == 0:
                for component in self.components:
                    if hasattr(component, "resize"):
                        component.resize()
            # Stopping criteria
            if it > min_iter and np.abs(self.loss[-1] - self.loss[-2]) < e_rel * np.abs(self.loss[-1]):
                break
            it += 1
        self.it = it
        if reweight:
            weight_sources(self)
        return it, self.loss[-1]

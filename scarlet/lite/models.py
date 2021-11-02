# This module contains classes and methods to use the so-called "lite" version of scarlet.
# In this instance "lite" just means scarlet stripped down to the bare essentials needed
# to deblend observations from a single instrument, where all of the observations have
# been resampled to the same pixel grid (but may have differing PSFs).
# Because some of the assumptions and constraints are different, this module is designed
# to run faster and use less memory than the standard scarlet modules while obtaining
# a similar log likelihood.
from abc import ABC, abstractmethod

import numpy as np
import proxmin

from .bbox import overlapped_slices, Box
from .renderer import convolve
from . import fft, interpolation, initialization
from .constraint import  MonotonicityConstraint


def grow_array(x, newshape, dist):
    """grow an array and pad it with zeros

    This is faster than `numpy.pad` by a factor of ~20.

    Parameters
    ----------
     x: `~numpy.array`
        The array to grow
    newshape: `tuple` of `int`
        This is the new shape of the array.
        It would be trivial to calculate in this function,
        however in most cases this is already calculated for
        other purposes, so it might as well be reused.
    dist: int
        The amount to pad each side of the input array
        (so the new shape is extended by 2*dist on each axis).

    Returns
    -------
    result: `~numpy.array`
        The larger array that contains `x`.
    """
    result = np.zeros(newshape, dtype=x.dtype)
    result[dist:-dist, dist:-dist] = x
    return result


class LiteParameter(ABC):
    """A parameter in a `LiteComponent`

    Unlike the main scarlet `Parameter` class,
    a `LiteParameter` also contains methods to update the parameter,
    using any given optimizer, provided the abstract methods
    are all implemented. The main parameter should always be
    stored as `LiteParameter.x`, but the names of the meta parameters
    can be different.
    """
    @abstractmethod
    def update(self, it, input_grad, *args):
        """Update the parameter in one iteration.

        This includes the gradient update, proximal update,
        and any meta parameters that are stored as class
        attributes to update the parameter.

        Parameters
        ----------
        it: int
            The current iteration
        input_grad: `~numpy.array`
            The gradient from the full model, passed to the parameter.
        """
        pass

    @abstractmethod
    def grow(self, newshape, dist):
        """Grow the parameter and all of the meta parameters

        Parameters
        ----------
        newshape: `tuple` of `int`
            The new shape of the parameter.
        dist: `int`
            The amount to extend the array in each direction
        """
        pass

    @abstractmethod
    def shrink(self, dist):
        """Shrink the parameter and all of the meta parameters

        Parameters
        ----------
        dist: `int`
            The amount to shrink the array in each direction
        """
        pass


class BeckTeboulleParameter(LiteParameter):
    """A `LiteParameter` that updates itself using the Beck-Teboulle
    proximal gradient method.
    """
    def __init__(self, x, step, grad=None, prox=None, t0=1, z0=None):
        """Initialize the parameter

        Parameters
        ----------
        x: `~numpy.array`
            The initial guess for the parameter.
        step: `float`
            The step size for the parameter.
        grad: `func`
            The function to use to calculate the gradient.
            `grad` should accept the `input_grad` and a list
            of arguments.
        prox: `func`
            The function that acts as a proximal operator.
            This function should take `x` as an input, however
            the input `x` might not be the same as the input
            parameter, but a meta parameter instead.
        t0: `float`
            The initial value of the acceleration parameter.
        z0: `~numpy.array`
            The initial value of the meta parameter `z`.
            If this is `None` then `z` is initialized to the
            initial `x`.
        """
        if z0 is None:
            z0 = x
        self.x = x
        self.step = step
        self.grad = grad
        self.prox = prox
        self.z = z0
        self.t = t0

    def update(self, it, input_grad, *args):
        """Update the parameter and meta-parameters using the PGM

        See `~LiteParameter` for more.
        """
        y = self.z - self.step*self.grad(input_grad)
        x = self.prox(y, self.step)
        t = 0.5*(1 + np.sqrt(1 + self.t**2))
        omega = 1 + (self.t -1)/t
        self.z = self.x + omega*(x-self.x)
        self.x = x
        self.t = t

    def grow(self, newshape, dist):
        self.x = grow_array(self.x, newshape, dist)
        self.z = grow_array(self.z, newshape, dist)

    def shrink(self, dist):
        self.x = self.x[dist:-dist, dist:-dist]
        self.z = self.z[dist:-dist, dist:-dist]


def _amsgrad_phi_psi(it, G, M, V, Vhat, b1, b2, eps, p):
    # moving averages
    M[:] = (1 - b1) * G + b1 * M
    V[:] = (1 - b2) * (G ** 2) + b2 * V

    Phi = M
    Vhat[:] = np.maximum(Vhat, V)
    # sanitize zero-gradient elements
    if eps > 0:
        Vhat = np.maximum(Vhat, eps)
    Psi = np.sqrt(Vhat)
    return Phi, Psi


phi_psi = {
    "adam": proxmin.algorithms._adam_phi_psi,
    "nadam": proxmin.algorithms._nadam_phi_psi,
    "amsgrad": _amsgrad_phi_psi,
    "padam": proxmin.algorithms._padam_phi_psi,
    "adamx": proxmin.algorithms._adamx_phi_psi,
    "radam": proxmin.algorithms._radam_phi_psi,
}


class AdaproxParameter(LiteParameter):
    """Operator updated using te Proximal ADAM algorithm

    Uses multiple variants of adaptive quasi-Newton gradient descent
        * Adam (Kingma & Ba 2015)
        * NAdam (Dozat 2016)
        * AMSGrad (Reddi, Kale & Kumar 2018)
        * PAdam (Chen & Gu 2018)
        * AdamX (Phuong & Phong 2019)
        * RAdam (Liu et al. 2019)
    and PGM sub-iterations to satisfy feasibility and optimality. See details of the
    algorithms in the respective papers.

    TODO: implement other schemes by making `b1` use a list instead of a single value.
    """
    def __init__(self, x, step, grad=None, prox=None, b1=0.9, b2=0.999, eps=1e-8, p=0.25,
                 m0=None, v0=None, vhat0=None, scheme="amsgrad", max_prox_iter=10, prox_e_rel=1e-6):
        """Initialize the parameter

         NOTE:
        Setting `m`, `v`, `vhat` allows to continue a previous run, e.g. for a warm start
        of a slightly changed problem. If not set, they will be initialized with 0.

        Parameter
        ---------
        x: `~numpy.array`
            The initial guess for the parameter.
        step: `func`
            The step size for the parameter that takes the
            parameter `x` and the iteration `it` as arguments.
        grad: `func`
            The function to use to calculate the gradient.
            `grad` should accept the `input_grad` and a list
            of arguments.
        prox: `func`
            The function that acts as a proximal operator.
            This function should take `x` as an input, however
            the input `x` might not be the same as the input
            parameter, but a meta parameter instead.
        b1: `float`
            The strength parameter for the weighted gradient
            (`m`) update.
        b2: `float`
            The strength for the weighted gradient squared
            (`v`) update.
        eps: `float`
            Minimum value of the cumulative gradient squared
            (`vhat`) meta paremeter.
        p: `float`
            Meta parameter used by some of the ADAM schemes
        m0: `~numpy.array`
            Initial value of the weighted gradient (`m`) parameter
            for a warm start.
        v0: `~numpy.array`
            Initial value of the weighted gradient squared(`v`) parameter
            for a warm start.
        vhat0: `~numpy.array`
            Initial value of the
            cumulative weighted gradient squared (`vhat`) parameter
            for a warm start.
        scheme: str
            Name of the ADAM scheme to use.
            One of ["adam", "nadam", "adamx", "amsgrad", "padam", "radam"]
        """
        self.x = x
        self.b1 = b1
        self.b2 = b2
        self.eps = eps
        self.p = p

        if not hasattr(step, "__call__"):
            def _step(x, it):
                return step
            self.step = _step
        else:
            self.step = step
        self.grad = grad
        self.prox = prox
        if m0 is None:
            m0 = np.zeros(x.shape, dtype=x.dtype)
        self.m = m0
        if v0 is None:
            v0 = np.zeros(x.shape, dtype=x.dtype)
        self.v = v0
        if vhat0 is None:
            vhat0 = np.ones(x.shape, dtype=x.dtype) * -np.inf
        self.vhat = vhat0
        self.phi_psi = phi_psi[scheme]
        self.max_prox_iter = max_prox_iter
        self.e_rel = prox_e_rel

    def update(self, it, input_grad, *args):
        """Update the parameter and meta-parameters using the PGM

        See `~LiteParameter` for more.
        """
        # Calculate the gradient
        grad = self.grad(input_grad, self.x, *args)
        # Get the update for the parameter
        phi, psi = self.phi_psi(
            it, grad, self.m, self.v, self.vhat,
            self.b1, self.b2, self.eps, self.p
        )
        # Calculate the step size
        step = self.step(self.x, it)
        if it > 0:
            self.x -= step * phi/psi
        else:
            self.x -= step * phi / psi/10

        # Iterate over the proximal operators until convergence
        if self.prox is not None:
            z = self.x.copy()
            gamma = step/np.max(psi)
            for tau in range(1, self.max_prox_iter + 1):
                _z = self.prox(z - gamma/step * psi * (z-self.x), gamma)
                converged = proxmin.utils.l2sq(_z-z) <= self.e_rel**2 * proxmin.utils.l2sq(z)
                z = _z

                if converged:
                    break

            self.x = z

    def grow(self, newshape, dist):
        self.x = grow_array(self.x, newshape, dist)
        self.m = grow_array(self.m, newshape, dist)
        self.v = grow_array(self.v, newshape, dist)
        self.vhat = grow_array(self.vhat, newshape, dist)

    def shrink(self, dist):
        self.x = self.x[dist:-dist, dist:-dist]
        self.m = self.m[dist:-dist, dist:-dist]
        self.v = self.v[dist:-dist, dist:-dist]
        self.vhat = self.vhat[dist:-dist, dist:-dist]


class LiteFactorizedComponent:
    """Implementation of a `FactorizedComponent` for simplified observations.
    """
    def __init__(self, sed, morph, center, bbox, model_bbox, bg_rms, bg_thresh=0.5, floor=1e-20):
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
        self._sed = sed
        self._morph = morph
        self.center = center
        self.center_index = tuple([
            int(np.round(center[0]-bbox.origin[-2])),
            int(np.round(center[1]-bbox.origin[-1]))
        ])
        self.bbox = bbox
        # Initialize the monotonicity constraint
        self.monotonicity = MonotonicityConstraint(neighbor_weight="angle", min_gradient=0)
        self.floor = floor
        self.bg_rms = bg_rms
        self.bg_thresh = bg_thresh
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

    def resize(self):
        """Test whether or not the component needs to be resized
        TODO: this code is similar enough to `ImageMorphology.resize` that it should be abstracted as a function
        """
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

        newsize = initialization.get_minimal_boxsize(size - 2 * dist)
        if newsize < size:
            dist = (size - newsize) // 2
            self.bbox.origin = (self.bbox.origin[0], self.bbox.origin[1]+dist, self.bbox.origin[2]+dist)
            self.bbox.shape = (self.bbox.shape[0], newsize, newsize)
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

        if self.bg_thresh is not None and np.any(edge_flux > self.bg_thresh*self.bg_rms[:, None, None]):
            newsize = initialization.get_minimal_boxsize(size + 1)
            dist = (newsize - size) // 2
            self.bbox.origin = (self.bbox.origin[0], self.bbox.origin[1]-dist, self.bbox.origin[2]-dist)
            self.bbox.shape = (self.bbox.shape[0], newsize, newsize)
            self._morph.grow(self.bbox.shape[1:], dist)
            self.slices = overlapped_slices(self.model_bbox, self.bbox)
            return True
        return False

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
    def __init__(self, components):
        self.components = components
        self.nbr_components = len(components)
        self.dtype = self.components[0].get_model().dtype

    @property
    def bbox(self):
        """The minimal bounding box to contain all of this sources components"""
        bbox = self.components[0].bbox
        for component in self.components[1:]:
            bbox = bbox | component.bbox
        return bbox

    def get_model(self, bbox=None):
        """Build the model for the source

        This is never called during optimization and is only used
        to generate a model of the source for investigative purposes.
        """
        if bbox is None:
            bbox = self.bbox
        model = np.zeros(self.bbox.shape, dtype=self.dtype)
        for component in self.components:
            slices = overlapped_slices(bbox, component.bbox)
            model[slices[0]] = component.get_model()[slices[1]]
        return model

    def __str__(self):
        return f"LiteSource<{','.join([str(c) for c in self.components])}>"

    def __repr__(self):
        return f"LiteSource<{len(self.components)}>"


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
    def __init__(self, sources, images, weights, psfs, model_psf=None,
                 bbox=None, padding=3, convolution_mode="fft"):
        self.sources = sources
        self.components = []
        for source in sources:
            self.components.extend(source.components)
        self.images = images
        self.weights = weights
        self.psfs = psfs
        self.mode = convolution_mode

        # Create a difference kernel to convolve the model to the PSF
        # in each band
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

        # Initialzie the iteration count and loss function
        self.it = 0
        self.loss = []

    def get_model(self):
        """Generate a model of the entire blend"""
        model = np.zeros(self.bbox.shape, dtype=self.images.dtype)
        for component in self.components:
            _model = component.get_model()
            model[component.slices[0]] += _model[component.slices[1]]
        return model

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

        if mode is None:
            mode = self.mode
        if mode == "fft":
            result = fft.convolve(
                fft.Fourier(image), kernel, axes=(1, 2),
            ).image
        elif mode == "real":
            result = convolve(image, kernel.image, self.convolution_bounds)
        else:
            raise ValueError(f"mode must be either 'fft' or 'rea', got {mode}")
        return result

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

    def grad_logL(self):
        """Gradient of the likelihood wrt the unconvolved model"""
        model = self.convolve(self.get_model())
        # Update the loss
        self.loss.append(0.5 * -np.sum(self.weights * (self.images - model)**2))
        # Calculate the gradient wrt the model d(logL)/d(model)
        grad_logL = self.weights * (model - self.images)
        grad_logL = self.convolve(grad_logL, grad=True)
        return grad_logL

    def fit(self, max_iter, e_rel=1e-3, min_iter=1, resize=10):
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
        return it, self.loss[-1]
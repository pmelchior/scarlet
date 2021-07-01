from abc import ABC, abstractmethod

import numpy as np
import proxmin

from .bbox import overlapped_slices, Box
from .renderer import convolve
from . import fft, interpolation, initialization
from .constraint import  MonotonicityConstraint


def grow_array(x, newshape, dist):
    result = np.zeros(newshape, dtype=x.dtype)
    result[dist:-dist, dist:-dist] = x
    return result


class LiteParameter(ABC):
    @abstractmethod
    def update(self, it, input_grad):
        pass

    @abstractmethod
    def grow(self, newshape, dist):
        pass

    @abstractmethod
    def shrink(self, dist):
        pass


class BeckTeboulleParameter(LiteParameter):
    def __init__(self, x, step, grad=None, prox=None, t0=1, z0=None):
        if z0 is None:
            z0 = x
        self.x = x
        self.step = step
        self.grad = grad
        self.prox = prox
        self.z = z0
        self.t = t0

    def update(self, it, input_grad):
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
    def __init__(self, x, step, grad=None, prox=None, b1=0.9, b2=0.999, eps=1e-8, p=0.25,
                 m0=None, v0=None, vhat0=None, scheme="amsgrad", max_prox_iter=10, prox_e_rel=1e-6):
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
        grad = self.grad(input_grad, self.x, *args)
        # todo: remove
        self.G = grad
        phi, psi = self.phi_psi(
            it, grad, self.m, self.v, self.vhat,
            self.b1, self.b2, self.eps, self.p
        )
        step = self.step(self.x, it)
        # TODO: remove
        #self.updates = step * phi/psi
        if it > 0:
            self.x -= step * phi/psi
        else:
            self.x -= step * phi / psi/10
        self.updates = self.x.copy()

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


class LiteComponent:
    def __init__(self, sed, morph, center, bbox, model_bbox, bg_rms, bg_thresh=0.5, floor=1e-20):
        self._sed = sed
        self._morph = morph
        self.center = center
        self.center_index = tuple([
            int(np.round(center[0]-bbox.origin[-2])),
            int(np.round(center[1]-bbox.origin[-1]))
        ])
        self.bbox = bbox
        self.monotonicity = MonotonicityConstraint(neighbor_weight="angle", min_gradient=0)
        self.floor = floor
        self.bg_thresh = bg_rms * bg_thresh
        self.model_bbox = model_bbox

        # update the parameters
        self._sed.grad = self.grad_sed
        self._sed.prox = self.prox_sed
        self._morph.grad = self.grad_morph
        self._morph.prox = self.prox_morph
        self.slices = overlapped_slices(model_bbox, bbox)

    @property
    def sed(self):
        return self._sed.x

    @property
    def morph(self):
        return self._morph.x

    def get_model(self, bbox=None):
        model = self.sed[:, None, None] * self.morph[None, :, :]

        if bbox is not None:
            slices = overlapped_slices(bbox, self.bbox)
            _model = np.zeros(bbox.shape, self.morph.dtype)
            _model[slices[0]] = model[slices[1]]
            model = _model
        return model

    def grad_sed(self, input_grad, sed, morph):
        _grad = np.zeros(self.bbox.shape, dtype=self.morph.dtype)
        _grad[self.slices[1]] = input_grad[self.slices[0]]
        return np.einsum("...jk,jk", _grad, morph)

    def grad_morph(self, input_grad, morph, sed):
        _grad = np.zeros(self.bbox.shape, dtype=self.morph.dtype)
        _grad[self.slices[1]] = input_grad[self.slices[0]]
        return np.einsum("i,i...", sed, _grad)

    def prox_sed(self, sed, prox_step=0):
        # prevent divergent SED
        sed[sed < self.floor] = self.floor
        return sed

    def prox_morph(self, morph, prox_step=0):
        # monotonicity
        morph = self.monotonicity(morph, 0)

        if self.bg_thresh is not None:
            # Enforce background thresholding
            model = self.sed[:, None, None] * morph[None, :, :]
            morph[np.all(model < self.bg_thresh[:, None, None], axis=0)] = 0
        else:
            # enforce positivity
            morph[morph < 0] = 0

        # prevent divergent morphology
        shape = morph.shape
        center = (shape[0] // 2, shape[1] // 2)
        morph[center] = np.max([morph[center], self.floor])
        # Normalize the morphology
        # TODO: test removing this prox
        morph[:] = morph / morph.max()
        return morph

    def resize(self):
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

        if self.bg_thresh is not None and np.any(edge_flux > self.bg_thresh[:, None, None]):
            newsize = initialization.get_minimal_boxsize(size + 1)
            dist = (newsize - size) // 2
            self.bbox.origin = (self.bbox.origin[0], self.bbox.origin[1]-dist, self.bbox.origin[2]-dist)
            self.bbox.shape = (self.bbox.shape[0], newsize, newsize)
            self._morph.grow(self.bbox.shape[1:], dist)
            self.slices = overlapped_slices(self.model_bbox, self.bbox)
            return True
        return False

    def update(self, it, input_grad):
        sed = self.sed.copy()
        self._sed.update(it, input_grad, self.morph)
        self._morph.update(it, input_grad, sed)

    def __str__(self):
        return "LiteComponent"

    def __repr__(self):
        return "LiteComponent"


class LiteSource:
    def __init__(self, components):
        self.components = components
        self.nbr_components = len(components)
        self._bbox = components[0].bbox
        for component in self.components:
            assert component.bbox == self.bbox
        self.dtype = self.components[0].get_model().dtype

    @property
    def bbox(self):
        return self._bbox

    def get_model(self, bbox=None):
        model = np.zeros(self.bbox.shape, dtype=self.dtype)
        for component in self.components:
            model += component.get_model()
        if bbox is not None and bbox != self.bbox:
            slices = overlapped_slices(bbox, self.bbox)
            _model = np.zeros(bbox.shape, dtype=self.dtype)
            _model[slices[0]] = model[slices[1]]
            model = _model
        return model

    def __str__(self):
        return f"LiteSource<{','.join([str(c) for c in self.components])}>"

    def __repr__(self):
        return f"LiteSource<{len(self.components)}>"


class LiteBlend:
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

        if model_psf is not None:
            self.diff_kernel = fft.match_psf(psfs, model_psf, padding=padding)
            diff_img = self.diff_kernel.image
            self.grad_kernel = fft.Fourier(diff_img[:, ::-1, ::-1])
        else:
            self.diff_kernel = self.grad_kernel = None

        if bbox is None:
            self.bbox = Box(images.shape)
        else:
            self.bbox = bbox

        self.it = 0
        self.loss = []

    def get_model(self):
        model = np.zeros(self.bbox.shape, dtype=self.images.dtype)
        for component in self.components:
            _model = component.get_model()
            model[component.slices[0]] += _model[component.slices[1]]
        return model

    def convolve(self, image, mode=None, grad=False):
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
        model = self.convolve(self.get_model())
        # Update the loss
        self.loss.append(0.5 * -np.sum(self.weights * (self.images - model)**2))
        # Calculate the gradient wrt the model d(logL)/d(model)
        grad_logL = self.weights * (model - self.images)
        grad_logL = self.convolve(grad_logL, grad=True)
        return grad_logL

    def fit(self, max_iter, e_rel=1e-3, min_iter=1, resize=10):
        it = self.it
        while it < max_iter:
            grad_logL = self.grad_logL()
            for component in self.components:
                component.update(it, grad_logL)

            if resize is not None and it > 0 and it % resize == 0:
                for component in self.components:
                    if hasattr(component, "resize"):
                        component.resize()

            if it > min_iter and np.abs(self.loss[-1] - self.loss[-2]) < e_rel * np.abs(self.loss[-1]):
                break
            it += 1
        self.it = it
        return it, self.loss[-1]
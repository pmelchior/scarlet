from __future__ import print_function, division
import numpy as np
from functools import partial

import proxmin
from . import transformations
from . import operators

import logging
logger = logging.getLogger("scarlet")

class Source(object):
    def __init__(self, center, size, K=1, B=1, psf=None, constraints=None, fix_sed=False, fix_morph=False, fix_frame=False, shift_center=0.2):

        # set size of the source frame
        self._set_frame(center, size)
        size = (self.Ny, self.Nx)
        self.fix_frame = fix_frame

        # create containers
        self.K = K
        self.B = B
        self.sed = np.zeros((self.K, self.B))
        self.morph = np.zeros((self.K, self.Ny*self.Nx))

        # set up psf and translations matrices
        self.psf = psf
        self._gammaOp = transformations.GammaOp(size, psf=self.psf)

        # set center coordinates and translation operators
        # needs to have GammaOp set up first
        self.set_center(center)
        self.shift_center = shift_center

        # updates for sed or morph?
        if hasattr(fix_sed, '__iter__') and len(fix_sed) == self.K:
            self.fix_sed = fix_sed
        else:
            self.fix_sed = [fix_sed] * self.K
        if hasattr(fix_morph, '__iter__') and len(fix_morph) == self.K:
            self.fix_morph = fix_morph
        else:
            self.fix_morph = [fix_morph] * self.K

        # set sed and morph constraints
        self.set_constraints(constraints)

    @property
    def Nx(self):
        return self.right-self.left

    @property
    def Ny(self):
        return self.top - self.bottom

    @property
    def shape(self):
        return (self.B, self.Ny, self.Nx)

    @property
    def image(self):
        morph_shape = (self.K, self.Ny, self.Nx)
        return self.morph.reshape(morph_shape) # this *should* be a view

    @property
    def center_int(self):
        return np.array(self.center, dtype='int')

    def get_slice_for(self, im_shape):
        # slice so that self.image[k][slice] corresponds to image[self.bb]
        NY, NX = im_shape[1:]

        left = max(0, -self.left)
        bottom = max(0, -self.bottom)
        right = self.Nx - max(0, self.right - NX)
        top = self.Ny - max(0, self.top - NY)
        return (slice(None), slice(bottom, top), slice(left, right))

    def get_model(self, combine=True, Gamma=None):
        if Gamma is None:
            Gamma = self.Gamma
        # model for all components of this source
        if hasattr(Gamma, 'shape'): # single matrix: one for all bands
            model = np.empty((self.K, self.B, self.Ny*self.Nx))
            for k in range(self.K):
                model[k] = np.outer(self.sed[k], Gamma.dot(self.morph[k]))
        else:
            model = np.zeros((self.K, self.B, self.Ny*self.Nx))
            for k in range(self.K):
                for b in range(self.B):
                    model[k,b] += self.sed[k,b] * Gamma[b].dot(self.morph[k])

        # reshape the image into a 2D array
        model = model.reshape(self.K, self.B, self.Ny, self.Nx)

        if combine:
            model = model.sum(axis=0)
        return model

    def _set_frame(self, center, size):
        assert len(center) == 2
        self.center = np.array(center)

        if hasattr(size, '__iter__'):
            size = size[:2]
        else:
            size = (size,) * 2

        # make cutout of in units of the original image frame (that defines xy)
        # ensure odd pixel number
        y_, x_ = self.center_int
        self.bottom, self.top = y_ - int(size[0]//2), y_ + int(size[0]//2) + 1
        self.left, self.right = x_ - int(size[1]//2), x_ + int(size[1]//2) + 1

        # since slice wrap around if start or stop are negative, need to sanitize
        # start values (stop always postive)
        self.bb = (slice(None), slice(max(0, self.bottom), self.top), slice(max(0, self.left), self.right))

    def set_center(self, center):
        size = (self.Ny, self.Nx)
        self._set_frame(center, size)

        # update translation operator
        dx = self.center - self.center_int
        self.Gamma = self._gammaOp(dx)

    def resize(self, size):
        # store old edge coordinates
        top, right, bottom, left = self.top, self.right, self.bottom, self.left
        self._set_frame(self.center, size)

        # check if new size is larger or smaller
        new_slice_y = slice(max(0, bottom - self.bottom), min(self.top - self.bottom, self.top - self.bottom - (self.top - top)))
        old_slice_y = slice(max(0, self.bottom - bottom), min(top - bottom, top - bottom - (top - self.top)))
        if top-bottom == self.Ny:
            new_slice_y = old_slice_y = slice(None)
        new_slice_x = slice(max(0, left - self.left), min(self.right - self.left, self.right - self.left - (self.right - right)))
        old_slice_x = slice(max(0, self.left - left), min(right - left, right - left - (right - self.right)))
        if right-left == self.Nx:
            new_slice_x = old_slice_x = slice(None)
        new_slice = (slice(None), new_slice_y, new_slice_x)
        old_slice = (slice(None), old_slice_y, old_slice_x)

        if new_slice != old_slice:
            # change morph
            _morph = self.morph.copy().reshape((self.K, top-bottom, right-left))
            self.morph = np.zeros((self.K, self.Ny, self.Nx))

            self.morph[new_slice] = _morph[old_slice]
            self.morph = self.morph.reshape((self.K, self.Ny*self.Nx))

            # update GammaOp and center (including subpixel shifts)
            size = (self.Ny, self.Nx)
            self._gammaOp = transformations.GammaOp(size, psf=self.psf)
            self.set_center(self.center)

            # set constraints
            self.set_constraints(self.constraints)

    def init_source(self, img, weights=None):
        # init with SED of the peak pixels
        # TODO: what should we do if peak is saturated?
        B = img.shape[0]
        self.sed = np.empty((1, B))
        y_, x_ = self.center_int
        self.sed[0] = img[:,y_,x_]
        # ensure proper normalization
        self.sed[0] = proxmin.operators.prox_unity_plus(self.sed[0], 0)

        # same for morph: just one pixel with an estimate of the total flux
        self.morph = np.zeros((1, self.Ny*self.Nx))
        tiny = 1e-10
        cx, cy = self.Nx // 2, self.Ny // 2
        self.morph[0, cy*self.Nx+cx] = img[:,y_,x_].sum(axis=0) + tiny

    def get_morph_error(self, weights):
        w = np.zeros(self.shape)
        w[self.get_slice_for(weights.shape)] = weights[self.bb]
        w = w.reshape(self.B, -1)
        # compute direct error propagation assuming only this source SED(s)
        # and the pixel covariances: Sigma_morph = diag((A^T Sigma^-1 A)^-1)
        # CAVEAT: If done on the entire A matrix, degeneracies in the linear
        # solution arise and substantially amplify the error estimate:
        # A = self.sed.T
        # return np.sqrt(np.diagonal(np.linalg.inv(np.dot(np.multiply(self.w.T[:,None,:], A.T), A)), axis1=1, axis2=2))
        # Instead, estimate noise for each component separately:
        # simple multiplication for diagonal pixel covariance matrix
        from .utils import invert_with_zeros
        return [invert_with_zeros(np.sqrt(np.dot(a.T, np.multiply(w, a[:,None])))) for a in self.sed]

    def get_sed_error(self, weights):
        w = np.zeros(self.shape)
        w[self.get_slice_for(weights.shape)] = weights[self.bb]
        w = w.reshape(self.B, -1)
        # See explanation in get_morph_error
        from .utils import invert_with_zeros
        return [invert_with_zeros(np.sqrt(np.dot(s,np.multiply(w.T, s[None,:].T)))) for s in self.morph]

    def set_morph_sparsity(self, weights):
        if "l0" in self.constraints.keys():
            morph_error = self.get_morph_error(weights)
            # filter out -1s for pixels outside of weight images
            morph_std = np.array([np.median(mek[mek != -1]) for mek in morph_error])
            # Note: don't use hard/soft thresholds with _plus (non-negative) because
            # that is either happening with prox_plus before in the
            # AlternatingProjections or is not indended
            morph_std *= self.constraints['l0']
            for k in range(self.K):
                pos = self.prox_morph[k].find(proxmin.operators.prox_hard)
                self.prox_morph[k].operators[pos] = partial(proxmin.operators.prox_hard, thresh=morph_std[k])
            return morph_std
        else:
            return np.zeros(self.K)

    def set_constraints(self, constraints):
        self.constraints = constraints # save for later
        if self.constraints is None:
            self.constraints = {}

        self.proxs_g = [None, []] # no constraints on A matrix
        self.Ls = [None, []]
        if self.constraints is None:
            self.proxs_g[1] = None
            self.Ls[1] = None
            return

        self.prox_sed = [proxmin.operators.prox_unity_plus] * self.K
        self.prox_morph = [[proxmin.operators.prox_plus],] * self.K

        shape = (self.Ny, self.Nx)
        for c in self.constraints.keys():

            # Note: don't use hard/soft thresholds with _plus (non-negative) because
            # that is either happening with prox_plus before or is not indended
            # Note: l0 thresh is not set yet, needs set_morph_sparsity()
            if c == "l0":
                if "l1" in self.constraints.keys():
                    # L0 has preference
                    logger.info("l1 penalty ignored in favor of l0 penalty")
                for k in range(self.K):
                    self.prox_morph[k].append(partial(proxmin.operators.prox_hard, thresh=0))
            elif c == "l1":
                thresh = self.constraints["l1"]
                for k in range(self.K):
                    self.prox_morph[k].append(partial(proxmin.operators.prox_soft, thresh=thresh))
            if c == "m":
                thresh = self.constraints["m"]
                for k in range(self.K):
                    self.prox_morph[k].append(operators.prox_strict_monotonic(shape, thresh=thresh))

            if c == "M":
                # positive gradients
                self.Ls[1].append(transformations.getRadialMonotonicOp(shape, useNearest=self.constraints[c]))
                self.proxs_g[1].append(proxmin.operators.prox_plus)
            elif c == "S":
                # zero deviation of mirrored pixels
                self.Ls[1].append(transformations.getSymmetryOp(shape))
                self.proxs_g[1].append(proxmin.operators.prox_zero)
            elif c == "c":
                useNearest = self.constraints.get("M", False)
                G = transformations.getRadialMonotonicOp(shape, useNearest=useNearest).toarray()
                self.proxs_g[1].append(partial(operators.prox_cone, G=G))
                self.Ls[1].append(None)
            elif (c == "X" or c == "x"): # X gradient
                cx = int(self.Nx)
                self.Ls[1].append(proxmin.transformations.get_gradient_x(shape, cx))
                if c == "X": # all positive
                    self.proxs_g[1].append(proxmin.operators.prox_plus)
                else: # l1 norm for TV_x
                    self.proxs_g[1].append(partial(proxmin.operators.prox_soft, thresh=self.constraints[c]))
            elif (c == "Y" or c == "y"): # Y gradient
                cy = int(self.Ny)
                self.Ls[1].append(proxmin.transformations.get_gradient_y(shape, cy))
                if c == "Y": # all positive
                    self.proxs_g[1].append(proxmin.operators.prox_plus)
                else: # l1 norm for TV_x
                    self.proxs_g[1].append(partial(proxmin.operators.prox_soft, thresh=self.constraints[c]))

        for k in range(self.K):
            if len(self.prox_morph[k]) > 1:
                self.prox_morph[k] = proxmin.operators.AlternatingProjections(self.prox_morph[k], repeat=1)
            else:
                self.prox_morph[k] = self.prox_morph[k][0]

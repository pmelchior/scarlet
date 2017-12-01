from __future__ import print_function, division
import numpy as np
from functools import partial

import proxmin
from . import transformations
from . import operators

import logging
logger = logging.getLogger("scarlet")

class Source(object):
    def __init__(self, center, size, psf=None, constraints=None, sed=None, morph=None, fix_sed=False, fix_morph=False, shift_center=0.2, prox_sed=None, prox_morph=None):

        # size needs to be odd
        size = int(size)
        if size%2 == 0:
            size += 1
        size = (size,) * 2

        # copy sed/morph if present
        self.sed = np.copy(sed) # works even if None
        if sed is not None:
            # to allow multi-component sources, need to have KxB array
            self.sed = self.sed.reshape((self.K, -1))

        self.morph = np.copy(morph)
        if morph is not None:
            # to allow multi-component sources, need to have K x Nx*Ny array
            self.morph = self.morph.reshape((self.K, -1))

        # set up psf and translations matrices
        if psf is None:
            P = None
        else:
            self.P = self._adapt_PSF(psf)
        self._gammaOp = transformations.GammaOp(size, P=P)

        # set center coordinates and translation operators
        # needs to have GammaOp set up first
        self.set_center(center, size=size)
        self.shift_center = shift_center

        # set constraints: first projection-style
        self.constraints = constraints
        self._set_sed_prox(prox_sed, fix_sed)
        self._set_morph_prox(prox_morph, fix_morph)
        # ... then ADMM-style constraints (proxs and L matrices)
        self._set_constraints()

    def __len__(self):
        # because SED/morph may not be properly set, need to return default 1
        try:
            return self.sed.shape[0]
        except IndexError:
            try:
                return self.morph.shape[0]
            except IndexError:
                return 1

    @property
    def K(self):
        return self.__len__()

    @property
    def B(self):
        try:
            return self.sed.shape[1]
        except IndexError:
            return 0

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
        from math import floor
        return (int(floor(self.y)), int(floor(self.x)))

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

    def set_center(self, center, size=None):
        self.y, self.x = center

        if size is None:
            size = (self.Ny, self.Nx)

        # make cutout of in units of the original image frame (that defines xy)
        # ensure odd pixel number
        y_, x_ = self.center_int
        self.left, self.right = x_ - size[0]//2, x_ + size[0]//2 + 1
        self.bottom, self.top = y_ - size[1]//2, y_ + size[1]//2 + 1

        # since slice wrap around if start or stop are negative, need to sanitize
        # start values (stop always postive)
        self.bb = (slice(None), slice(max(0, self.bottom), self.top), slice(max(0, self.left), self.right))

        dx = self.x - x_
        dy = self.y - y_
        self.Gamma = self._gammaOp(dy,dx)

    def resize(self, size):
        print ("resize to " + str(size))
        #raise NotImplementedError()

    def init_source(self, img, weights=None):
        # init with SED of the peak pixels
        # TODO: what should we do if peak is saturated?
        B = img.shape[0]
        self.sed = np.empty((1, B))
        y_, x_ = self.center_int
        self.sed[0] = img[:,y_,x_]
        # ensure proper normalization
        self.sed[0] = proxmin.operators.prox_unity_plus(self.sed[0], 0)

        # init with monotonized image in bb
        if self._gammaOp.P is None:
            self.morph = np.zeros((1, self.Ny, self.Nx))
            morph_slice = self.get_slice_for(img.shape)
            # TODO: account for per-band variations
            self.morph[0][morph_slice[1:]] = img[self.bb].sum(axis=0)
            self.morph = self.morph[0].reshape((1, self.Ny*self.Nx))
            # use predefined strict_monotonicity op if present
            if self.constraints is not None and "m" in self.constraints.keys():
                pos = self.prox_morph[0].find(operators.prox_strict_monotonic)
                prox_mono = self.prox_morph[0].operators[pos]
            else:
                shape = (self.Ny, self.Nx)
                thresh = 1./min(shape)
                prox_mono = operators.prox_strict_monotonic(shape, thresh=thresh)
            self.morph[0] = prox_mono(self.morph[0], 0)
        else: # simply central pixel
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
        return [self._invert_with_zeros(np.sqrt(np.dot(a.T, np.multiply(w, a[:,None])))) for a in self.sed]

    def get_sed_error(self, weights):
        w = np.zeros(self.shape)
        w[self.get_slice_for(weights.shape)] = weights[self.bb]
        w = w.reshape(self.B, -1)
        # See explanation in get_morph_error
        return [self._invert_with_zeros(np.sqrt(np.dot(s,np.multiply(w.T, s[None,:].T)))) for s in self.morph]

    def _invert_with_zeros(self, x):
        mask = (x == 0)
        x[~mask] = 1./x[~mask]
        x[mask] = -1
        return x

    def set_morph_sparsity(self, weights):
        if self.constraints is not None and ("l0" in self.constraints.keys() or "l1" in self.constraints.keys()):
            morph_error = self.get_morph_error(weights)
            # filter out -1s for pixels outside of weight images
            morph_std = np.array([np.median(mek[mek != -1]) for mek in morph_error])
            # Note: don't use hard/soft thresholds with _plus (non-negative) because
            # that is either happening with prox_plus before in the
            # AlternatingProjections or is not indended
            if "l0" in self.constraints.keys():
                morph_std *= self.constraints['l0']
                for k in range(self.K):
                    pos = self.prox_morph[k].find(proxmin.operators.prox_hard)
                    self.prox_morph[k].operators[pos] = partial(proxmin.operators.prox_hard, thresh=morph_std[k])
            elif "l1" in self.constraints.keys():
                # TODO: Is l1 penalty relative to noise meaningful?
                morph_std *= self.constraints['l1']
                for k in range(self.K):
                    pos = self.prox_morph[k].find(proxmin.operators.prox_soft)
                    self.prox_morph[k].operators[pos] = partial(proxmin.operators.prox_soft, thresh=morph_std[k])
            return morph_std
        else:
            return np.zeros(self.K)

    def _set_sed_prox(self, prox_sed, fix_sed):
        if hasattr(fix_sed, '__iter__') and len(fix_sed) == self.K:
            self.fix_sed = fix_sed
        else:
            self.fix_sed = [fix_sed] * self.K

        if prox_sed is None:
            self.prox_sed = [proxmin.operators.prox_unity_plus] * self.K
        else:
            if hasattr(prox_sed, '__iter__') and len(prox_sed) == self.K:
                self.prox_sed = prox_sed
            else:
                self.prox_sed = [prox_sed] * self.K

    def _set_morph_prox(self, prox_morph, fix_morph):

        if hasattr(fix_morph, '__iter__') and len(fix_morph) == self.K:
            self.fix_morph = fix_morph
        else:
            self.fix_morph = [fix_morph] * self.K

        # prox_morph overwrites everything!
        if prox_morph is not None:
            logger.info("prox_morph set from init argument")

            if hasattr(prox_morph, '__iter__') and len(prox_morph) == self.K:
                self.prox_morph = prox_morph
            else:
                self.prox_morph = [prox_morph] * self.K
        else:
            self.prox_morph = [[proxmin.operators.prox_plus],] * self.K
            if self.constraints is not None:

                # Note: don't use hard/soft thresholds with _plus (non-negative) because
                # that is either happening with prox_plus before or is not indended
                if "l0" in self.constraints.keys():
                    if "l1" in self.constraints.keys():
                        # L0 has preference
                        logger.info("l1 penalty ignored in favor of l0 penalty")
                    for k in range(self.K):
                        self.prox_morph[k].append(partial(proxmin.operators.prox_hard, thresh=0))
                elif "l1" in self.constraints.keys():
                    for k in range(self.K):
                        self.prox_morph[k].append(partial(proxmin.operators.prox_soft, thresh=0))

                if "m" in self.constraints.keys():
                    shape = (self.Ny, self.Nx)
                    thresh = 0
                    for k in range(self.K):
                        self.prox_morph[k].append(operators.prox_strict_monotonic(shape, thresh=thresh))

            for k in range(self.K):
                if len(self.prox_morph[k]) > 1:
                    self.prox_morph[k] = proxmin.operators.AlternatingProjections(self.prox_morph[k], repeat=1)
                else:
                    self.prox_morph[k] = self.prox_morph[k][0]

    def _set_constraints(self):
        self.proxs_g = [None, []] # no constraints on A matrix
        self.Ls = [None, []]
        if self.constraints is None:
            self.proxs_g[1] = None
            self.Ls[1] = None
            return

        shape = (self.Ny, self.Nx)
        for c in self.constraints.keys():
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

    def _adapt_PSF(self, psf):
        # Simpler for likelihood gradients if psf = const across B
        if hasattr(psf, 'shape'): # single matrix
            return transformations.getPSFOp(psf, (self.Ny, self.Nx))

        P = []
        for b in range(len(psf)):
            P.append(transformations.getPSFOp(psf[b], (self.Ny, self.Nx)))
        return P

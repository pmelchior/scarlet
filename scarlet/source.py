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
        if isinstance(psf, transformations.GammaOp):
            self._gammaOp = psf
        else:
            self._gammaOp = transformations.GammaOp(self.shape, psf=psf)

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
        return np.round(self.center).astype('int')

    @property
    def has_psf(self):
        return self._gammaOp.psf is not None

    def get_slice_for(self, im_shape):
        # slice so that self.image[k][slice] corresponds to image[self.bb]
        NY, NX = im_shape[1:]

        left = max(0, -self.left)
        bottom = max(0, -self.bottom)
        right = self.Nx - max(0, self.right - NX)
        top = self.Ny - max(0, self.top - NY)
        return (slice(None), slice(bottom, top), slice(left, right))

    def get_model(self, combine=True, Gamma=None, use_sed=True):
        if Gamma is None:
            Gamma = self.Gamma
        if use_sed:
            sed = self.sed
        else:
            sed = np.ones_like(self.sed)
        # model for all components of this source
        if not self.has_psf:
            model = np.empty((self.K, self.B, self.Ny*self.Nx))
            for k in range(self.K):
                model[k] = np.outer(sed[k], Gamma.dot(self.morph[k]))
        else:
            model = np.zeros((self.K, self.B, self.Ny*self.Nx))
            for k in range(self.K):
                for b in range(self.B):
                    model[k,b] += sed[k,b] * Gamma[b].dot(self.morph[k])

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
        self.Gamma = self._gammaOp(dx, self.shape)

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
        # prevent zeros from messing up:
        # set them at a very small value, and zero them out at the end
        mask = (w.sum(axis=0) == 0).flatten()
        if mask.sum():
            w[:,mask] = 1e-3 * w[:,~mask].min(axis=1)[:,None]

        # compute direct error propagation assuming only this source SED(s)
        # and the pixel covariances: Sigma_morph = diag((A^T Sigma^-1 A)^-1)
        # CAVEAT: If done on the entire A matrix, degeneracies in the linear
        # solution arise and substantially amplify the error estimate:
        # Instead, estimate noise for each component separately:
        # simple multiplication for diagonal pixel covariance matrix
        if not self.has_psf:
            me = [1./np.sqrt(np.dot(a.T, np.multiply(w, a[:,None]))) for a in self.sed]
        else:
            # see Blend.steps_f for details for the complete covariance matrix
            import scipy.sparse
            Sigma_pix = scipy.sparse.diags(w.flatten(), 0)
            PA = [scipy.sparse.bmat([[self.sed[k,b] * self.Gamma[b]] for b in range(self.B)])  for k in range(self.K)]
            Sigma_s = [PAk.T.dot(Sigma_pix.dot(PAk)) for PAk in PA]
            me = [np.sqrt(np.diag(np.linalg.inv(Sigma_sk.toarray()))) for Sigma_sk in Sigma_s]

            # TODO: the matrix inversion is instable if the PSF gets wide
            # possible options: Tikhonov regularization or similar
        if mask.sum():
            for mek in me:
                mek[mask] = 0
        return me

    def get_sed_error(self, weights):
        w = np.zeros(self.shape)
        w[self.get_slice_for(weights.shape)] = weights[self.bb]
        w = w.reshape(self.B, -1)
        # NOTE: zeros weights would only be a problem if an entire band is missing

        # See explanation in get_morph_error and Blend.steps_f
        if not self.has_psf:
            return [1./np.sqrt(np.dot(s,np.multiply(w.T, s[None,:].T))) for s in self.morph]
        else:
            import scipy.sparse
            Sigma_pix = scipy.sparse.diags(w.flatten(), 0)
            model = self.get_model(combine=False, use_sed=False)
            PS = [scipy.sparse.block_diag([model[k,b,:,:].reshape((1,-1)).T for b in range(self.B)]) for k in range(self.K)]
            return [np.sqrt(np.diag(np.linalg.inv(PSk.T.dot(Sigma_pix.dot(PSk)).toarray()))) for PSk in PS]

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
                thresh = self.constraints["l0"]
                if "l1" in self.constraints.keys():
                    # L0 has preference
                    logger.info("l1 penalty ignored in favor of l0 penalty")
                for k in range(self.K):
                    self.prox_morph[k].append(partial(proxmin.operators.prox_hard, thresh=thresh))
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
                M = transformations.getRadialMonotonicOp(shape, useNearest=self.constraints[c])
                for k in range(self.K):
                    self.Ls[1].append(M)
                    self.proxs_g[1].append(proxmin.operators.prox_plus)
            elif c == "S":
                # zero deviation of mirrored pixels
                S = transformations.getSymmetryOp(shape)
                for k in range(self.K):
                    self.Ls[1].append(S)
                    self.proxs_g[1].append(proxmin.operators.prox_zero)
            elif c == "C":
                # cone method for monotonicity: exact but VERY slow
                useNearest = self.constraints.get("M", False)
                G = transformations.getRadialMonotonicOp(shape, useNearest=useNearest).toarray()
                for k in range(self.K):
                    self.Ls[1].append(None)
                    self.proxs_g[1].append(partial(operators.prox_cone, G=G))
            elif c == "X":
                # l1 norm on gradient in X for TV_x
                cx = int(self.Nx)
                Gx = proxmin.transformations.get_gradient_x(shape, cx)
                for k in range(self.K):
                    self.Ls[1].append(Gx)
                    self.proxs_g[1].append(partial(proxmin.operators.prox_soft, thresh=self.constraints[c]))
            elif c == "Y":
                # l1 norm on gradient in Y for TV_y
                cy = int(self.Ny)
                Gy = proxmin.transformations.get_gradient_y(shape, cy)
                for k in range(self.K):
                    self.Ls[1].append(Gy)
                    self.proxs_g[1].append(partial(proxmin.operators.prox_soft, thresh=self.constraints[c]))

        # with several projection operators in prox_morph:
        # use AlternatingProjections to link them together
        for k in range(self.K):
            if len(self.prox_morph[k]) > 1:
                self.prox_morph[k] = proxmin.operators.AlternatingProjections(self.prox_morph[k], repeat=1)
            else:
                self.prox_morph[k] = self.prox_morph[k][0]

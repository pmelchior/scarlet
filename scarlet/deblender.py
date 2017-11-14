from __future__ import print_function, division
import numpy as np
from functools import partial

import proxmin
from proxmin.nmf import Steps_AS
from . import transformations
from . import operators

import logging
logger = logging.getLogger("scarlet")

class Source(object):
    def __init__(self, x, y, img, psf=None, constraints=None, sed=None, morph=None, fix_sed=False, fix_morph=False, shift_center=0.2, prox_sed=None, prox_morph=None):

        # set up coordinates and images sizes
        self.x = x
        self.y = y
        self.B, self.Ny, self.Nx = img.shape
        # TODO: use bounding box as argument, make cutout of img (odd pixel number)
        self.bb_ll = (0,0)
        self.bb_shape = (self.Ny, self.Nx)

        if psf is None:
            self.P = None
        else:
            self.P = self._adapt_PSF(psf)
        self.shift_center = shift_center
        self._translate_psf()

        # set up sed and morphology: initial values, proxs and update
        self.constraints = constraints
        self._set_sed(img, sed, prox_sed, fix_sed)
        self._set_morph(img, morph, prox_morph, fix_morph)

        # set up ADMM-style constraints: proxs and matrices
        self._set_constraints()

    def __len__(self):
        return self.sed.shape[0]

    @property
    def K(self):
        return self.__len__()

    @property
    def image(self):
        return self.morph.reshape((self.K,self.Ny,self.Nx)) # this *should* be a view

    def get_model(self, combine=True, Gamma=None):
        if Gamma is None:
            Gamma = self.Gamma
        # model for all components of this source
        if hasattr(Gamma, 'shape'): # single matrix: one for all bands
            model = np.empty((self.K,self.B,self.Ny*self.Nx))
            for k in range(self.K):
                model[k] = np.outer(self.sed[k], Gamma.dot(self.morph[k]))
        else:
            model = np.zeros((self.K,self.B,self.Ny*self.Nx))
            for k in range(self.K):
                for b in range(self.B):
                    model[k,b] += self.sed[k,b] * Gamma[b].dot(self.morph[k])

        # reshape the image into a 2D array
        model = model.reshape(self.K,self.B,self.Ny,self.Nx)
        if combine:
            model = model.sum(axis=0)
        return model

    def get_shifted_model(self, model=None):
        if model is None:
            model = self.get_model(combine=True)
        diff_img = [self.get_model(combine=True, Gamma=self.dGamma_x), self.get_model(combine=True, Gamma=self.dGamma_y)]
        diff_img[0] = (model-diff_img[0])/self.shift_center
        diff_img[1] = (model-diff_img[1])/self.shift_center
        return diff_img

    def _set_sed(self, img, sed, prox_sed, fix_sed):
        if sed is None:
            self._init_sed(img)
        else:
            # to allow multi-component sources, need to have KxB array
            if len(sed.shape) != 2:
                self.sed = sed.reshape((1,len(sed)))
            else:
                self.sed = sed.copy()

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

    def _init_sed(self, img):
        # init A from SED of the peak pixels
        self.sed = np.empty((1,self.B))
        self.sed[0] = img[:,int(self.y),int(self.x)]
        # ensure proper normalization
        self.sed[0] = proxmin.operators.prox_unity_plus(self.sed[0], 0)

    def _set_morph(self, img, morph, prox_morph, fix_morph):
        if morph is None:
            self._init_morph(img)
        else:
            # to allow multi-component sources, need to have K x Nx*Ny array
            if len(morph.shape) == 3: # images for each component
                self.morph = morph.reshape((morph.shape[0], -1))
            elif len(morph.shape) == 2:
                if morph.shape[0] == self.K: # vectors for each component
                    self.morph = morph.copy()
                else: # morph image for one component
                    self.morph = morph.flatten().reshape((1, -1))
            elif len(morph.shape) == 1: # vector for one component
                self.morph = morph.reshape((1, -1))
            else:
                raise NotImplementedError("Shape of morph not understood: %r" % morph.shape)

        if hasattr(fix_morph, '__iter__') and len(fix_morph) == self.K:
            self.fix_morph = fix_morph
        else:
            self.fix_morph = [fix_morph] * self.K

        if prox_morph is None:
            if self.constraints is None or ("l0" not in self.constraints.keys() and "l1" not in self.constraints.keys()):
                self.prox_morph = [proxmin.operators.prox_plus] * self.K
            else:
                # L0 has preference
                if "l0" in self.constraints.keys():
                    if "l1" in self.constraints.keys():
                        logger.warn("l1 penalty ignored in favor of l0 penalty")
                    self.prox_morph = [partial(proxmin.operators.prox_hard, thresh=self.constraints['l0'])] * self.K
                else:
                    self.prox_morph = [partial(proxmin.operators.prox_soft_plus, thresh=self.constraints['l1'])] * self.K
        else:
            if hasattr(prox_morph, '__iter__') and len(prox_morph) == self.K:
                self.prox_morph = prox_morph
            else:
                self.prox_morph = [prox_morph] * self.K

    def _init_morph(self, img):
        # TODO: init from the cutout values (ignoring blending)
        cx, cy = int(self.Nx/2), int(self.Ny/2)
        self.morph = np.zeros((1, self.Ny*self.Nx))
        tiny = 1e-10
        flux = np.abs(img[:,cy,cx].mean()) + tiny
        self.morph[0,cy*self.Nx+cx] = flux

    def _translate_psf(self):
        """Build the operators to perform a translation
        """
        Tx, Ty = transformations.getTranslationOps((self.Ny, self.Nx), self.x, self.y)
        self.Gamma = transformations.getGammaOp(Tx, Ty, self.B, self.P)

        # for centroid shift: compute shifted Gammas
        if self.shift_center:
            # TODO: optimize dxy to be comparable to likely shift
            # TODO: Alternative: Grid of shifted PSF to interpolate at any given ddx/ddy
            dxy = self.shift_center
            Tx_, Ty_ = transformations.getTranslationOps((self.Ny, self.Nx), self.x, self.y, dxy, dxy)
            # get the shifted image in x/y by adjusting only the Tx/Ty
            self.dGamma_x = transformations.getGammaOp(Tx_, Ty, self.B, self.P)
            self.dGamma_y = transformations.getGammaOp(Tx, Ty_, self.B, self.P)

    def _adapt_PSF(self, psf):
        shape = (self.Ny, self.Nx)
        # Simpler for likelihood gradients if psf = const across B
        if hasattr(psf, 'shape'): # single matrix
            return transformations.getPSFOp(psf, shape)

        P = []
        for b in range(self.B):
            P.append(transformations.getPSFOp(psf[b], shape))
        return P

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


class Blend(object):
    """The blended scene as interpreted by the deblender.
    """
    def __init__(self, sources):
        assert len(sources)
        self._register_sources(sources)

        # list of all proxs_g and Ls: first A, then S
        self.proxs_g = [source.proxs_g[0] for source in self.sources] + [source.proxs_g[1] for source in self.sources]
        self.Ls = [source.Ls[0] for source in self.sources] + [source.Ls[1] for source in self.sources]    # for S

    def _register_sources(self, sources):
        self.sources = sources # do not copy!
        self.M = len(self.sources)
        self.K =  sum([source.K for source in self.sources])
        self.psf_per_band = not hasattr(sources[0].Gamma, 'shape')

        # lookup of source/component tuple given component number k
        self._source_of = []
        self.update_centers = False
        for m in range(self.M):
            self.update_centers |= bool(self.sources[m].shift_center)
            for l in range(self.sources[m].K):
                self._source_of.append((m,l))

    def source_of(self, k):
        return self._source_of[k]

    def component_of(self, m, l):
        # search for k that has this (m,l), inverse of source_of
        for k in range(self.K):
            if self._source_of[k] == (m,l):
                return k
        raise IndexError

    def __len__(self):
        """Number of distinct sources"""
        return self.M

    def setData(self, img, psfs=None, weights=None, update_order=None, slack=0.9):
        self.it = 0
        self.center_min_dist = 1e-3
        self.center_wait = 10
        self.center_skip = 10
        if weights is None:
            self.weights = Wmax = 1
        else:
            self.weights = weights
            Wmax = np.max(self.weights)
        if update_order is None:
            update_order = range(2)
        self.img = img
        self.update_order = update_order
        self._stepAS = Steps_AS(Wmax=Wmax, slack=slack, update_order=self.update_order)
        self.step_AS = [None] * 2
        B, Ny, Nx = img.shape
        self.A, self.S = np.empty((B,self.K)), np.empty((self.K,Nx*Ny))
        self._set_AS()

    def _set_AS(self):
        for k in range(self.K):
            m,l = self._source_of[k]
            self.A[:,k] = self.sources[m].sed[l]
            # TODO: This will be more complicated with source boxes!
            self.S[k,:] = self.sources[m].morph[l].flatten()

    def prox_f(self, X, step, Xs=None, j=None):

        # which update to do now
        AorS = j//self.K
        k = j%self.K
        B, Ny, Nx = self.img.shape

        # computing likelihood gradients for S and A: only once per iteration
        if AorS == self.update_order[0] and k==0:
            models =  [source.get_model(combine=True) for source in self.sources]
            model = np.sum(models, axis=0)
            self.diff = (self.weights*(model-self.img)).reshape(B, Ny*Nx)

            # update positions
            if self.update_centers:
                if self.it >= self.center_wait and self.it % self.center_skip == 0:
                    self._update_positions(models)
            self.it += 1

        # A update
        if AorS == 0:
            m,l = self.source_of(k)
            if not self.sources[m].fix_sed[l]:
                # gradient of likelihood wrt A
                if not self.psf_per_band:
                    grad = self.diff.dot(self.sources[m].Gamma.dot(self.sources[m].morph[l]))
                else:
                    grad = np.empty_like(X)
                    for b in range(B):
                        grad_[b] = self.diff[b].dot(self.sources[m].Gamma[b].dot(self.sources[m].morph[l]))

                # apply per component prox projection and save in source
                self.sources[m].sed[l] =  self.sources[m].prox_sed[l](X - step*grad, step)
            return self.sources[m].sed[l]

        # S update
        elif AorS == 1:
            m,l = self.source_of(k)
            if not self.sources[m].fix_morph[l]:
                # gradient of likelihood wrt S
                grad = np.zeros_like(X)
                if not self.psf_per_band:
                    for b in range(B):
                        grad += self.sources[m].sed[l,b]*self.sources[m].Gamma.T.dot(self.diff[b])
                else:
                    for b in range(B):
                        grad += self.sources[m].sed[l,b]*self.sources[m].Gamma[b].T.dot(self.diff[b])

                # apply per component prox projection and save in source
                self.sources[m].morph[l] = self.sources[m].prox_morph[l](X - step*grad, step)
            return self.sources[m].morph[l]
        else:
            raise ValueError("Expected index j in [0,%d]" % (2*self.K))

    def _update_positions(self, models):
        y = self.diff.flatten()
        for k in range(self.K):
            if self.sources[k].shift_center:
                diff_x,diff_y = self.sources[k].get_shifted_model(model=models[k])
                # least squares for the shifts given the model residuals
                MT = np.vstack([diff_x.flatten(), diff_y.flatten()])
                if not hasattr(self.weights,'shape'): # no/flat weights
                    ddx,ddy = np.dot(np.dot(np.linalg.inv(np.dot(MT, MT.T)), MT), y)
                else:
                    ddx,ddy = np.dot(np.dot(np.linalg.inv(np.dot(MT, MT.T*self.weights.flatten()[:,None])), MT), y)
                if ddx**2 + ddy**2 > self.center_min_dist**2:
                    self.sources[k].x -= ddx
                    self.sources[k].y -= ddy
                    self.sources[k]._translate_psf()
                    logger.info("Source %d shifted by (%.3f/%.3f) to (%.3f/%.3f)" % (k, -ddx, -ddy, self.sources[k].x, self.sources[k].y))

    def steps_f(self, j, Xs):
        # which update to do now
        AorS = j//self.K
        k = j%self.K

        # computing likelihood gradients for S and A: only once per iteration
        if AorS == self.update_order[0] and k==0:
            self._set_AS()
            self.step_AS[0] = self._stepAS(0, [self.A, self.S])
            self.step_AS[1] = self._stepAS(1, [self.A, self.S])
        return self.step_AS[AorS]

    def get_model(self, m=None, combine=True):
        """Build the current model
        """
        if m is None:
            # needs to have source components combined
            model = [source.get_model(combine=True) for source in self.sources]
            if combine:
                model = np.sum(model, axis=0)
            return model
        else:
            return self.source[m].get_model(combine=combine)


def deblend(img,
            sources,
            weights=None,
            psf=None,
            sky=None,
            max_iter=200,
            e_rel=1e-3,
            e_abs=1e-3,
            slack = 0.9,
            update_order=None,
            steps_g=None,
            steps_g_update='steps_f',
            traceback=False
            ):

    if sky is None:
        Y = img
    else:
        Y = img-sky

    # construct Blender from sources and define objective function
    blend = Blend(sources)
    blend.setData(Y, weights=weights, update_order=update_order)
    #f = partial(blend.prox_likelihood, Y=Y, W=weights, update_order=update_order)
    prox_f = blend.prox_f
    steps_f = blend.steps_f
    proxs_g = blend.proxs_g
    Ls = blend.Ls

    # run the NMF with those constraints
    XA = []
    XS = []
    for k in range(blend.K):
        m,l = blend.source_of(k)
        XA.append(blend.sources[m].sed[l])
        XS.append(blend.sources[m].morph[l])
    X = XA+XS
    res = proxmin.algorithms.bsdmm(X, prox_f, steps_f, proxs_g, steps_g=steps_g, Ls=Ls, update_order=update_order, steps_g_update=steps_g_update, max_iter=max_iter, e_rel=e_rel, e_abs=e_abs, accelerated=True, traceback=traceback)

    if not traceback:
        return blend
    else:
        _, tr = res
        return blend, tr

from __future__ import print_function, division
import numpy as np

from . import transformations
from . import operators

import proxmin
from proxmin.nmf import Steps_AS

from functools import partial
import logging
from numbers import Number


# Set basestring in python 3
try:
    basestring
except NameError:
    basestring = str

logger = logging.getLogger("scarlet.deblender")

class Source(object):
    def __init__(self, x, y, img, psf=None, constraints=None, sed=None, morph=None, fix_sed=False, fix_morph=False, fix_center=False, prox_sed=None, prox_morph=None):
        # TODO: use bounding box as argument, make cutout of img (odd pixel number)
        # and initialize morph directly from cutout instead if point source

        # copy data structures
        self.x = x
        self.y = y
        self.B, self.Ny, self.Nx = img.shape
        self._translate_psfs(psf)

        from copy import deepcopy
        self.constraints = deepcopy(constraints)

        if sed is None:
            self._init_sed(self.x, self.y, img)
        else:
            # to allow multi-component sources, need to have KxB array
            if len(sed.shape) != 2:
                self.sed = sed.reshape((1,len(sed)))
            else:
                self.sed = sed.copy()

        if morph is None:
            self._init_morph(self.x, self.y, img)
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

        if hasattr(fix_sed, '__inter__') and len(fix_sed) == self.K:
            self.fix_sed = fix_sed
        else:
            self.fix_sed = [fix_sed] * self.K
        if hasattr(fix_morph, '__inter__') and len(fix_morph) == self.K:
            self.fix_morph = fix_morph
        else:
            self.fix_morph = [fix_morph] * self.K
        self.fix_center = fix_center

        if prox_sed is None:
            self.prox_sed = [proxmin.operators.prox_unity_plus] * self.K
        else:
            if hasattr(prox_sed, '__inter__') and len(prox_sed) == self.K:
                self.prox_sed = prox_sed
            else:
                self.prox_sed = [prox_sed] * self.K

        if prox_morph is None:
            if constraints is None or (constraints['l0'] <= 0 and constraints['l1'] <= 0):
                self.prox_morph = [proxmin.operators.prox_plus] * self.K
            else:
                # L0 has preference
                if constraints['l0'] > 0:
                    if constraints['l1'] > 0:
                        logger.warn("warning: l1 penalty ignored in favor of l0 penalty")
                    self.prox_morph = [partial(proxmin.operators.prox_hard, thresh=constraints['l0'])] * self.K
                else:
                    self.prox_morph = [partial(proxmin.operators.prox_soft_plus, thresh=constraints['l1'])] * self.K
        else:
            if hasattr(prox_morph, '__inter__') and len(prox_morph) == self.K:
                self.prox_morph = prox_morph
            else:
                self.prox_morph = [prox_morph] * self.K

        # TODO: generate proxs_g and Ls from suitable entries in constraints
        self.proxs_g = [None] * self.K
        self.Ls = [None] * self.K
        # if constraints is not None:
        #     linear_constraints = {
        #         "M": proxmin.operators.prox_plus,  # positive gradients
        #         "S": proxmin.operators.prox_zero,  # zero deviation of mirrored pixels,
        #         "c": partial(operators.prox_cone, G=transformations.getRadialMonotonicOp((N,M), useNearest=monotonicUseNearest).toarray()),
        #         "X": proxmin.operators.prox_plus, # positive X gradient
        #         "Y": proxmin.operators.prox_plus, # positive Y gradient
        #         "x": partial(proxmin.operators.prox_soft, thresh=smoothness), # l1 norm on X gradient
        #         "y": partial(proxmin.operators.prox_soft, thresh=smoothness), # l1 norm on Y gradient
        #     }

        # TODO: need to get per-source translation incl PSF
        # T = Translation(_peaks, (N,M), B, P_, txy_diff, max_shift, txy_thresh, fit_positions, txy_wait, txy_skip, traceback)
        # self.Tx = Tx
        # self.Ty = Ty
        # self.Gamma = Gamma

    def __len__(self):
        return self.sed.shape[0]

    @property
    def K(self):
        return self.__len__()

    @property
    def image(self):
        return self.morph.reshape((-1,self.Nx,self.Ny)) # this *should* be a view

    def get_model(self, combine=True):
        # model for all components of this source
        if hasattr(self.Gamma, 'shape'): # single matrix: one for all bands
            model = np.empty((self.K,self.B,self.Ny*self.Nx))
            for k in range(self.K):
                model[k] = np.outer(self.sed[k], self.Gamma.dot(self.morph[k]))
        else:
            model = np.zeros((self.K,self.B,self.Ny*self.Nx))
            for k in range(self.K):
                for b in range(self.B):
                    model[k,b] += self.sed[k,b] * self.Gamma[b].dot(self.morph[k])

        # reshape the image into a 2D array
        model = model.reshape(self.K,self.B,self.Ny,self.Nx)
        if combine:
            model = model.sum(axis=0)
        return model

    # def get_diff_images(self, data, models, A, S, W):
    #     """Get differential images to fit translations
    #     """
    #     from .deblender import get_peak_model
    #
    #     dxy = self.differential
    #     diff_images = []
    #     for pk, peak in enumerate(self.cat.objects):
    #         dx = self.cx - peak.x
    #         dx = self.cy - peak.y
    #         # Combine all of the components of the current peak into a model
    #         model = []
    #         for k in peak.component_indices:
    #             model.append(models[k])
    #         model = np.sum(model, axis=0)
    #         Tx, Ty = self.get_translation_ops(pk, dxy, dxy, update=False)
    #         # Get the difference image in x by adjusting only the x
    #         # component by the differential amount dxy
    #         Gk = self.build_Gamma(pk, Tx=Tx, update=False)
    #         diff_img = []
    #         for k in peak.component_indices:
    #             diff_img.append(get_peak_model(A[:,k], S[k], Gk))
    #         diff_img = np.sum(diff_img, axis=0)
    #         diff_img = (model-diff_img)/dxy
    #         diff_images.append(diff_img)
    #         # Do the same for the y difference image
    #         Gk = self.build_Gamma(pk, Ty=Ty, update=False)
    #         diff_img = []
    #         for k in peak.component_indices:
    #             diff_img.append(get_peak_model(A[:,k], S[k], Gk))
    #         diff_img = np.sum(diff_img, axis=0)
    #         diff_img = (model-diff_img)/dxy
    #         diff_images.append(diff_img)
    #     return diff_images

    def _translate_psfs(self, psf=None, ddx=0, ddy=0):
        """Build the operators to perform a translation
        """
        self.int_tx = {}
        self.int_ty = {}
        self.Tx, self.Ty = transformations.getTranslationOps((self.Ny, self.Nx), self, ddx, ddy)
        if psf is None:
            P = None
        else:
            P = self._adapt_PSF(psf)
        self.Gamma = transformations.getGammaOp(self.Tx, self.Ty, self.B, P)

    def _adapt_PSF(self, psf):
        shape = (self.Ny, self.Nx)
        # Simpler for likelihood gradients if psf = const across B
        if hasattr(psf, 'shape'): # single matrix
            return transformations.getPSFOp(psf, shape)

        P = []
        for b in range(self.B):
            P.append(transformations.getPSFOp(psf[b], shape))
        return P

    def _init_sed(self, x, y, img):
        # init A from SED of the peak pixels
        B, Ny, Nx = img.shape
        self.sed = np.empty((1,B))
        self.sed[0] = img[:,int(self.y),int(self.x)]
        # ensure proper normalization
        self.sed[0] = proxmin.operators.prox_unity_plus(self.sed[0], 0)

    def _init_morph(self, x, y, img):
        B, Ny, Nx = img.shape
        cx, cy = int(Nx/2), int(Ny/2)
        self.morph = np.zeros((1, Ny*Nx))
        tiny = 1e-10
        flux = np.abs(img[:,cy,cx].mean()) + tiny
        self.morph[0,cy*Nx+cx] = flux


class Blender(object):
    """The blended scene as interpreted by the deblender.
    """
    def __init__(self, sources):
        assert len(sources)
        self._register_sources(sources)

        # container for gradients of S and A. Note: A is B x K
        B, Nx, Ny = self.sources[0].B, self.sources[0].Nx, self.sources[0].Ny
        self.A, self.S = np.empty((B,self.K)), np.empty((self.K,Nx*Ny))
        for k in range(self.K):
            m,l = self._source_of[k]
            self.A[:,k] = self.sources[m].sed[l]
            self.S[k,:] = self.sources[m].morph[l].flatten()

        # list of all proxs_g and Ls
        self.proxs_g = None
        self.Ls = None
        # TODO: activate this with per-source optimization
        # self.proxs_g = [[source.proxs_g[0] for source in self.sources], # for A
        #                 [source.proxs_g[1] for source in self.sources]] # for S
        # self.Ls = [[source.Ls[0] for source in self.sources], # for A
        #            [source.Ls[1] for source in self.sources]] # for S

    def _register_sources(self, sources):
        self.sources = sources # do not copy!
        self.M = len(self.sources)
        self.K =  sum([source.K for source in self.sources])
        self.psf_per_band = not hasattr(sources[0].Gamma, 'shape')

        # lookup of source/component tuple given component number k
        self._source_of = []
        for m in range(self.M):
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

    def prox_likelihood(self, X, step, Xs=None, j=None, Y=None, W=1, update_order=[0,1]):

        # computing likelihood gradients for S and A: only once per iteration
        B, Nx, Ny = self.sources[0].B, self.sources[0].Nx, self.sources[0].Ny
        if j == update_order[0]:
            model = self.get_model(combine=True)
            self.diff = (W*(model-Y)).reshape(B, Ny*Nx)
            """
            # TODO: centroid updates
            if T.fit_positions:
                T.update_positions(Y, model, A, S, W)
            """

        # A update
        if j == 0:
            for k in range(self.K):
                m,l = self.source_of(k)
                if not self.sources[m].fix_sed[l]:
                    # gradient of likelihood wrt A
                    if not self.psf_per_band:
                        self.A[:,k] = self.diff.dot(self.sources[m].Gamma.dot(self.sources[m].morph[l]))
                    else:
                        for b in range(B):
                            self.A[b,k] = self.diff[b].dot(self.sources[m].Gamma[b].dot(self.sources[m].morph[l]))

                    # apply per component prox projection and save in source
                    self.sources[m].sed[l] =  self.sources[m].prox_sed[l](self.sources[m].sed[l] - step*self.A[:,k], step)

                    # copy into result matrix
                    self.A[:,k] = self.sources[m].sed[l]
            return self.A

        # S update
        elif j == 1:
            for k in range(self.K):
                m,l = self.source_of(k)
                if not self.sources[m].fix_morph[l]:
                    # gradient of likelihood wrt S
                    self.S[k,:] = 0
                    if not self.psf_per_band:
                        for b in range(B):
                            self.S[k,:] += self.sources[m].sed[l,b]*self.sources[m].Gamma.T.dot(self.diff[b])
                    else:
                        for b in range(B):
                            self.S[k,:] += self.sources[m].sed[l,b]*self.sources[m].Gamma[b].T.dot(self.diff[b])

                    # apply per component prox projection and save in source
                    self.sources[m].morph[l] = self.sources[m].prox_morph[l](self.sources[m].morph[l] - step*self.S[k], step)
                    # copy into result matrix
                    self.S[k,:] = self.sources[m].morph[l]
            return self.S

        else:
            raise ValueError("Expected index j in [0,1]")


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

def get_constraint_op(constraint, shape, seeks, useNearest=True):
    """Get appropriate constraint operator
    """
    N,M = shape
    if constraint is None or constraint == "c":
        return None
    elif constraint == "M":
        L = transformations.getRadialMonotonicOp((N,M), useNearest=useNearest)
    elif constraint == "S":
        L = transformations.getSymmetryOp((N,M))
    elif constraint == "X" or constraint == "x":
        cx = int(shape[1]/2)
        L = proxmin.transformations.get_gradient_x(shape, cx)
    elif constraint == "Y" or constraint == "y":
        cy = int(shape[0]/2)
        L = proxmin.transformations.get_gradient_y(shape, cy)
    # Create the matrix adapter for the operator
    adapter =  proxmin.utils.MatrixAdapter(L, axis=1)
    return adapter


def deblend(img,
            sources,
            weights=None,
            psf=None,
            max_iter=1000,
            sky=None,
            e_rel=1e-3,
            e_abs=0,
            #monotonicUseNearest=False,
            traceback=False,
            #translation_thresh=1e-8,
            slack = 0.9,
            update_order=None,
            steps_g=None,
            steps_g_update='steps_f',
            truncate=False,
            #txy_diff=0.1,
            #max_shift=2,
            #txy_thresh=1e-8,
            #txy_wait=10,
            #txy_skip=10,
            ):

    if sky is None:
        Y = img
    else:
        Y = img-sky
    if weights is None:
        W = Wmax = 1
    else:
        Wmax = np.max(weights)

    """
    # TODO: move to Source
    T = Translation(_peaks, (N,M), B, P_, txy_diff, max_shift,
                    txy_thresh, fit_positions, txy_wait, txy_skip, traceback)
    if psf is None:
        P = psf
    else:
        P = adapt_PSF(psf, B, (Ny, Nx))
    logger.debug("Shape: {0}".format((Ny,Nx)))
    """

    # construct Blender from sources and define objective function
    blender = Blender(sources)
    if update_order is None:
        update_order = range(2)
    f = partial(blender.prox_likelihood, Y=Y, W=weights, update_order=update_order)
    steps_f = Steps_AS(Wmax=Wmax, slack=slack, update_order=update_order)
    proxs_g = blender.proxs_g
    Ls = blender.Ls

    # run the NMF with those constraints
    Xs = [blender.A, blender.S]
    res = proxmin.algorithms.bsdmm(Xs, f, steps_f, proxs_g, steps_g=steps_g, Ls=Ls, update_order=update_order,
                                  steps_g_update=steps_g_update, max_iter=max_iter, e_rel=e_rel, e_abs=e_abs,
                                  accelerated=True, traceback=traceback)

    if not traceback:
        # A, S = res
        return blender
    else:
        [A, S], tr = res
        return blender, tr

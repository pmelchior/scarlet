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
    def __init__(self, x, y, img, psfs=None, constraints=None, sed=None, morph=None, fix_sed=False, fix_morph=False, fix_center=False, prox_sed=None, prox_morph=None):
        self.x = x
        self.y = y
        self.B, self.Ny, self.Nx = img.shape

        # TODO: use bounding box as argument, make cutout of img (odd pixel number)
        # and initialize morph directly from cutout instead if point source

        from copy import deepcopy
        self.constraints = deepcopy(constraints)

        if sed is None:
            self._init_sed(self.x, self.y, img)
        else:
            # to allow multi-component sources, need to have BxK array
            if len(sed.shape) != 2:
                self.sed = sed.reshape((len(sed),1))
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
        return self.sed.shape[1]

    @property
    def K(self):
        return self.__len__()

    @property
    def image(self):
        return self.morph.reshape((-1,self.Nx,self.Ny)) # this *should* be a view

    def _init_sed(self, x, y, img):
        # init A from SED of the peak pixels
        B, Ny, Nx = img.shape
        if img is None:
            logger.warn("Using random A matrix for source at ({0},{0})".format(self.x, self.y))
            self.sed = np.random.rand((B,1))
        else:
            self.sed = np.empty((B,1))
            self.sed[:,0] = img[:,int(self.y),int(self.x)]
        # ensure proper normalization
        self.sed[:,0] = proxmin.operators.prox_unity_plus(self.sed[:,0], 0)

    def _init_morph(self, x, y, img):
        B, Ny, Nx = img.shape
        cx, cy = int(Nx/2), int(Ny/2)
        self.morph = np.zeros((1, Ny*Nx))
        tiny = 1e-10
        flux = np.abs(img[:,cy,cx].mean()) + tiny
        self.morph[0,cy*Nx+cx] = flux

    def get_model(self, combine=True):
        # model for all components of this source
        model = np.empty((self.K,self.B,self.Ny*self.Nx))
        for k in range(self.K):
            for b in range(self.B):
                model[k,b] += self.sed[b,k] * self.Gamma[b].dot(self.morph[k])
        # reshape the image into a 2D array
        model = model.reshape(self.K,self.B,self.Ny,self.Nx)
        if combine:
            model = model.sum(axis=0)
        return model

class Blender(object):
    """The blended scene as interpreted by the deblender.
    """
    def __init__(self, sources):
        assert len(sources)
        self._register_sources(sources)

        # container for gradients of S and A
        B, Nx, Ny = self.sources[0].B, self.sources[0].Nx, self.sources[0].Ny
        self.update = [np.empty((self.K,Nx*Ny)), np.empty((B,self.K))]

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

        # lookup of source/component tuple given component number k
        self.source_of = []
        for m in range(self.M):
            for l in range(self.sources[m].K):
                self.source_of.append((m,l))

    def component_of(self, m, l):
        # search for k that has this (m,l), inverse of source_of
        for k in range(self.K):
            if self.source_of[k] == (m,l):
                return k
        raise IndexError

    def __len__(self):
        """Number of distinct sources"""
        return self.M

    def prox_likelihood(self, X, step, Xs=None, j=None, Y=None, W=1, update_order=[0,1]):

        print (Y.shape, W.shape)
        if j > 0:
            raise ValueError("Expected index j in [0,1]")

        """
        # TODO: centroid updates
        if j == 0 and T.fit_positions:
            # Update the translation operators
            A, S = Xs
            model = get_model(A, S, T.Gamma, combine=False)
            T.update_positions(Y, model, A, S, W)
        """
        # computing likelihood gradients for S and A: only once per iteration
        B, Nx, Ny = self.sources[0].B, self.sources[0].Nx, self.sources[0].Ny
        if j == update_order[0]:
            model = self.get_model(combine=True)
            self.diff = W*(model-Y).reshape(B, Ny*Nx)

        # A update
        if j == 0:
            self.update[j][:,:] = 0
            for k in range(K):
                m,l = self.source_of(k)
                if not self.sources[m].fix_sed[l]:
                    # gradient of likelihood wrt A
                    self.update[j][:,k] = 0
                    for b in range(B):
                        self.update[j][b,k] = self.diff[b].dot(self.sources[m].Gamma[k][b].dot(self.sources[m].morph[l][k]))

                    # apply per component prox projection and save in source
                    self.sources[m].sed[l] = self.sources[m].prox_sed[l](self.sources[m].sed[l] - step*self.update[j][k], step)
                    # copy into result matrix
                    self.update[j][:,k] = self.sources[m].sed[l]

        # S update
        if j == 1:
            for k in range(self.K):
                m,l = self.source_of(k)
                if not self.sources[m].fix_morph[l]:
                    # gradient of likelihood wrt S
                    self.update[j][k,:] = 0
                    for b in range(B):
                        self.update[j][k,:] += self.sources[m].sed[l][b,k]*self.sources[m].Gamma[k][b].T.dot(self.diff[b])

                    # apply per component prox projection and save in source
                    self.sources[m].morph[l] = self.sources[m].prox_morph[l](self.sources[m].morph[l] - step*self.update[j][k], step)
                    # copy into result matrix
                    self.update[j][k,:] = self.sources[m].morph[l]

        return self.update[j]

    def get_model(self, m=None, combine=True):
        """Build the current model
        """
        if m is None:
            # needs to have source components combined
            models = [source.get_model(combine=True) for source in self.sources]
            if combine:
                model = np.sum(models, axis=0)
        else:
            return self.source[m].get_model(combine=combine)

def adapt_PSF(psf, B, shape):
    # Simpler for likelihood gradients if psf = const across B
    if len(psf.shape)==2: # single matrix
        return transformations.getPSFOp(psf, shape)

    P_ = []
    for b in range(B):
        P_.append(transformations.getPSFOp(psf[b], shape))
    return P_

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

def oddify(shape, truncate=False):
    """Get an odd number of rows and columns
    """
    if shape is None:
        shape = img.shape
    B,N,M = shape
    if N % 2 == 0:
        if truncate:
            N -= 1
        else:
            N += 1
    if M % 2 == 0:
        if truncate:
            M -= 1
        else:
            M += 1
    return B, N, M

def reshape_img(img, new_shape=None, truncate=False, fill=0):
    """Ensure that the image has an odd number of rows and columns
    """
    if new_shape is None:
        new_shape = oddify(img.shape, truncate)

    if img.shape != new_shape:
        B,N,M = img.shape
        _B,_N,_M = new_shape
        if B != _B:
            raise ValueError("The old and new shape must have the same number of bands")
        if truncate:
            _img = img[:,:_N, :_M]
        else:
            if fill==0:
                _img = np.zeros((B,_N,_M))
            else:
                _img = np.empty((B,_N,_M))
                _img[:] = fill
            _img[:,:N,:M] = img[:]
    else:
        _img = img
    return _img

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

    # vectorize image cubes
    B,Ny,Nx = img.shape

    # Ensure that the image has an odd number of rows and columns
    # TODO: should not be necessary if sources are defined on odd-pixel boxes
    _img = reshape_img(img, truncate=truncate)
    if _img.shape != img.shape:
        logger.warn("Reshaped image from {0} to {1}".format(img.shape, _img.shape))
        if weights is not None:
            _weights = reshape_img(weights, _img.shape, truncate=truncate)
        if sky is not None:
            _sky = reshape_img(sky, _img.shape, truncate=truncate)
        B,Ny,Nx = _img.shape
    else:
        _img = img
        _weights = weights
        _sky = sky

    if sky is None:
        Y = _img.reshape(B,Ny*Nx)
    else:
        Y = (_img-_sky).reshape(B,Ny*Nx)
    if weights is None:
        W = Wmax = 1
    else:
        W = _weights.reshape(B,Ny*Nx)
        Wmax = np.max(W)

    if psf is None:
        P_ = psf
    else:
        P_ = adapt_PSF(psf, B, (Ny, Nx))
    logger.debug("Shape: {0}".format((Ny,Nx)))

    """
    T = Translation(_peaks, (N,M), B, P_, txy_diff, max_shift,
                    txy_thresh, fit_positions, txy_wait, txy_skip, traceback)
    """

    # construct Blender from sources
    blender = Blender(sources)
    # assemble A and S matrix from source sed and morph (temporary!)
    A = np.empty((B, blender.K))
    S = np.empty((blender.K, Nx*Ny))
    for k in range(blender.K):
        m,l = blender.source_of[k]
        A[:,k] = blender.sources[m].sed[:,l]
        S[k,:] = blender.sources[m].morph[l].flatten()

    # define objective function
    if update_order is None:
        update_order = range(2)
    f = partial(blender.prox_likelihood, Y=Y, W=W, update_order=update_order)
    steps_f = Steps_AS(Wmax=Wmax, slack=slack, update_order=update_order)
    proxs_g = blender.proxs_g
    Ls = blender.Ls

    # run the NMF with those constraints
    Xs = [A, S]
    res = proxmin.algorithms.bsdmm(Xs, f, steps_f, proxs_g, steps_g=steps_g, Ls=Ls, update_order=update_order,
                                  steps_g_update=steps_g_update, max_iter=max_iter, e_rel=e_rel, e_abs=e_abs,
                                  accelerated=True, traceback=traceback)

    if not traceback:
        # A, S = res
        return blender
    else:
        [A, S], tr = res
        return blender, tr

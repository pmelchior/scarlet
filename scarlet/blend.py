from __future__ import print_function, division
import numpy as np
from functools import partial

import proxmin
from proxmin.nmf import Steps_AS

import logging
logger = logging.getLogger("scarlet")

# declare special exception for resizing events
class ScarletResizeException(Exception):
    pass

class Blend(object):
    """The blended scene as interpreted by the deblender.
    """
    def __init__(self, sources, img, weights=None, sky=None, init_sources=True):
        assert len(sources)
        # store all source and make search structures
        self._register_sources(sources)
        self.M = len(self.sources)
        self.B = self.sources[0].B

        # source refinement parameters
        self.refine_wait = 10
        self.refine_skip = 10
        self.center_min_dist = 1e-3
        self.edge_flux_thresh = 1.
        self.update_order = [1,0]
        self.slack = 0.9

        # set up data structures
        self.set_data(img, weights=weights, sky=sky)

        if init_sources:
            self.init_sources()

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

    # _prox_g need to be properties so that they can react to runtime changes
    # in the sources
    @property
    def _proxs_g(self):
        return [source.proxs_g[0] for source in self.sources] + [source.proxs_g[1] for source in self.sources]

    @property
    def _Ls(self):
        return [source.Ls[0] for source in self.sources] + [source.Ls[1] for source in self.sources]

    def fit(self, e_rel=1e-2, max_iter=200):

        # set sparsity cutoff for morph based on the error level
        # TODO: Computation only correct if psf=None!
        self.e_rel = [e_rel] * 2*self.K
        self.e_abs = [e_rel / self.B] * self.K + [0.] * self.K
        self.update_source_sparsity()

        # perform up to max_iter steps
        self.it = 0
        self._model_it = -1
        return self._step(steps=max_iter)

    def _step(self, steps=1, max_iter=None):
        if max_iter is None:
            max_iter = steps

        # collect all SEDs and morphologies, plus associated errors
        XA = []
        XS = []
        for k in range(self.K):
            m,l = self.source_of(k)
            XA.append(self.sources[m].sed[l])
            XS.append(self.sources[m].morph[l])
        X = XA + XS

        # update_order for bSDMM is over *all* components
        if self.update_order[0] == 0:
            _update_order = range(2*self.K)
        else:
            _update_order = range(self.K,2*self.K) + range(self.K)

        # run bSDMM on all SEDs and morphologies
        steps_g = None
        steps_g_update = 'steps_f'
        traceback = False
        accelerated = True
        try:
            res = proxmin.algorithms.bsdmm(X, self._prox_f, self._steps_f, self._proxs_g, steps_g=steps_g, Ls=self._Ls, update_order=_update_order, steps_g_update=steps_g_update, max_iter=steps, e_rel=self.e_rel, e_abs=self.e_abs, accelerated=accelerated, traceback=traceback)
        except ScarletResizeException:
            steps = max_iter - self.it
            self._step(steps=steps, max_iter=max_iter)
        return self

    def set_data(self, img, weights=None, sky=None):

        if sky is None:
            self._ = img
        else:
            self._img = img-sky

        self._set_weights(weights)
        WAmax = np.max(self._weights[1])
        WSmax = np.max(self._weights[2])
        self._stepAS = Steps_AS(WAmax=WAmax, WSmax=WSmax, slack=self.slack, update_order=self.update_order)
        self.step_AS = [None] * 2

    def init_sources(self):
        for m in range(self.M):
            self.sources[m].init_source(self._img, weights=self._weights[0])

    def get_model(self, m=None, combine=True, combine_source_components=True):
        """Compute the current model for the entire image
        """
        if m is not None:
            source = self.sources[m]
            model = source.get_model(combine=combine_source_components)
            model_slice = source.get_slice_for(self._img.shape)

            if combine_source_components:
                model_img = np.zeros(self._img.shape)
                model_img[source.bb] = model[model_slice]
            else:
                # keep record of flux at edge of the source model
                self._set_edge_flux(m, model)

                model_img = np.zeros((source.K,) + (self._img.shape))
                for k in range(source.K):
                    model_img[k][source.bb] = model[k][model_slice]
            return model_img

        # for all sources
        if combine:
            return np.sum([self.get_model(m=m, combine_source_components=True) for m in range(self.M)], axis=0)
        else:
            models = [self.get_model(m=m, combine_source_components=combine_source_components) for m in range(self.M)]
            return np.vstack(models)

    def _register_sources(self, sources):
        self.sources = sources # do not copy!
        self.K =  sum([source.K for source in self.sources])
        self.psf_per_band = not hasattr(sources[0].Gamma, 'shape')

        # lookup of source/component tuple given component number k
        self._source_of = []
        for m in range(len(sources)):
            for l in range(self.sources[m].K):
                self._source_of.append((m,l))

    def _set_weights(self, weights):
        if weights is None:
            self._weights = [1,1,1]
            self._noise_eff = [[0,] * self.B] * self.M
        else:
            self._weights = [weights, None, None] # [W, WA, WS]

            # store local noise level for each source in each bands
            from .utils import invert_with_zeros
            self._noise_eff = []
            for m in range(self.M):
                w = weights[self.sources[m].bb].reshape(self.B ,-1)
                std = invert_with_zeros(w)
                mask = (std == -1)
                m = np.ma.array(std, mask=mask)
                self._noise_eff.append(np.sqrt(np.median(m, axis=1).data))

            # for S update: normalize the per-pixel variation
            # i.e. in every pixel: utilize the bands with large weights
            # CAVEAT: need to filter out pixels that are saturated in every band
            norm_pixel = np.median(weights, axis=0)
            mask = norm_pixel > 0
            self._weights[2] = weights.copy()
            self._weights[2][:,mask] /= norm_pixel[mask]

            # reverse is true for A update: for each band, use the pixels that
            # have the largest weights
            norm_band = np.median(weights, axis=(1,2))
            # CAVEAT: some regions may have one band missing completely
            mask = norm_band > 0
            self._weights[1] = weights.copy()
            self._weights[1][mask] /= norm_band[mask,None,None]
            # CAVEAT: mask all pixels in which at least one band has W=0
            # these are likely saturated and their colors have large weights
            # but are incorrect due to missing bands
            mask = ~np.all(weights>0, axis=0)
            # and mask all bands for that pixel:
            # when estimating A do not use (partially) saturated pixels
            self._weights[1][:,mask] = 0

    def _compute_model(self):
        # make sure model at current iteration is computed when needed
        # irrespective of function that needs it
        if self._model_it < self.it:
            self._models = self.get_model(combine=False, combine_source_components=False) # model each each component over image
            self._model = np.sum(self._models, axis=0)
            self._model_it = self.it

    def _prox_f(self, X, step, Xs=None, j=None):

        # which update to do now
        AorS = j//self.K
        k = j%self.K

        # computing likelihood gradients for S and A:
        # build model only once per iteration
        if k == 0:
            if AorS == self.update_order[0]:
                self.it += 1

                # refine sources
                if self.it >= self.refine_wait and self.it % self.refine_skip == 0:
                    resized = self.resize_sources()
                    self.recenter_sources()
                    self.update_source_sparsity()
                    if resized:
                        raise ScarletResizeException()

                self._compute_model()

            # compute weighted residuals
            self._diff = self._weights[AorS + 1]*(self._model-self._img)

        # A update
        if AorS == 0:
            m,l = self.source_of(k)
            if not self.sources[m].fix_sed[l]:
                # gradient of likelihood wrt A: nominally np.dot(diff, S^T)
                # but with PSF convolution, S_ij -> sum_q Gamma_bqi S_qj
                # however, that's exactly the operation done for models[k]
                # caveat: the model is SED * convolved model -> need to divide
                grad = np.einsum('...ij,...ij', self._diff, self._models[k] / self.sources[m].sed[l].T[:,None,None])

                # apply per component prox projection and save in source
                self.sources[m].sed[l] =  self.sources[m].prox_sed[l](X - step*grad, step)
            return self.sources[m].sed[l]

        # S update
        elif AorS == 1:
            m,l = self.source_of(k)
            if not self.sources[m].fix_morph[l]:
                # gradient of likelihood wrt S: nominally np.dot(A^T,diff)
                # but again: with convolution, it's more complicated

                # first create diff image in frame of source k
                slice_m = self.sources[m].get_slice_for(self._img.shape)
                diff_k = np.zeros(self.sources[m].shape)
                diff_k[slice_m] = self._diff[self.sources[m].bb]

                # now a gradient vector and a mask of pixel with updates
                grad = np.zeros_like(X)
                if not self.psf_per_band:
                    for b in range(self.B):
                        grad += self.sources[m].sed[l,b]*self.sources[m].Gamma.T.dot(diff_k[b].flatten())
                else:
                    for b in range(self.B):
                        grad += self.sources[m].sed[l,b]*self.sources[m].Gamma[b].T.dot(diff_k[b].flatten())

                # apply per component prox projection and save in source
                self.sources[m].morph[l] = self.sources[m].prox_morph[l](X - step*grad, step)
            return self.sources[m].morph[l]
        else:
            raise ValueError("Expected index j in [0,%d]" % (2*self.K))

    def _steps_f(self, j, Xs):
        # which update to do now
        AorS = j//self.K
        k = j%self.K

        # computing likelihood gradients for S and A: only once per iteration
        if AorS == self.update_order[0] and k==0:
            self._compute_model()

            # build temporary A,S matrices
            B, Ny, Nx = self._img.shape
            A = np.empty((self.B,self.K))
            for k_ in range(self.K):
                m,l = self._source_of[k_]
                A[:,k_] = self.sources[m].sed[l]
            if not self.psf_per_band:
                # model[b] is simple SED[b] * S, need to divide by SED
                b = 0
                S = self._models[:,b,:,:].reshape((self.K, Ny*Nx)) / A.T[:,b][:,None]
            else:
                # TODO: replace this with the current models, for each band
                # i.e. one S per band -> step_size per band
                raise NotImplementedError
            self.step_AS[0] = self._stepAS(0, [A, S])
            self.step_AS[1] = self._stepAS(1, [A, S])
        return self.step_AS[AorS]

    def recenter_sources(self):
        # residuals weighted with full/original weight matrix
        y = self._weights[0]*(self._model-self._img)
        for m in range(self.M):
            if self.sources[m].shift_center:
                source = self.sources[m]
                bb_m = source.bb
                diff_x,diff_y = self._get_shift_differential(m)
                diff_x[:,:,-1] = 0
                diff_y[:,-1,:] = 0
                # least squares for the shifts given the model residuals
                MT = np.vstack([diff_x.flatten(), diff_y.flatten()])
                if not hasattr(self._weights[0],'shape'): # no/flat weights
                    ddx,ddy = np.dot(np.dot(np.linalg.inv(np.dot(MT, MT.T)), MT), y[bb_m].flatten())
                else:
                    w = self._weights[0][bb_m].flatten()[:,None]
                    ddx,ddy = np.dot(np.dot(np.linalg.inv(np.dot(MT, MT.T*w)), MT), y[bb_m].flatten())
                if ddx**2 + ddy**2 > self.center_min_dist**2:
                    center = source.center + (ddy, ddx)
                    source.set_center(center)
                    logger.info("shifting source %d by (%.3f/%.3f) to (%.3f/%.3f)" % (m, ddy, ddx, source.center[0], source.center[1]))

    def _get_shift_differential(self, m):
        # compute (model - dxy*shifted_model)/dxy for first-order derivative
        source = self.sources[m]
        slice_m = source.get_slice_for(self._img.shape)
        k = self.component_of(m, 0)
        model_m = self._models[k][self.sources[m].bb]
        # in self._models, per-source components aren't combined,
        # need to combine here
        for k in range(1,source.K):
            model_m += self._models[k][self.sources[m].bb]

        # get Gamma matrices of source m with additional shift
        offset = source.shift_center
        dx = source.center - source.center_int
        pos_x = dx + (0, offset)
        pos_y = dx + (offset, 0)
        dGamma_x = source._gammaOp(pos_x)
        dGamma_y = source._gammaOp(pos_y)
        diff_img = [source.get_model(combine=True, Gamma=dGamma_x), source.get_model(combine=True, Gamma=dGamma_y)]
        diff_img[0] = (model_m-diff_img[0][slice_m])/source.shift_center
        diff_img[1] = (model_m-diff_img[1][slice_m])/source.shift_center
        return diff_img

    def _set_edge_flux(self, m, model):
        try:
            self._edge_flux
        except AttributeError:
            self._edge_flux = np.zeros((self.M, 4, self.B))

        # top, right, bottom, left
        self._edge_flux[m,0,:] = np.abs(model[:,:,-1,:]).sum(axis=0).max(axis=1)
        self._edge_flux[m,1,:] = np.abs(model[:,:,:,-1]).sum(axis=0).max(axis=1)
        self._edge_flux[m,2,:] = np.abs(model[:,:,0,:]).sum(axis=0).max(axis=1)
        self._edge_flux[m,3,:] = np.abs(model[:,:,:,0]).sum(axis=0).max(axis=1)

    def resize_sources(self):
        resized = False
        for m in range(self.M):
            if not self.sources[m].fix_frame:
                size = [self.sources[m].Ny, self.sources[m].Nx]
                increase = [max(0.25*s, 10) for s in size]

                # check if max flux along edge in band b < avg noise level along edge in b
                at_edge = (self._edge_flux[m] > self._noise_eff[m]*self.edge_flux_thresh)
                # TODO: without symmetry constraints, the four edges of the box
                # should be allowed to resize independently
                if at_edge[0].any() or at_edge[2].any():
                    size[0] += increase[0]
                if at_edge[1].any() or at_edge[3].any():
                    size[1] += increase[1]
                if at_edge.any():
                    logger.info("resizing source %d to (%d/%d)" % (m, size[0], size[1]))
                    self.sources[m].resize(size)
                    resized = True
        return resized

    def update_source_sparsity(self):
        if self._weights[0] is not None:
            for m in range(self.M):
                morph_std = self.sources[m].set_morph_sparsity(self._weights[0])
                for l in range(self.sources[m].K):
                    k = self.K + self.component_of(m,l)
                    self.e_abs[k] = self.e_rel[k] * morph_std[l]

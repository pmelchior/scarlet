from __future__ import print_function, division
import numpy as np
from functools import partial

import proxmin

import logging
logger = logging.getLogger("scarlet.blend")

# declare special exception for resizing events
class ScarletRestartException(Exception):
    pass

class Blend(object):
    """The blended scene as interpreted by the deblender.
    """
    def __init__(self, sources, img, weights=None, sky=None, bg_rms=None, init_sources=True):
        """Constructor

        Parameters
        ----------
        sources: list of `~scarlet.Source` objects
            Individual sources in the blend.
            The scarlet deblender requires the user to detect sources
            and configure their constraints before initializing a blend
        img: array-like
            (Bands, Height, Width) image array containing the data.
        weights: `~numpy.array`, default=`None`
            Array (Bands, Height, Width) of weights to use for each pixel in each band.
            .. warning::
                
                The weights should be flat in each band
                (except for zero weighted masked out pixels).
                Our analysis has shown that when the weights are widely varying
                over an image in a single band, SED colors (and hence morphologies)
                are poorly modeled.
        sky: array-like, default=`None`
            Array (Bands, Height, Width) of the estimated sky level in each image, in each band.
            This is subtracted from `img`, so if `img` has already happend then this
            should be `None`.
        bg_rms: array-like, default=`None`
            Array of length `Bands` that contains the RMS in the image for each band
        init_sources: bool
            Whether or not the sources need to be initialized.
        """
        assert len(sources)
        # store all source and make search structures
        self._register_sources(sources)
        self.M = len(self.sources)
        self.B = self.sources[0].B

        # source refinement parameters
        self.refine_skip = 10
        self.center_min_dist = 1e-3
        self.edge_flux_thresh = 1.
        self.update_order = [1,0]
        self.slack = 0.2
        self.e_rel = 1e-2

        # set up data structures
        self.set_data(img, weights=weights, sky=sky, bg_rms=bg_rms)

        if init_sources:
            self.init_sources()

    def source_of(self, k):
        """Get the indices of source k

        Each of `m` `~scarlet.source.Source`s in the model can have multiple
        components, but the main algorithm recognizes each component as a single
        source, for `k=Sum_m(m_l)` total sources.
        This method returns the tuple of indices `(m,l)` for source `k`.
        """
        return self._source_of[k]

    def component_of(self, m, l):
        """Search for k index of source m, component l

        This is the inverse of source_of, and returns `k` for the given
        pair `(m,l)`.
        """
        for k in range(self.K):
            if self._source_of[k] == (m,l):
                return k
        raise IndexError

    def __len__(self):
        """Number of `~scarlet.source.Source`s (not including components)
        """
        return self.M

    @property
    def _proxs_g(self):
        """Proximal operator for each source in the dual update

        Each source can have it's own value for prox g, and because the size and shape of
        a source can change at runtime, _proxs_g is property called by the bSDMM algorithm.
        These functions are created in `~scarlet.source.Source.set_constraints`.

        This is the proximal operator that is applied to the `A` or `S` update
        of the dual variable.
        See Algorithm 3, line 12 in Moolekamp and Melchior 2017
        (https://arxiv.org/pdf/1708.09066.pdf) for more.
        """
        return [source.proxs_g[0] for source in self.sources] + [source.proxs_g[1] for source in self.sources]

    @property
    def _Ls(self):
        """Linear operator for each source in the dual update

        See section 2.3 in Moolekamp and Melchior 2017
        (https://arxiv.org/pdf/1708.09066.pdf) for more.
        """
        return [source.Ls[0] for source in self.sources] + [source.Ls[1] for source in self.sources]

    def fit(self, steps=200, e_rel=None, max_iter=None, traceback=False):
        """Fit the model for each source to the data

        Parameters
        ----------
        steps: int
            Maximum number of iterations, even if the algorithm doesn't converge.
            See `max_iter` description for details.
        e_rel: float, default=`None`
            Relative error for convergence. Of `e_rel` is `None` then
            relative error isn't used as a convergence check
        max_iter: int, default=`None`
            Maximum number of iterations, including restarts.
            The difference between `max_iter` and `steps` is that the
            deblender might throw a `ScarletRestartException`, which
            restarts the deblender. In this case the total number of
            iterations will still not exceed `max_iter`.
            If `max_iter` is `None` then `max_iter=`steps`.
        traceback: bool, default=False
            Whether or not to store the history of a traceback.
            .. warning::
        
               This can be very costly in terms of memory. It is highly recommended to leave
               this off unless you really know what you're doing!
        
        Returns
        -------
        self: `~scarlet.blends.Blend`
            This object, which contains the results of the deblender.
        """
        try:
            self.it
        except AttributeError:
            self.it = 0
            self._model_it = -1
            # Caches for 1/Lipschitz for A and S
            self._cbAS = [proxmin.utils.ApproximateCache(self._one_over_lipschitz, slack=self.slack),
                          proxmin.utils.ApproximateCache(self._one_over_lipschitz, slack=self.slack)]

        # only needed if the restart exception has been thrown
        if max_iter is None:
            max_iter = steps

        # define error limits
        if e_rel is not None:
            self.e_rel = e_rel
        self._set_error_limits()

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
        accelerated = True
        try:
            res = proxmin.algorithms.bsdmm(X, self._prox_f, self._steps_f, self._proxs_g, steps_g=steps_g,
                Ls=self._Ls, update_order=_update_order, steps_g_update=steps_g_update, max_iter=steps,
                e_rel=self._e_rel, e_abs=self._e_abs, accelerated=accelerated, traceback=traceback)
        except ScarletRestartException:
            steps = max_iter - self.it
            self.fit(steps=steps, max_iter=max_iter)
        return self

    def set_data(self, img, weights=None, sky=None, bg_rms=None):
        """Initialize the data
        
        Subtract the sky from the image, initialize the weights and background,
        and create the full Gamma matrix
        """
        if sky is None:
            self._ = img
        else:
            self._img = img-sky
        if bg_rms is None:
            self._bg_rms = np.zeros(self.B)
        else:
            assert len(bg_rms) == self.B
            self._bg_rms = np.array(bg_rms)
        self._set_weights(weights)

        if self.use_psf:
            from .transformations import GammaOp
            pos = (0,0)
            self._Gamma_full = [ source._gammaOp(pos, self._img.shape, offset_int=source.center_int)
                                    for source in self.sources]

    def init_sources(self):
        """Initialize the model for each source
        """
        for m in range(self.M):
            self.sources[m].init_source(self._img, weights=self._weights)

    def get_model(self, m=None, combine=True, combine_source_components=True, use_sed=True):
        """Compute the current model for the entire image
        """
        if m is not None:
            source = self.sources[m]
            model = source.get_model(combine=combine_source_components, use_sed=use_sed)
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
            return np.sum([self.get_model(m=m, combine_source_components=True, use_sed=use_sed)
                                for m in range(self.M)], axis=0)
        else:
            models = [self.get_model(m=m, combine_source_components=combine_source_components,
                                     use_sed=use_sed) for m in range(self.M)]
            return np.vstack(models)

    def _register_sources(self, sources):
        """Unpack the components to register them as individual sources
        """
        self.sources = sources # do not copy!
        self.K =  sum([source.K for source in self.sources])
        have_psf = [source.has_psf for source in self.sources]
        self.use_psf = any(have_psf)
        assert any(have_psf) == all(have_psf)

        # lookup of source/component tuple given component number k
        self._source_of = []
        for m in range(len(sources)):
            for l in range(self.sources[m].K):
                self._source_of.append((m,l))

    def _set_weights(self, weights):
        """Set the weights and correlation matrix `_Sigma_1`

        Parameters
        ----------
        weights: array-like
            Array (Band, Height, Width) of weights for each image, in each band

        Returns
        -------
        None, but sets `self._Sigma_1` and `self._weights`.
        """
        import scipy.sparse
        if weights is None:
            self._weights = 1
            B, Ny, Nx = self._img.shape
            self._Sigma_1 = [scipy.sparse.identity(B*Ny*Nx)] * 2
        else:
            self._weights = weights
            self._Sigma_1 = [None] * 2

            # for S update: normalize the per-pixel variation
            # i.e. in every pixel: utilize the bands with large weights
            # CAVEAT: need to filter out pixels that are saturated in every band
            norm_pixel = np.median(weights, axis=0)
            mask = norm_pixel > 0
            _weights = weights.copy()
            _weights[:, mask] /= norm_pixel[mask]
            self._Sigma_1[1] = scipy.sparse.diags(_weights.flatten())

            # reverse is true for A update: for each band, use the pixels that
            # have the largest weights
            norm_band = np.median(weights, axis=(1,2))
            # CAVEAT: some regions may have one band missing completely
            mask = norm_band > 0
            _weights[:] = weights
            _weights[mask] /= norm_band[mask,None,None]
            # CAVEAT: mask all pixels in which at least one band has W=0
            # these are likely saturated and their colors have large weights
            # but are incorrect due to missing bands
            mask = ~np.all(weights>0, axis=0)
            # and mask all bands for that pixel:
            # when estimating A do not use (partially) saturated pixels
            _weights[:,mask] = 0
            self._Sigma_1[0] = scipy.sparse.diags(_weights.flatten())

    def _compute_model(self):
        """Build the entire model

        Calculate the full model once per iteration.
        This creates `self._models`, the morphological model of each source
        projected onto the full image, and `self._model`, which convolves
        those models with the SED for each source and adds them into a
        single model.
        """
        # make sure model at current iteration is computed when needed
        # irrespective of function that needs it
        if self._model_it < self.it:
            # model each each component over image
            # do not use SED, so that it can be reused later
            self._models = self.get_model(combine=False, combine_source_components=False, use_sed=False)
            self._A = np.empty((self.B,self.K))
            for k_ in range(self.K):
                m,l = self._source_of[k_]
                self._A[:,k_] = self.sources[m].sed[l]
            self._model = np.sum(self._A.T[:,:,None,None] * self._models, axis=0)
            self._model_it = self.it

    def _prox_f(self, X, step, Xs=None, j=None):
        """Proximal operator for the X update

        To save processing time, the model is calculated when the first source
        is updated and all subsequent prox_f calculations (in the same iteration)
        use the same cached model.

        This is the proximal operator that is applied to the `A` or `S` update.
        See Algorithm 3, line 10 in Moolekamp and Melchior 2017
        (https://arxiv.org/pdf/1708.09066.pdf) for more.

        Parameters
        ----------
        X: array-like
            Either `A` (sed) or `S` (morphology) matrix of the model
        step: float
            Step size calculated using `self._steps_f`.
        Xs: array-like
            List of all matrices (in this case [A,S]) that are modeled
            using the bSDMM algorithm.
        j: int
            Index of the current matrix in `Xs`.
            So `j=0` for `A` and `j=1` for `S`.

        Returns
        -------
        result: `~numpy.array`
            Array of `X` after `_prox_f` has been applied
        """

        # which update to do now
        block = j//self.K
        k = j%self.K

        # computing likelihood gradients for S and A:
        # build model only once per iteration
        if k == 0:
            if block == self.update_order[0]:
                self.it += 1

                # refine sources
                if self.it > 0 and self.it % self.refine_skip == 0:
                    resized = self.resize_sources()
                    self.recenter_sources()
                    self.adjust_absolute_error()
                    if resized:
                        raise ScarletRestartException()

                self._compute_model()

            # compute weighted residuals
            B, Ny, Nx = self._img.shape
            self._diff = self._Sigma_1[block].dot((self._model-self._img).flatten()).reshape(B, Ny, Nx)

        # A update
        if block == 0:
            m,l = self.source_of(k)
            if not self.sources[m].fix_sed[l]:
                # gradient of likelihood wrt A: nominally np.dot(diff, S^T)
                # but with PSF convolution, S_ij -> sum_q Gamma_bqi S_qj
                # however, that's exactly the operation done for models[k]
                grad = np.einsum('...ij,...ij', self._diff, self._models[k])

                # apply per component prox projection and save in source
                self.sources[m].sed[l] =  self.sources[m].prox_sed[l](X - step*grad, step)
            return self.sources[m].sed[l]

        # S update
        elif block == 1:
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
                if not self.use_psf:
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

    def _one_over_lipschitz(self, block):
        """Calculate 1/Lipschitz constant for A and S
        """
        import scipy.sparse
        import scipy.sparse.linalg

        B, Ny, Nx = self._img.shape
        if block == 0: # A
            # model[b] is S in band b, but need to go to frame in which
            # A and S are serialized. Then Lischitz constant of A:
            # LA = ||Sigma_a||_s with
            # Sigma_a = ((PS)^T Sigma_pixel^-1 PS)^-1
            # in the frame where A is a vector of length K*B
            # NOTE: convolution implicitly treated in the models
            PS = scipy.sparse.block_diag([self._models[:,b,:,:].reshape((self.K, Ny*Nx)).T
                                            for b in range(self.B)])
            # Lipschitz constant for grad_A = || S Sigma_1 S.T||_s
            SSigma_1S = PS.T.dot(self._Sigma_1[0].dot(PS))
            LA = np.real(scipy.sparse.linalg.eigs(SSigma_1S, k=1, return_eigenvectors=False)[0])
            return 1./LA

        if block == 1: # S
            if not self.use_psf:
                # Lipschitz constant for grad_S = || A.T Sigma_1 A||_s
                # need to go to frame in which A and S are serialized
                PA = scipy.sparse.bmat([[scipy.sparse.identity(Ny*Nx) * self._A[b,k]
                                        for k in range(self.K)]
                                        for b in range(self.B)])
            else:
                # similar calculation for S: ||Sigma_s||_s with
                # Sigma_s = ((PA)^T Sigma_pixel^-1 PA)^-1
                # in the frame where S is a vector of length N*K
                PA = scipy.sparse.bmat([[self._A[b,k] * self._Gamma_full[self.source_of(k)[0]][b]
                                        for k in range(self.K)] for b in range(self.B)])
            ASigma_1A = PA.T.dot(self._Sigma_1[1].dot(PA))
            LS = np.real(scipy.sparse.linalg.eigs(ASigma_1A, k=1, return_eigenvectors=False)[0])
            return 1./LS
        raise NotImplementedError("block {0} not < 2!".format(block))

    def _steps_f(self, j, Xs):
        """Calculate the current step-size for prox f

        This method used the cached 1/Lipschitz value when possible and
        uses it to calculate the step sizes for both A and S only once per
        iteration.

        Parameters
        ----------
        Xs: array-like
            List of all matrices (in this case [A,S]) that are modeled
            using the bSDMM algorithm.
        j: int
            Index of the current matrix in `Xs`.
            So `j=0` for `A` and `j=1` for `S`.

        Returns
        -------
        step: float
            Step size for the current block (either A or S).
        """
        # which update to do now
        block = j//self.K
        k = j%self.K

        # computing likelihood gradients for S and A: only once per iteration
        # equal to spectral norm of covariance matrix of A or S
        if block == self.update_order[0] and k==0:
            self._compute_model()
            # compute step sizes and save to reuse for every component of A or S
            # _cbAS is the cached values 1/Lipschitz for A and S
            self._stepAS = [self._cbAS[block](block) for block in [0,1]]

        return self._stepAS[block]

    def recenter_sources(self):
        """Shift center position of sources to better match the data
        """
        # residuals weighted with full/original weight matrix
        y = self._weights*(self._model-self._img)
        for m in range(self.M):
            if self.sources[m].shift_center:
                source = self.sources[m]
                bb_m = source.bb
                diff_x,diff_y = self._get_shift_differential(m)
                diff_x[:,:,-1] = 0
                diff_y[:,-1,:] = 0
                # least squares for the shifts given the model residuals
                MT = np.vstack([diff_x.flatten(), diff_y.flatten()])
                if not hasattr(self._weights,'shape'): # no/flat weights
                    ddx,ddy = np.dot(np.dot(np.linalg.inv(np.dot(MT, MT.T)), MT), y[bb_m].flatten())
                else:
                    w = self._weights[bb_m].flatten()[:,None]
                    ddx,ddy = np.dot(np.dot(np.linalg.inv(np.dot(MT, MT.T*w)), MT), y[bb_m].flatten())
                if ddx**2 + ddy**2 > self.center_min_dist**2:
                    center = source.center + (ddy, ddx)
                    source.set_center(center)
                    msg = "shifting source {0} by ({1}/2}) to (3}/4})"
                    logger.info(msg.format(m, ddy, ddx, source.center[0], source.center[1]))

    def _get_shift_differential(self, m):
        """Calculate the difference image used ot fit positions

        Parameters
        ----------
        m: int
            Index of the source in `Blend.sources`

        Returns
        -------
        diff_img: `~numpy.array`
            Difference image in each band used to fit the position
        """
        # compute (model - dxy*shifted_model)/dxy for first-order derivative
        source = self.sources[m]
        slice_m = source.get_slice_for(self._img.shape)
        k = self.component_of(m, 0)
        model_m = self._models[k][self.sources[m].bb].copy()
        # in self._models, per-source components aren't combined,
        # need to combine here
        for k in range(1,source.K):
            model_m += self._models[k][self.sources[m].bb]

        # get Gamma matrices of source m with additional shift
        offset = source.shift_center
        dx = source.center - source.center_int
        pos_x = dx + (0, offset)
        pos_y = dx + (offset, 0)
        dGamma_x = source._gammaOp(pos_x, source.shape)
        dGamma_y = source._gammaOp(pos_y, source.shape)
        diff_img = [source.get_model(combine=True, Gamma=dGamma_x),
                    source.get_model(combine=True, Gamma=dGamma_y)]
        diff_img[0] = (model_m-diff_img[0][slice_m])/source.shift_center
        diff_img[1] = (model_m-diff_img[1][slice_m])/source.shift_center
        return diff_img

    def _set_edge_flux(self, m, model):
        """Keep track of the flux at the edge of the model

        Parameters
        ----------
        m: int
            Index of the source
        model: `~numpy.array`
            (Band,Height,Width) array of the model.

        Returns
        -------
        None, but sets `self._edge_flux`
        """
        try:
            self._edge_flux
        except AttributeError:
            self._edge_flux = np.zeros((self.M, 4, self.B))

        # top, right, bottom, left
        self._edge_flux[m,0,:] = np.abs(model[:,:,-1,:]).sum(axis=0).mean(axis=1)
        self._edge_flux[m,1,:] = np.abs(model[:,:,:,-1]).sum(axis=0).mean(axis=1)
        self._edge_flux[m,2,:] = np.abs(model[:,:,0,:]).sum(axis=0).mean(axis=1)
        self._edge_flux[m,3,:] = np.abs(model[:,:,:,0]).sum(axis=0).mean(axis=1)

    def resize_sources(self):
        """Resize frames for sources (if necessary)

        If there is flux at the edges of the frame for a given source,
        increase it's frame size.
        """
        resized = False
        for m in range(self.M):
            if not self.sources[m].fix_frame:
                size = [self.sources[m].Ny, self.sources[m].Nx]
                increase = [max(0.25*s, 10) for s in size]

                # check if max flux along edge in band b < avg noise level along edge in b
                at_edge = (self._edge_flux[m] > self._bg_rms*self.edge_flux_thresh)
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

    def _absolute_morph_error(self):
        """Get the absolute morphology error
        """
        m = 0 # needed otherwise python 2 complains about "local variable 'm' referenced before assignment"
        return [self.e_rel * self.sources[m].morph[l].mean()
                for l in range(self.sources[m].K) for m in range(self.M)]

    def _set_error_limits(self):
        """Set the error limits for each source
        """
        self._e_rel = [self.e_rel] * 2 * self.K
        # absolute errors: e_rel * mean signal, will be updated later
        self._e_abs = [self.e_rel / self.B] * self.K
        self._e_abs += self._absolute_morph_error()

    def adjust_absolute_error(self):
        """Adjust the absolute error for each source
        """
        self._e_abs[self.K:] = self._absolute_morph_error()

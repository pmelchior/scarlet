from __future__ import print_function, division
import numpy as np
from functools import partial

import proxmin
from .config import Config

import logging
logger = logging.getLogger("scarlet.blend")

# declare special exception for resizing events
class ScarletRestartException(Exception):
    """Restart the bSDMM algorithm

    If the size or number of components changes during the fit,
    the `proxmin.algorithms.bsdmm` function will fail.
    The `ScarletRestartException` is thrown by `Blend.fit`
    when a breaking change is made to one of the components so
    that `Blend.fit` can catch the exception and restart
    the bSDMM algorithm from the last iteration.
    """
    pass

class Blend(object):
    """The blended scene as interpreted by the deblender.
    """
    def __init__(self, sources, img, weights=None, bg_rms=None, config=None):
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
        bg_rms: array-like, default=`None`
            Array of length `Bands` that contains the sky background RMS in the image for each band,
            used primarily as a minimum flux threshold to set the box size for each source during resizing.
            If `bg_rms` is `None` then a zero valued array is used as the minimum flux threshold.
        config: `~scarlet.Config` instance, default=`None`
            Special configuration to overwrite default optimization parameters
        """
        assert len(sources)
        # store all source and make search structures
        self._register_sources(sources)
        self.M = len(self.sources)
        self.B = self.sources[0].B

        if config is None:
            config = Config()
        self.config = config

        # set up data structures
        self.set_data(img, weights=weights, bg_rms=bg_rms)

    def source_of(self, k):
        """Get the indices of model component k.

        Each of `m` `~scarlet.source.Source`s in the model can have multiple
        components, but the main algorithm recognizes each component as a single
        source, for `k=Sum_m(m_l)` total sources.
        This method returns the tuple of indices `(m,l)` for source `k`.
        """
        return self._source_of[k]

    def component_of(self, m, l):
        """Search for k index of source m, component l.

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

    def save(self, filename):
        # store S and A for each source
        _dict = {}
        for m in range(self.M):
            _dict["sed_%d" % m] = self.sources[m].sed
            _dict["morph_%d" % m] = self.sources[m].morph
        np.savez(filename, **_dict)

    def load(self, filename):
        _dict = np.load(filename)
        for m in range(self.M):
            self.sources[m].sed = _dict["sed_%d" % m]
            self.sources[m].morph = _dict["morph_%d" % m]

    @property
    def _proxs_g(self):
        """Proximal operator for each source in the dual update

        Each source can have its own value for prox g, and because the size and shape of
        a source can change at runtime, _proxs_g is property called by the bSDMM algorithm.
        These functions are created in `~scarlet.source.Source.set_constraints`.

        This is the proximal operator that is applied to the `A` or `S` update
        of the dual variable.
        See Algorithm 3, line 12 in Moolekamp and Melchior 2017
        (https://arxiv.org/pdf/1708.09066.pdf) for more.
        """
        proxs_g_sed = []
        proxs_g_morph = []
        for k in range(self.K):
            m,l = self.source_of(k)
            proxs_g_sed.append(self.sources[m].constraints[l].prox_g_sed)
            proxs_g_morph.append(self.sources[m].constraints[l].prox_g_morph)
        return proxs_g_sed + proxs_g_morph

    @property
    def _Ls(self):
        """Linear operator for each source in the dual update

        See section 2.3 in Moolekamp and Melchior 2017
        (https://arxiv.org/pdf/1708.09066.pdf) for details.
        """
        Ls_sed = []
        Ls_morph = []
        for k in range(self.K):
            m,l = self.source_of(k)
            Ls_sed.append(self.sources[m].constraints[l].L_sed)
            Ls_morph.append(self.sources[m].constraints[l].L_morph)
        return Ls_sed + Ls_morph

    def fit(self, steps=200, e_rel=1e-2):
        """Fit the model for each source to the data

        Parameters
        ----------
        steps: int
            Maximum number of iterations if the algorithm doesn't converge.
        e_rel: float, default=`None`
            Relative error for convergence. If `e_rel` is `None`, the default
            `~scarlet.blend.Blend.e_rel` is used for convergence checks

        Returns
        -------
        self: `~scarlet.blend.Blend`
            This object, which contains the results of the deblender.
            For investigating the model components, use the source list specified
            at construction time or `~scarlet.blend.Blend.sources`, which is
            the internal reference to that list.
        """
        try:
            self.it # test of this is first time fit is called
        except AttributeError:
            self.it = 0
            self._model_it = -1
            # Caches for 1/Lipschitz for A and S
            self._cbAS = [proxmin.utils.ApproximateCache(self._one_over_lipschitz, slack=self.config.slack),
                          proxmin.utils.ApproximateCache(self._one_over_lipschitz, slack=self.config.slack)]

        if self.config.exact_lipschitz:
            # use full weight matrixes
            try:
                self._Sigma_1
            except AttributeError:
                B, Ny, Nx = self._img.shape
                import scipy.sparse
                if self._weights[0] is 1:
                    self._Sigma_1 = [scipy.sparse.identity(B*Ny*Nx)] * 2
                else:
                    self._Sigma_1 = [scipy.sparse.diags(w.flatten()) for w in self._weights]
            # use full-frame Gamma matrices
            if self.use_psf:
                pos = (0,0)
                self._Gamma_full = [ source._gamma(pos, self._img.shape, offset_int=source.center_int)
                                    for source in self.sources]

        # define error limits
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
        if self.config.update_order[0] == 0:
            _update_order = list(range(2*self.K))
        else:
            _update_order = list(range(self.K,2*self.K)) + list(range(self.K))

        # run bSDMM on all SEDs and morphologies
        steps_g = None
        steps_g_update = 'steps_f'
        max_iter = self.it + steps
        try:
            res = proxmin.algorithms.bsdmm(X, self._prox_f, self._steps_f, self._proxs_g, steps_g=steps_g,
                Ls=self._Ls, update_order=_update_order, steps_g_update=steps_g_update, max_iter=steps,
                e_rel=self._e_rel, e_abs=self._e_abs)
        except ScarletRestartException:
            if self.it < max_iter: # don't restart at last iteration
                steps = max_iter - self.it
                self.fit(steps=steps)
        return self

    def set_data(self, img, weights=None, bg_rms=None):
        """Initialize the data.

        Hold reference to img, initialize the weights and background rms.
        """
        self._img = img
        B, Ny, Nx = img.shape
        max_size = self.config.source_sizes[-1]
        if max(Ny,Nx) > max_size:
            logger.info("max source size {0} smaller than image size ({1},{2}); truncation possible".format(max_size, Ny, Nx))

        if bg_rms is None:
            self._bg_rms = np.zeros(self.B)
        else:
            assert len(bg_rms) == self.B
            self._bg_rms = np.array(bg_rms)
        self._set_weights(weights)

    def get_model(self, m=None, combine=True, combine_source_components=True, use_sed=True, flat=True):
        """Compute the current model for the entire image.
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
            if flat:
                models = np.vstack(models)
            else:
                models = np.array(models)
            return models

    def _register_sources(self, sources):
        """Unpack the components to register them as individual sources.
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
        """Set the weights and pixel covariance matrix `_Sigma_1`.

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
            self._weights = [1,1]
        else:
            self._weights = [None] * 2

            # for S update: normalize the per-pixel variation
            # i.e. in every pixel: utilize the bands with large weights
            # CAVEAT: need to filter out pixels that are saturated in every band
            norm_pixel = np.median(weights, axis=0)
            mask = norm_pixel > 0
            self._weights[1] = weights.copy()
            self._weights[1][:, mask] /= norm_pixel[mask]

            # reverse is true for A update: for each band, use the pixels that
            # have the largest weights
            norm_band = np.median(weights, axis=(1,2))
            # CAVEAT: some regions may have one band missing completely
            mask = norm_band > 0
            self._weights[0] = weights.copy()
            self._weights[0][mask] /= norm_band[mask,None,None]
            # CAVEAT: mask all pixels in which at least one band has W=0
            # these are likely saturated and their colors have large weights
            # but are incorrect due to missing bands
            mask = ~np.all(weights>0, axis=0)
            # and mask all bands for that pixel:
            # when estimating A do not use (partially) saturated pixels
            self._weights[0][:,mask] = 0

    def _compute_model(self):
        """Build the entire model.

        Calculate the full model once per iteration.
        This creates `self._models`, the morphological model of each source
        projected onto the full image, and `self._model`, which weighs
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
        """Proximal operator for the X update.

        To save processing time, the model is calculated when the first source
        is updated and all subsequent prox_f calculations (in the same iteration)
        use the same cached model.

        This is the proximal operator that is applied to the `A` or `S` update.
        See Algorithm 2, in Melchior & Moolekamp (in prep).

        Parameters
        ----------
        X: array-like
            Either `A` (sed) or `S` (morphology) of a single component of the model
        step: float
            Step size calculated using `self._steps_f`
        Xs: array-like
            List of all SEDs and morphologies of the model
            SEDs for all components, then morphologies for all components
        j: int
            Index of the current model component in `Xs`

        Returns
        -------
        result: `~numpy.array`
            Array of `X` after `_prox_f` has been applied
        """

        # which update to do now
        block = j//self.K
        k = j%self.K
        m,l = self.source_of(k)

        # computing likelihood gradients for S and A:
        # build model only once per iteration
        if k == 0:
            if block == self.config.update_order[0]:
                self._compute_model()

            # compute weighted residuals
            self._diff = self._weights[block]*(self._model-self._img)

        # A update
        if block == 0:
            if not self.sources[m].fix_sed[l]:
                # gradient of likelihood wrt A: nominally np.dot(diff, S^T)
                # but with PSF convolution, S_ij -> sum_q Gamma_bqi S_qj
                # however, that's exactly the operation done for models[k]
                grad = np.einsum('...ij,...ij', self._diff, self._models[k])

                # apply per component prox projection and save in source
                X = self.sources[m].sed[l] =  self.sources[m].constraints[l].prox_sed(X - step*grad, step)

        # S update
        elif block == 1:
            if not self.sources[m].fix_morph[l]:
                # gradient of likelihood wrt S: nominally np.dot(A^T,diff)
                # but again: with convolution, it's more complicated

                # first create diff image in frame of source k
                slice_m = self.sources[m].get_slice_for(self._img.shape)
                diff_k = np.zeros(self.sources[m].shape)
                diff_k[slice_m] = self._diff[self.sources[m].bb]

                # now a gradient vector and a mask of pixel with updates
                grad = np.zeros(X.shape, dtype=X.dtype)
                if not self.use_psf:
                    for b in range(self.B):
                        grad += self.sources[m].sed[l,b]*self.sources[m].Gamma.T.dot(diff_k[b])
                else:
                    for b in range(self.B):
                        grad += self.sources[m].sed[l,b]*self.sources[m].Gamma[b].T.dot(diff_k[b])

                # apply per component prox projection and save in source
                X = self.sources[m].morph[l] = self.sources[m].constraints[l].prox_morph(X - step*grad, step)


        # resize & recenter: after all blocks are updated
        if k == self.K - 1 and block == self.config.update_order[1]:
            self.it += 1

            if self.it % self.config.refine_skip == 0:
                resized = self.resize_sources()
                self.recenter_sources()
                self.adjust_absolute_error()
                if resized:
                    raise ScarletRestartException()

        return X


    def _one_over_lipschitz(self, block):
        """Calculate 1/Lipschitz constant for A and S
        """
        import scipy.sparse
        import scipy.sparse.linalg

        B, Ny, Nx = self._img.shape
        if block == 0: # A
            if self.config.exact_lipschitz:
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
            else:
                # Lipschitz constant for grad_A = || S S.T||_s
                # and the PSF is implicit
                # NOTE: if PSFs are very different between bands, this will fail
                # because we average over the bands
                PS = self._models.mean(axis=1).reshape((self.K, Ny*Nx))
                SSigma_1S = PS.dot(PS.T)
                LA = np.real(np.linalg.eigvals(SSigma_1S).max())
            return 1./LA

        if block == 1: # S
            if self.config.exact_lipschitz:
                if not self.use_psf:
                    # Lipschitz constant for grad_S = || A.T Sigma_1 A||_s
                    # need to go to frame in which A and S are serialized
                    PA = scipy.sparse.bmat([[scipy.sparse.identity(Ny*Nx) * self._A[b,k]
                                                for k in range(self.K)] for b in range(self.B)])
                else:
                    # similar calculation for S: ||Sigma_s||_s with
                    # Sigma_s = ((PA)^T Sigma_pixel^-1 PA)^-1
                    # in the frame where S is a vector of length N*K
                    PA = scipy.sparse.bmat([[self._A[b,k] * self._Gamma_full[self.source_of(k)[0]][b]
                                            for k in range(self.K)] for b in range(self.B)])
                ASigma_1A = PA.T.dot(self._Sigma_1[1].dot(PA))
                LS = np.real(scipy.sparse.linalg.eigs(ASigma_1A, k=1, return_eigenvectors=False)[0])
            else:
                # Lipschitz constant for grad_S = || A.T Sigma_1 A||_s
                ASigma_1A = self._A.T.dot(self._A)
                LS = np.real(np.linalg.eigvals(ASigma_1A).max())
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
            List of all SEDs and morphologies of the model
            SEDs for all components, then morphologies for all components
        j: int
            Index of the current model component in `Xs`

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
        if block == self.config.update_order[0] and k==0:
            self._compute_model()
            # compute step sizes and save to reuse for every component of A or S
            # _cbAS is the cached values 1/Lipschitz for A and S
            self._stepAS = [self._cbAS[block](block) for block in [0,1]]

        return self._stepAS[block]

    def recenter_sources(self):
        """Shift center position of sources to minimize residuals in all bands
        """
        # residuals weighted with full/original weight matrix
        y = self._weights[1]*(self._model-self._img)

        # Create the differential images for all sources
        MT = []
        updated = []
        for m in range(self.M):
            if self.sources[m].shift_center:
                source = self.sources[m]
                bb_m = source.bb
                diff_x,diff_y = self._get_shift_differential(m)
                if np.sum(diff_x)==0 or np.sum(diff_y)==0:
                    # The source might not have any flux,
                    # so don't try to fit it's position
                    logger.info("No flux in {0}, skipping recentering in it {1}".format(m, self.it))
                    continue
                diff_x[:,:,-1] = 0
                diff_y[:,-1,:] = 0

                # Project the difference image onto the full difference model
                # (which contains the difference images for all sources)
                _img_x = np.zeros(y.shape)
                _img_x[source.bb] = diff_x
                _img_y = np.zeros(y.shape)
                _img_y[source.bb] = diff_y
                updated.append(m)
                MT.append(_img_x.flatten())
                MT.append(_img_y.flatten())
        if len(MT)==0:
            # No sources needing updates
            logger.debug("No sources centers updated")
            return

        MT = np.array(MT)
        # Simultaneously fit the positions
        if not hasattr(self._weights,'shape'): # no/flat weights
            result = np.dot(np.dot(np.linalg.inv(np.dot(MT, MT.T)), MT), y.flatten())
        else:
            w = self._weights.flatten()[:,None]
            result = np.dot(np.dot(np.linalg.inv(np.dot(MT, MT.T*w)), MT), y.flatten())

        # Apply the corrections to all of the sources
        for m in range(self.M):
            if m not in updated:
                continue
            _m = updated.index(m)
            ddx, ddy = result[2*_m:2*_m+2]
            if ddx**2 + ddy**2 > self.config.center_min_dist**2:
                source = self.sources[m]
                center = source.center + (ddy, ddx)
                source.set_center(center)
                msg = "shifting source {0} by ({1:.3f}/{2:.3f}) to ({3:.3f}/{4:.3f}) in it {5}"
                logger.debug(msg.format(m, ddy, ddx, source.center[0], source.center[1], self.it))

    def _get_shift_differential(self, m):
        """Calculate the difference image used ot fit positions

        Parameters
        ----------
        m: int
            Index of the source in `~scarlet.blend.Blend.sources`

        Returns
        -------
        diff_img: `~numpy.array`
            Difference image in each band used to fit the position
        """
        # compute (model - dxy*shifted_model)/dxy for first-order derivative
        source = self.sources[m]
        slice_m = source.get_slice_for(self._img.shape)
        k = self.component_of(m, 0)
        model_m = self._models[k][self.sources[m].bb].copy() * source.sed[0][:,None,None]
        # in self._models, per-source components aren't combined,
        # need to combine here
        for l in range(1,source.K):
            k = self.component_of(m,l)
            model_m += self._models[k][self.sources[m].bb] * source.sed[l][:,None,None]

        # get Gamma matrices of source m with additional shift
        offset = source.shift_center
        dx = source.center - source.center_int
        pos_x = dx + (0, offset)
        pos_y = dx + (offset, 0)
        # If the source shifted off of the image,
        # recenter and re-initialize it
        center_x = source.center + pos_x
        center_y = source.center + pos_y
        width = self._img.shape[2]
        height = self._img.shape[1]
        #TODO: Implement bounding check on the source

        dGamma_x = source._gamma(pos_x)
        dGamma_y = source._gamma(pos_y)
        diff_img = [source.get_model(combine=True, Gamma=dGamma_x),
                    source.get_model(combine=True, Gamma=dGamma_y)]
        diff_img[0] = (model_m-diff_img[0][slice_m])/source.shift_center
        diff_img[1] = (model_m-diff_img[1][slice_m])/source.shift_center
        return diff_img

    def _set_edge_flux(self, m, model):
        """Keep track of the flux at the edge of the model.

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
        """Resize frames for sources (if necessary).

        If for any source, the mean flux at the edges of the frame exceeds
        `~scarlet.edge_flux_thresh` times the sky background in any band,
        increase the frame size of that source.

        The increase is set at `max(10, 0.25*size)` for the size of the source
        frame in either direction.

        """
        resized = False
        for m in range(self.M):
            if not self.sources[m].fix_frame:
                size = [self.sources[m].Ny, self.sources[m].Nx]
                increase = 1 # minimal increase, new size will be determine by config
                newsize = [self.config.find_next_source_size(size[i] + increase) for i in range(2)]

                # check if max flux along edge in band b < avg noise level along edge in b
                at_edge = (self._edge_flux[m] > self._bg_rms * self.config.edge_flux_thresh)
                # TODO: without symmetry constraints, the four edges of the box
                # should be allowed to resize independently
                _size = [size[0], size[1]]
                resized_source = False
                if (at_edge[0].any() or at_edge[2].any()):
                    size[0] = newsize[0]
                    resized = resized_source = True
                if (at_edge[1].any() or at_edge[3].any()):
                    size[1] = newsize[1]
                    resized = resized_source = True
                if resized_source:
                    logger.info("resizing source {0} from ({1},{2}) to ({3},{4}) at it {5}" .format(
                        m, _size[0], _size[1], size[0], size[1], self.it))
                    self.sources[m].resize(size)
        return resized

    def _absolute_morph_error(self):
        """Get the absolute morphology error
        """
        return [self.e_rel * self.sources[m].morph[l].mean()
                for m in range(self.M) for l in range(self.sources[m].K)]

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

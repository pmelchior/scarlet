import torch
import numpy as np

from .config import Config
from .source import ComponentTree

import logging
logger = logging.getLogger("scarlet.blend")


class Blend(ComponentTree):
    """The blended scene.

    The class represents a scene as collection of components, internally as a
    `~scarlet.component.ComponentTree`, and provides the functions to fit it
    to data.

    Attributes
    ----------
    B: int
        Number of bands in the image data
    it: int
        Number of iterations run in the `fit` method
    converged: `~numpy.array`
        Array (K, 2) of convergence flags, one for each components sed and morph
        in that order
    mse: list
        Array (it, 2) of mean squared errors in each iteration, for sed and morph
        in that order
    """

    def __init__(self, components, observations, config=None):
        """Constructor
        Form a blended scene from a collection of `~scarlet.component.Component`s
        Parameters
        ----------
        components: list of `~scarlet.component.Component` or `~scarlet.component.ComponentTree`
        """
        # check bands and PSFs
        self.it = 0
        if config is None:
            config = Config()
        self.config = config
        if not hasattr(observations, "__iter__"):
            observations = (observations,)
        self._observations = observations
        super(Blend, self).__init__(components)

    def fit(self, steps=200, e_rel=1e-2, padding=3):
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
        self.converged = False
        self.mse = []
        # Caches for 1/Lipschitz for A and S
        # self._cbAS = [proxmin.utils.ApproximateCache(self._one_over_lipschitz, slack=self.config.slack),
        #              proxmin.utils.ApproximateCache(self._one_over_lipschitz, slack=self.config.slack)]

        # define error limits
        self.e_rel = e_rel
        self._set_error_limits()

        for step in range(steps):
            self.it += 1
            full_morphs = torch.stack([c._morph for c in self.components])
            full_seds = torch.stack([c._sed for c in self.components])
            model = self.get_model()
            stepA, stepS = self._get_step_sizes(full_seds, full_morphs)

            for observation in self.observations:
                observation.gradient(model)

            for c in self.components:
                if not c.fix_sed:
                    c._sed.data = c.sed - 0.5 * stepA * c._sed.grad.data
                    c._sed.data = c.constraints.prox_sed(c.sed, step)
                    c._sed.grad.data.zero_()
                if not c.fix_morph:
                    c._morph.data = c.morph - 0.5 * stepS * c._morph.grad.data
                    center = np.unravel_index(np.argmax(c.morph), c.morph.shape)
                    c.update_center(center)
                    model = c.get_model(trim=False)
                    c._morph.data = c.constraints.prox_morph(c.morph, step)
                    c._morph.grad.data.zero_()

            if self._check_convergence():
                self.converged = True
                break

    def get_model(self, k=None, combine=True):
        """Compute the current model for the entire image.

        Parameters
        ----------
        k: int
            Index of a single component.
        combine: bool
            Whether all components should be combined.
        use_sed: bool
            Whether components are "colored" vs monochromatic.

        Returns
        -------
        `~numpy.array` with shape (B, Ny, Nx)
        """
        if k is not None:
            c = self.components[k]
            model = c.get_model(trim=False)
        else:
            # for all components
            model = torch.stack([self.get_model(k=k) for k in range(self.K)])
            if combine:
                model = torch.sum(model, dim=0)
        return model

    def _get_step_sizes(self, seds, morphs):
        """Get the Lipshitz constants for the observations and gradient priors
        """
        LA, LS = self._get_lipschitz(seds, morphs)
        LA *= len(self.observations)
        LS *= len(self.observations)
        return 1/LA, 1/LS

    @property
    def sources(self):
        """Return the list of `~scarlet.Source` used in the blend.
        """
        return self.nodes

    @property
    def observations(self):
        return self._observations

    def _check_convergence(self):
        seds = torch.zeros((self.K, self.B))
        size = self.components[0].morph.shape[0] * self.components[0].morph.shape[1]
        morphs = torch.zeros((self.K, size))
        for n, component in enumerate(self.components):
            seds[n] = component.sed
            morphs[n] = component.morph.reshape(size)
        if self.it <= 1:
            self.last_seds = seds.clone()
            self.last_morphs = morphs.clone()
            return False
        sed_diff = ((self.last_seds-seds)**2).sum(dim=1)
        morph_diff = ((self.last_morphs-morphs)**2).sum(dim=1)
        sed_converged = [sed_diff[k] <= self._e_rel[k]**2*(seds[k]**2).sum() for k in range(self.K)]
        morph_converged = [morph_diff[k] <= self._e_rel[k]**2*(morphs[k]**2).sum() for k in range(self.K)]
        converged = all(sed_converged) and all(morph_converged)
        self.last_seds = seds.clone()
        self.last_morphs = morphs.clone()
        return converged

    def _get_lipschitz(self, seds, morphs):
        _S = morphs.reshape(morphs.shape[0], -1)
        _SST = _S.mm(_S.t())
        _ATA = seds.t().mm(seds)
        # Lipschitz constant for A
        LA = torch.eig(_SST, eigenvectors=False)[0][:, 0].max().item()
        # Lipschitz constant for S
        LS = torch.eig(_ATA, eigenvectors=False)[0][:, 0].max().item()
        return LA, LS

    def _absolute_morph_error(self):
        """Get the absolute morphology error
        """
        return [self.e_rel * self.components[k].morph.mean() for k in range(self.K)]

    def _set_error_limits(self):
        """Set the error limits for each component
        """
        self._e_rel = [self.e_rel] * 2 * self.K
        # absolute errors: e_rel * mean signal, will be updated later
        self._e_abs = [self.e_rel / self.B] * self.K
        self._e_abs += self._absolute_morph_error()

    def _adjust_absolute_error(self):
        """Adjust the absolute error for each component
        """
        self._e_abs[self.K:] = self._absolute_morph_error()

import torch

from .component import ComponentTree, BlendFlag

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

    def __init__(self, scene, sources, observations):
        """Constructor
        Form a blended scene from a collection of `~scarlet.component.Component`s
        Parameters
        ----------
        components: list of `~scarlet.component.Component` or `~scarlet.component.ComponentTree`
        """
        super().__init__(sources)

        self.it = 0
        try:
            iter(observations)
        except TypeError:
            observations = (observations,)
        self._observations = observations
        self._scene = scene
        self.L_morph, self.L_sed = 1, 1

    @property
    def sources(self):
        """Return the list of `~scarlet.Source` used in the blend.
        """
        return self.nodes

    @property
    def observations(self):
        return self._observations

    @property
    def scene(self):
        return self._scene

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
        self.converged = False
        self.mse = []

        for step in range(steps):
            self.it += 1
            model = self.get_model()
            self._update_lipschitz()

            total_loss = 0
            for observation in self.observations:
                loss = observation.get_loss(model, self.scene)
                loss.backward()
                total_loss += loss.item()
            self.mse.append(total_loss)

            for c in self.components:
                c.backward_prior()
                if c._sed.requires_grad:
                    c._sed.data = c.sed - c._sed.grad.data / c.L_sed
                    c._sed.grad.data.zero_()
                if c._morph.requires_grad:
                    c._morph.data = c.morph - c._morph.grad.data / c.L_morph
                    c._morph.grad.data.zero_()

            self.update()

            if self._check_convergence(e_rel):
                break

    def _check_convergence(self, e_rel):
        e_rel2 = e_rel**2
        if self.it > 1:
            converged = True
            for component in self.components:
                # sed convergence
                diff2 = ((component._last_sed-component._sed)**2).sum()
                if diff2 <= e_rel2 * (component.sed**2).sum():
                    component.flags &= ~BlendFlag.SED_NOT_CONVERGED
                else:
                    component.flags |= BlendFlag.SED_NOT_CONVERGED
                    converged = False
                # morph convergence
                diff2 = ((component._last_morph-component._morph)**2).sum()
                if diff2 <= e_rel2 * (component.morph**2).sum():
                    component.flags &= ~BlendFlag.MORPH_NOT_CONVERGED
                else:
                    component.flags |= BlendFlag.MORPH_NOT_CONVERGED
                    converged = False
        else:
            converged = False

        for c in self.components:
            c._last_sed = c._sed.detach().clone()
            c._last_morph = c._morph.detach().clone()

        self.converged = converged
        return converged

    def _update_lipschitz(self):
        seds = torch.zeros(self.K, self.B).float()
        morphs = torch.zeros(self.K, self.scene.Ny, self.scene.Nx).float()
        for k, component in enumerate(self.components):
            seds[k] = component.sed
            morphs[k] = component.morph
        _S = morphs.reshape(morphs.shape[0], -1)
        _SST = _S.mm(_S.t())
        _ATA = seds.t().mm(seds)
        # Lipschitz constant for A
        LA = torch.eig(_SST, eigenvectors=False)[0][:, 0].max().item()
        # Lipschitz constant for S
        LS = torch.eig(_ATA, eigenvectors=False)[0][:, 0].max().item()

        LA *= len(self.observations)
        LS *= len(self.observations)

        self.L_sed = LA
        self.L_morph = LS
        for component in self.components:
            component.L_sed = self.L_sed
            component.L_morph = self.L_morph

import numpy as np

from .component import ComponentTree, BlendFlag
from .observation import Scene

import logging
logger = logging.getLogger("scarlet.blend")


class Blend(ComponentTree, Scene):
    """The blended scene.

    The class represents a scene as collection of components, internally as a
    `~scarlet.component.ComponentTree`, and provides the functions to fit it
    to data.

    Attributes
    ----------
    mse: list
        Array of mean squared errors in each iteration
    """

    def __init__(self, scene, sources, observations):
        """Constructor
        Form a blended scene from a collection of `~scarlet.component.Component`s
        Parameters
        ----------
        components: list of `~scarlet.component.Component` or `~scarlet.component.ComponentTree`
        """
        ComponentTree.__init__(self, sources)
        Scene.__init__(self, scene.shape, wcs=scene.wcs, psfs=scene.psfs, filtercurve=scene.filtercurve)

        try:
            iter(observations)
        except TypeError:
            observations = (observations,)
        self.observations = observations
        for obs in self.observations:
            obs.match(self)

        self.mse = []

    @property
    def it(self):
        """Number of iterations run in the `fit` method
        """
        return len(self.mse)

    @property
    def converged(self):
        """Whether the full Blend has converged within e_rel.

        For convergence tests of individual components, check its `flags` member.
        """
        for c in self.components:
            if (c.flags & (BlendFlag.SED_NOT_CONVERGED | BlendFlag.MORPH_NOT_CONVERGED)).value > 0:
                return False
        return True

    @property
    def sources(self):
        """Return the list of sources used in the blend.

        This will be different than `Blend.components` when
        some sources have multiple components.
        """
        return self.nodes

    def fit(self, max_iter=200, e_rel=1e-2, approximate_L=False):
        """Fit the model for each source to the data

        Parameters
        ----------
        max_iter: int
            Maximum number of iterations if the algorithm doesn't converge.
        e_rel: float
            Relative error for convergence of each component.
        approximate_L: bool
            Whether or not to use a rough approximation of the
            Lipschitz constants
        """

        for step in range(max_iter):

            # Combine all of the components into a single model
            model = self.get_model(False)

            # Caculate the gradients due to the observation likelihoods
            total_loss = 0
            for observation in self.observations:
                loss = observation.get_loss(model)
                loss.backward()
                total_loss += loss.item()
            self.mse.append(total_loss)

            # Calculate the Lipschitz constants,
            # which are needed to determine the step size for each component
            self._set_lipschitz(approximate_L)

            # Take the next gradient step for each component
            for c in self.components:
                c.L_sed = self.L_sed
                c.L_morph = self.L_morph

                c.backward_prior()
                if c._sed.requires_grad:
                    c._sed.data = c._sed.data - c.step_sed * c._sed.grad.data
                    c._sed.grad.data.zero_()
                if c._morph.requires_grad:
                    c._morph.data = c._morph.data - c.step_morph * c._morph.grad.data
                    c._morph.grad.data.zero_()

            # Call the update functions for all of the sources
            self.update()

            if self._check_convergence(e_rel):
                break

    def _check_convergence(self, e_rel):
        """Check to see if all of the components have converged

        The `flag` property of each component is updated to reflect
        whether or not its SED and morphology have converged.

        Parameters
        ----------
        e_rel: float
            Relative error for convergence of each component.

        Returns
        -------
        converged: bool
            Whether or not all of the components have converged.
        """
        e_rel2 = e_rel**2
        if self.it > 1:
            converged = True
            for component in self.components:
                # sed convergence
                diff2 = ((component._last_sed-component.sed)**2).sum()
                if diff2 <= e_rel2 * (component.sed**2).sum():
                    component.flags &= ~BlendFlag.SED_NOT_CONVERGED
                else:
                    component.flags |= BlendFlag.SED_NOT_CONVERGED
                    converged = False
                # morph convergence
                diff2 = ((component._last_morph-component.morph)**2).sum()
                if diff2 <= e_rel2 * (component.morph**2).sum():
                    component.flags &= ~BlendFlag.MORPH_NOT_CONVERGED
                else:
                    component.flags |= BlendFlag.MORPH_NOT_CONVERGED
                    converged = False
        else:
            converged = False

        # Store a copy of each SED and morphology for the
        # convergence check in the next iteration
        for c in self.components:
            c._last_sed = c.sed.copy()
            c._last_morph = c.morph.copy()

        return converged

    def _set_lipschitz(self, approximate_L):
        """Update the Lipschitz constants from the observations
        """
        if approximate_L:
            # spectral norm of S*S^T or A^T*A is of order the mean amplitude
            # of their elements times the number of components
            LA, LS = 0, 0
            for c in self.components:
                LS += (c.sed**2).sum().item()
                LA += (c.morph**2).sum().item()

            # if loss has increased from last iteration, increase L
            # TODO: this is insufficient if the estimation above is miles off!
            # Then: escalate the increase
            if self.it > 1 and self.mse[-1] > self.mse[-2]:
                LS *= 2
                LA *= 2
        else:
            # This is still an approximation, but a less crude (albiet slower) one
            seds = np.zeros((self.K, self.B), dtype=self.components[0].sed.dtype)
            morphs = np.zeros((self.K, self.Ny, self.Nx), dtype=self.components[0].morph.dtype)
            for k, component in enumerate(self.components):
                seds[k] = component.sed
                morphs[k] = component.morph
            _S = morphs.reshape(morphs.shape[0], -1)
            _SST = _S.dot(_S.T)
            _ATA = seds.T.dot(seds)
            # Lipschitz constant for A
            LA = np.real(np.linalg.eigvals(_SST).max())
            # Lipschitz constant for S
            LS = np.real(np.linalg.eigvals(_ATA).max())

        LA *= len(self.observations)
        LS *= len(self.observations)

        self.L_sed = LA
        self.L_morph = LS

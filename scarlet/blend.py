import autograd.numpy as np
from autograd import grad

from .component import ComponentTree, BlendFlag

import logging

logger = logging.getLogger("scarlet.blend")


class Blend(ComponentTree):
    """The blended scene

    The class represents a scene as collection of components, internally as a
    `~scarlet.component.ComponentTree`, and provides the functions to fit it
    to data.

    Attributes
    ----------
    mse: list
        Array of mean squared errors in each iteration
    """

    def __init__(self, sources, observations):
        """Constructor

        Form a blended scene from a collection of `~scarlet.component.Component`s

        Parameters
        ----------
        sources: list of `~scarlet.component.Component` or `~scarlet.component.ComponentTree`
            Intitialized components or sources to fit to the observations
        observations: a `scarlet.Observation` instance or a list thereof
            Data package(s) to fit
        """
        ComponentTree.__init__(self, sources)

        try:
            iter(observations)
        except TypeError:
            observations = (observations,)
        self.observations = observations

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

    def fit(self, max_iter=200, e_rel=1e-2, step_size=0.1, b1=0.5, b2=0.5, eps=1e-8):
        """Fit the model for each source to the data

        Parameters
        ----------
        max_iter: int
            Maximum number of iterations if the algorithm doesn't converge.
        e_rel: float
            Relative error for convergence of each component.
        """
        # compute the backward gradient tree
        x = self.parameters
        n_params = len(x)
        m = [np.zeros(x_.shape, x_.dtype) for x_ in x]
        v = [np.zeros(x_.shape, x_.dtype) for x_ in x]
        self._grad = grad(self._loss, tuple(range(n_params)))

        for it in range(max_iter):
            g = self._grad(*x)

            # TODO: need prior gradients here...

            # Adam gradient updates
            for p in range(n_params):
                m[p] = (1 - b1) * g[p] + b1 * m[p]       # First  moment estimate
                v[p] = (1 - b2) * (g[p]**2) + b2 * v[p]  # Second moment estimate
                mhat = m[p] / (1 - b1**(it + 1))          # Bias corrections
                vhat = v[p] / (1 - b2**(it + 1))
                # inline update
                x[p] -= step_size*mhat/(np.sqrt(vhat) + eps)

                # TODO: need to store effective step size in components
                # these are not constant even within one parameter!

                # TODO: check convergence
                # BlendFlag need to be enumerated because number of parameters is free

            # Call the update functions for all of the sources
            self.update()

    def _loss(self, *parameters):
        """Loss function for autograd

        This method combines the seds and morphologies
        into a model that is used to calculate the loss
        function and update the gradient for each
        parameter
        """
        model = self.get_model(*parameters)
        # Caculate the total loss function from all of the observations
        total_loss = 0
        for observation in self.observations:
            total_loss = total_loss + observation.get_loss(model)
        self.mse.append(total_loss._value)
        return total_loss

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
        e_rel2 = e_rel ** 2
        if self.it > 1:
            converged = True
            for component in self.components:
                # sed convergence
                diff2 = ((component._last_sed - component.sed) ** 2).sum()
                if diff2 <= e_rel2 * (component.sed ** 2).sum():
                    component.flags &= ~BlendFlag.SED_NOT_CONVERGED
                else:
                    component.flags |= BlendFlag.SED_NOT_CONVERGED
                    converged = False
                # morph convergence
                diff2 = ((component._last_morph - component.morph) ** 2).sum()
                if diff2 <= e_rel2 * (component.morph ** 2).sum():
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

import autograd.numpy as np
from autograd import grad

from .component import ComponentTree

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

    def fit(self, max_iter=200, e_rel=1e-3, step_size=1e-2, b1=0.5, b2=0.999):
        """Fit the model for each source to the data

        Parameters
        ----------
        max_iter: int
            Maximum number of iterations if the algorithm doesn't converge.
        e_rel: float
            Relative error for convergence of each component.
        """

        # dynamically call parameters to allow for addition / fixing
        x = self.parameters
        n_params = len(x)
        m = [np.zeros(x_.shape, x_.dtype) for x_ in x]
        v = [np.zeros(x_.shape, x_.dtype) for x_ in x]
        vhat = [np.zeros(x_.shape, x_.dtype) for x_ in x]

        # compute the backward gradient tree
        self._grad = grad(self._loss, tuple(range(n_params)))
        self._set_stepsizes(*x, step_size=step_size)
        e_rel2 = e_rel ** 2
        self.mse = []
        converged = False

        for it in range(max_iter):
            g = self._grad(*x)
            gp = self._grad_prior(*x)
            b1t = b1**(it+1)
            b1tm1 = b1**it

            # for convergence test
            x_ = [x_.view(np.ndarray).copy() for x_ in x]

            # AdamX gradient updates
            for j in range(n_params):
                ggp = g[j] + gp[j]
                m[j] = (1 - b1t) * ggp + b1t * m[j]
                v[j] = (1 - b2) * ggp**2 + b2 * v[j]
                if it == 0:
                    vhat[j] = v[j]
                else:
                    vhat[j] = np.maximum(v[j], vhat[j] * (1 - b1t)**2 / (1 - b1tm1)**2)

                # inline update from gradients
                x[j] -= x[j].step * m[j] / np.sqrt(vhat[j])

            # Call the update functions for all of the sources
            # convergence needs to be evaluated *after* the prox updates
            self.update()

            converged = True
            for j in range(n_params):
                x[j].converged = np.sum((x_[j] - x[j]._data)**2) <= e_rel2 * np.sum(x[j]._data**2)
                converged &= x[j].converged

            if converged:
                break

        return self


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

    def _grad_prior(self, *parameters):
        # TODO: could use collecting identical priors to run on mini-batches
        return [ p.prior(p.view(np.ndarray)) if p.prior is not None else 0 for p in parameters ]

    def _set_stepsizes(self, *parameters, step_size=1e-2):
        # pick step as a fraction of the mac value of the parameter
        # since we don't want faint sources to slow down, we set all sources to
        # the same step size
        max_value = {}
        for p in parameters:
            max_value[p.name] = max(p._data.max(), max_value.get(p.name, 0))
        for p in parameters:
            p.step = step_size * max_value[p.name]

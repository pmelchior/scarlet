import numpy.ma as ma
import autograd.numpy as np
from autograd import grad
import proxmin
from functools import partial

from .component import ComponentTree


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
        self.loss = []

    def fit(self, max_iter=200, e_rel=1e-3, min_iter=1, **alg_kwargs):
        """Fit the model for each source to the data

        Parameters
        ----------
        max_iter: int
            Maximum number of iterations if the algorithm doesn't converge
        e_rel: float
            Relative error for convergence of the loss function
        min_iter: int
            Maximum number of iterations if the algorithm doesn't converge
        alg_kwargs: dict
            Keywords for the `proxmin.adaprox` optimizer
        """
        # dynamically call parameters to allow for addition / fixing
        X = self.parameters
        n_params = len(X)

        # compute the backward gradient tree
        grad_logL = grad(self._loss, tuple(range(n_params)))
        grad_logP = lambda *X: tuple(
            x.prior(x.view(np.ndarray)) if x.prior is not None else 0 for x in X
        )
        _grad = lambda *X: tuple(l + p for l, p in zip(grad_logL(*X), grad_logP(*X)))
        _step = lambda *X, it: tuple(
            x.step(x, it=it) if hasattr(x.step, "__call__") else x.step for x in X
        )
        _prox = tuple(x.constraint for x in X)

        # good defaults for adaprox
        scheme = alg_kwargs.pop("scheme", "amsgrad")
        prox_max_iter = alg_kwargs.pop("prox_max_iter", 10)
        eps = alg_kwargs.pop("eps", 1e-8)
        callback = partial(
            self._callback, e_rel=e_rel, callback=alg_kwargs.pop("callback", None), min_iter=min_iter,
        )

        # do we have a current state of the optimizer to warm start?
        M = tuple(x.m if x.m is not None else np.zeros(x.shape) for x in X)
        V = tuple(x.v if x.v is not None else np.zeros(x.shape) for x in X)
        Vhat = tuple(x.vhat if x.vhat is not None else np.zeros(x.shape) for x in X)

        proxmin.adaprox(
            X,
            _grad,
            _step,
            prox=_prox,
            max_iter=max_iter,
            e_rel=e_rel,
            check_convergence=False,
            scheme=scheme,
            prox_max_iter=prox_max_iter,
            callback=callback,
            M=M,
            V=V,
            Vhat=Vhat,
            **alg_kwargs
        )

        # set convergence and standard deviation from optimizer
        for p, m, v, vhat in zip(X, M, V, Vhat):
            p.m = m
            p.v = v
            p.vhat = vhat
            p.std = 1 / np.sqrt(ma.masked_equal(v, 0))  # this is rough estimate!

        return self

    def get_model(self, *parameters):
        """Override `ComponentTree.get_model` to use the model frame

        See `~scarlet.ComponentTree.get_model` for more info.
        """
        return super().get_model(*parameters, frame=self.model_frame)

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

        self.loss.append(total_loss._value)
        return total_loss

    def _callback(self, *parameters, it=None, e_rel=1e-3, callback=None, min_iter=1):

        # raise ArithmeticError if some of the parameters have become inf/nan
        self.check_parameters()

        if it > min_iter and abs(self.loss[-2] - self.loss[-1]) < e_rel * np.abs(
            self.loss[-1]
        ):
            raise StopIteration("scarlet.Blend.fit() converged")

        if callback is not None:
            callback(*parameters, it=it)

    @property
    def bbox(self):
        """Bounding box of the blend

        Override the bounding box of the `ComponentTree`,
        which includes the area of sources that extend beyond
        the model frame boundaries.
        """
        return self.model_frame.bbox

import numpy.ma as ma
import autograd.numpy as np
from autograd import grad
from autograd.extend import defvjp, primitive
import proxmin
from functools import partial


@primitive
def _add_models(*models, full_model, slices):
    """Insert the models into the full model

    `slices` is a tuple `(full_model_slice, model_slices)` used
    to insert a model into the full_model in the region where the
    two models overlap.
    """
    for i in range(len(models)):
        if hasattr(models[i], "_value"):
            full_model[slices[i][0]] += models[i][slices[i][1]]._value
        else:
            full_model[slices[i][0]] += models[i][slices[i][1]]
    return full_model


def _grad_add_models(upstream_grad, *models, full_model, slices, index):
    """Gradient for a single model

    The full model is just the sum of the models,
    so the gradient is 1 for each model,
    we just have to slice it appropriately.
    """
    model = models[index]
    full_model_slices = slices[index][0]
    model_slices = slices[index][1]

    def result(upstream_grad):
        _result = np.zeros(model.shape, dtype=model.dtype)
        _result[model_slices] = upstream_grad[full_model_slices]
        return _result

    return result


class Blend:
    """The blended scene

    The class represents a scene as collection of and provides the functions to
    fit it to data.

    Attributes
    ----------
    loss: list
        Negative log likelihood in each iteration
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

        if hasattr(sources, "__iter__"):
            self.sources = sources
        else:
            self.sources = (sources,)

        if hasattr(observations, "__iter__"):
            self.observations = observations
        else:
            self.observations = (observations,)
        self.model_frame = self.sources[0].model_frame
        self.loss = []

    def fit(self, max_iter=200, e_rel=1e-3, min_iter=1, random_skip=0, **alg_kwargs):
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
        X = []
        for src in self.sources:
            X += src.parameters
        n_params = len(X)

        # compute the backward gradient tree
        grad_logL = grad(self._loss, tuple(range(n_params)))
        grad_logP = lambda *X: tuple(
            x.prior(x.view(np.ndarray)) if x.prior is not None else 0 for x in X
        )
        _grad = lambda *X: tuple(l + p for l, p in zip(grad_logL(*X), grad_logP(*X)))
        _step = lambda *X, it: tuple(
            1e-20
            if np.random.rand() < random_skip
            else x.step(x, it=it)
            if hasattr(x.step, "__call__")
            else x.step
            for x in X
        )
        _prox = tuple(x.constraint for x in X)

        # good defaults for adaprox
        scheme = alg_kwargs.pop("scheme", "amsgrad")
        prox_max_iter = alg_kwargs.pop("prox_max_iter", 10)
        callback = partial(
            self._callback,
            e_rel=e_rel,
            callback=alg_kwargs.pop("callback", None),
            min_iter=min_iter,
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

    def get_model(self, *parameters, frame=None):
        """Get the model of the entire blend

        Parameters
        ----------
        parameters: tuple of optimization parameters
        frame:  `scarlet.Frame`
            Alternative Frame to project the model into

        Returns
        -------
        model: array
            (Bands, Height, Width) data cube
        """
        if frame is None:
            frame = self.model_frame

        # if this is the model frame then the slices are already cached
        if frame == self.model_frame:
            use_cached = True
        else:
            use_cached = False

        full_model = np.zeros(frame.shape, dtype=frame.dtype)

        models = []
        slices = []
        i = 0
        for src in self.sources:
            if len(parameters):
                j = len(src.parameters)
                p = parameters[i : i + j]
                i += j
                model = src.get_model(*p)
            else:
                model = src.get_model()

            models.append(model)

            if use_cached:
                slices.append((src.model_frame_slices, src.model_slices))
            else:
                # Get the slices needed to insert the model
                slices.append(overlapped_slices(frame.bbox, src.bbox))

        # We have to declare the function that inserts sources
        # into the blend with autograd.
        # This has to be done each time we fit a blend,
        # since the number of components => the number of arguments,
        # which must be linked to the autograd primitive function.
        defvjp(
            _add_models,
            *([partial(_grad_add_models, index=k) for k in range(len(self.sources))])
        )

        full_model = _add_models(*models, full_model=full_model, slices=slices)

        return full_model

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
        for src in self.sources:
            src.check_parameters()

        if it > min_iter and abs(self.loss[-2] - self.loss[-1]) < e_rel * np.abs(
            self.loss[-1]
        ):
            raise StopIteration("scarlet.Blend.fit() converged")

        if callback is not None:
            callback(*parameters, it=it)

    @property
    def bbox(self):
        """Bounding box of the blend
        """
        return self.model_frame.bbox

import numpy.ma as ma
import autograd.numpy as np
from autograd import grad
import proxmin
from functools import partial

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
        self.loss = []

    def fit(self, max_iter=200, e_rel=1e-3, f_rel=1e-4, **alg_kwargs):
        """Fit the model for each source to the data

        Parameters
        ----------
        max_iter: int
            Maximum number of iterations if the algorithm doesn't converge.
        e_rel: float
            Relative error for convergence of each component.
        alg_kwargs: dict
            Keywords for the `proxmin.adaprox` optimizer
        """

        # dynamically call parameters to allow for addition / fixing
        X = self.parameters
        n_params = len(X)

        priors = [p.prior for p in x if p.prior is not None]
        batch_size = len(priors)
        if batch_size > 0:
            import tensorflow as tf
            # We are assuming that the same stamp size is used for all priors
            stamp_size = priors[0].stamp_size

            inx = tf.placeholder(shape=[batch_size, stamp_size, stamp_size, 1],
                                 dtype=tf.float32)
            splits = tf.split(inx, num_or_size_splits=batch_size)
            grad_prior = tf.concat([p.grad(s) for p,s in zip(priors, splits)],axis=0)

            self.sess = tf.Session()
            self.sess.run(tf.global_variables_initializer())

            self._compute_grad_prior = lambda x: self.sess.run(grad_prior, feed_dict={inx: x})

        # compute the backward gradient tree
        grad_logL = grad(self._loss, tuple(range(n_params)))
        grad_logP = lambda *X: tuple(x.prior(x.view(np.ndarray)) if x.prior is not None else 0 for x in X)
        _grad = lambda *X: tuple(l + p for l,p in zip(grad_logL(*X), grad_logP(*X)))
        _step = lambda *X, it: tuple(x.step(x, it=it) if hasattr(x.step, "__call__") else x.step for x in X)
        _prox = tuple(x.constraint for x in X)

        # good defaults for adaprox
        scheme = alg_kwargs.pop('scheme', 'amsgrad')
        prox_max_iter = alg_kwargs.pop('prox_max_iter', 10)
        eps = alg_kwargs.pop('eps', 1e-8)
        callback = partial(self._convergence_callback, f_rel=f_rel, callback=alg_kwargs.pop('callback', None))

        converged, G, V = proxmin.adaprox(X, _grad, _step, prox=_prox, max_iter=max_iter, e_rel=e_rel, scheme=scheme, prox_max_iter=prox_max_iter, callback=callback, **alg_kwargs)

        # set convergence and standard deviation from optimizer
        for p,c,g,v in zip(X, converged, G, V):
            p.converged = c
            p.std = 1/np.sqrt(ma.masked_equal(v, 0)) # this is rough estimate!

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
        self.loss.append(total_loss._value)
        return total_loss

    def _convergence_callback(self, *parameters, it=None, f_rel=1e-3, callback=None):
        if it > 1 and abs(self.loss[-2] - self.loss[-1]) < f_rel * np.abs(self.loss[-1]):
            raise StopIteration("scarlet.Blend.fit() converged")

        if callback is not None:
            callback(*parameters, it=it)


    def _grad_prior(self, *parameters):
        # TODO: could use collecting identical priors to run on mini-batches
        #return [ p.prior(p.view(np.ndarray)) if p.prior is not None else 0 for p in parameters ]
        batch = []
        for p in parameters:
            if p.prior is not None:
                bbox, padding = p.get_centered_ROI(p.prior.stamp_size)
                roi = np.pad(p[bbox.slices], padding, mode='constant')
                batch.append(roi.reshape((1, p.prior.stamp_size, p.prior.stamp_size, 1)))

        if len(batch) == 0:
            return [0,]*len(parameters)

        # Concatenate stamps and feed them to the network
        batch = self._compute_grad_prior(np.concatenate(batch, axis=0).astype('float32'))

        # Extract the results and interleave 0s for parameters not affected by
        # prior
        grad_prior = []
        ind = 0
        for p in parameters:
            if p.prior is not None:
                gp = np.zeros(p.shape, dtype=p.dtype)
                bbox, padding = p.get_centered_ROI(p.prior.stamp_size)
                (bottom, top), (left, right) = padding
                top = None if top == 0 else -top
                right = None if right == 0 else -right

                gp[bbox.slices] = batch[ind][bottom:top, left:right][:,:,0]
                ind += 1
                grad_prior.append(gp)
            else:
                grad_prior.append(0)

        return grad_prior

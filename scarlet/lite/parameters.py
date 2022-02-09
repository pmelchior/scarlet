# This module contains functions and classes for parameters in scarlet lite.
# Unlike scarlet main, a LiteParameter contains all of the information
# to update a parameter, so each type of regression algorithm will
# have its own `LiteParameter` class.
from abc import ABC, abstractmethod

import numpy as np
import proxmin


def grow_array(x, new_shape, dist):
    """grow an array and pad it with zeros

    This is faster than `numpy.pad` by a factor of ~20.

    Parameters
    ----------
     x: `~numpy.array`
        The array to grow
    new_shape: `tuple` of `int`
        This is the new shape of the array.
        It would be trivial to calculate in this function,
        however in most cases this is already calculated for
        other purposes, so it might as well be reused.
    dist: int
        The amount to pad each side of the input array
        (so the new shape is extended by 2*dist on each axis).

    Returns
    -------
    result: `~numpy.array`
        The larger array that contains `x`.
    """
    result = np.zeros(new_shape, dtype=x.dtype)
    result[dist:-dist, dist:-dist] = x
    return result


class LiteParameter(ABC):
    """A parameter in a `LiteComponent`

    Unlike the main scarlet `Parameter` class,
    a `LiteParameter` also contains methods to update the parameter,
    using any given optimizer, provided the abstract methods
    are all implemented. The main parameter should always be
    stored as `LiteParameter.x`, but the names of the meta parameters
    can be different.
    """
    @abstractmethod
    def update(self, it, input_grad, *args):
        """Update the parameter in one iteration.

        This includes the gradient update, proximal update,
        and any meta parameters that are stored as class
        attributes to update the parameter.

        Parameters
        ----------
        it: int
            The current iteration
        input_grad: `~numpy.array`
            The gradient from the full model, passed to the parameter.
        """
        pass

    @abstractmethod
    def grow(self, new_shape, dist):
        """Grow the parameter and all of the meta parameters

        Parameters
        ----------
        new_shape: `tuple` of `int`
            The new shape of the parameter.
        dist: `int`
            The amount to extend the array in each direction
        """
        pass

    @abstractmethod
    def shrink(self, dist):
        """Shrink the parameter and all of the meta parameters

        Parameters
        ----------
        dist: `int`
            The amount to shrink the array in each direction
        """
        pass


class FistaParameter(LiteParameter):
    """A `LiteParameter` that updates itself using the Beck-Teboulle 2009
    FISTA proximal gradient method.

    See https://www.ceremade.dauphine.fr/~carlier/FISTA
    """
    def __init__(self, x, step, grad=None, prox=None, t0=1, z0=None):
        """Initialize the parameter

        Parameters
        ----------
        x: `~numpy.array`
            The initial guess for the parameter.
        step: `float`
            The step size for the parameter.
            This is scaled in each step by the first argument to
            `update` after the `input_grad`.
        grad: `func`
            The function to use to calculate the gradient.
            `grad` should accept the `input_grad` and a list
            of arguments.
        prox: `func`
            The function that acts as a proximal operator.
            This function should take `x` as an input, however
            the input `x` might not be the same as the input
            parameter, but a meta parameter instead.
        t0: `float`
            The initial value of the acceleration parameter.
        z0: `~numpy.array`
            The initial value of the meta parameter `z`.
            If this is `None` then `z` is initialized to the
            initial `x`.
        """
        if z0 is None:
            z0 = x
        self.x = x
        self.step = step
        self.grad = grad
        self.prox = prox
        self.z = z0
        self.t = t0

    def update(self, it, input_grad, *args):
        """Update the parameter and meta-parameters using the PGM

        See `~LiteParameter` for more.
        """
        step = self.step/np.sum(args[0]*args[0])

        y = self.z - step*self.grad(input_grad, self.x, *args)
        x = self.prox(y, step)
        t = 0.5*(1 + np.sqrt(1 + 4*self.t**2))
        omega = 1 + (self.t -1)/t
        self.z = self.x + omega*(x-self.x)
        self.x = x
        self.t = t

    def grow(self, new_shape, dist):
        self.x = grow_array(self.x, new_shape, dist)
        self.z = grow_array(self.z, new_shape, dist)
        #self.t = 1

    def shrink(self, dist):
        self.x = self.x[dist:-dist, dist:-dist]
        self.z = self.z[dist:-dist, dist:-dist]
        #self.t = 1


phi_psi = {
    "adam": proxmin.algorithms._adam_phi_psi,
    "nadam": proxmin.algorithms._nadam_phi_psi,
    "amsgrad": proxmin.algorithms._amsgrad_phi_psi,
    "padam": proxmin.algorithms._padam_phi_psi,
    "adamx": proxmin.algorithms._adamx_phi_psi,
    "radam": proxmin.algorithms._radam_phi_psi,
}


class SingleItemArray:
    """Mock an array with only a single item
    """
    def __init__(self, value):
        self.value = value

    def __getitem__(self, item):
        return self.value


class AdaproxParameter(LiteParameter):
    """Operator updated using te Proximal ADAM algorithm

    Uses multiple variants of adaptive quasi-Newton gradient descent
        * Adam (Kingma & Ba 2015)
        * NAdam (Dozat 2016)
        * AMSGrad (Reddi, Kale & Kumar 2018)
        * PAdam (Chen & Gu 2018)
        * AdamX (Phuong & Phong 2019)
        * RAdam (Liu et al. 2019)
    and PGM sub-iterations to satisfy feasibility and optimality. See details of the
    algorithms in the respective papers.

    TODO: implement other schemes by making `b1` use a list instead of a single value.
    """
    def __init__(self, x, step, grad=None, prox=None, b1=0.9, b2=0.999, eps=1e-8, p=0.25,
                 m0=None, v0=None, vhat0=None, scheme="amsgrad", max_prox_iter=1, prox_e_rel=1e-6):
        """Initialize the parameter

         NOTE:
        Setting `m`, `v`, `vhat` allows to continue a previous run, e.g. for a warm start
        of a slightly changed problem. If not set, they will be initialized with 0.

        Parameter
        ---------
        x: `~numpy.array`
            The initial guess for the parameter.
        step: `func`
            The step size for the parameter that takes the
            parameter `x` and the iteration `it` as arguments.
        grad: `func`
            The function to use to calculate the gradient.
            `grad` should accept the `input_grad` and a list
            of arguments.
        prox: `func`
            The function that acts as a proximal operator.
            This function should take `x` as an input, however
            the input `x` might not be the same as the input
            parameter, but a meta parameter instead.
        b1: `float`
            The strength parameter for the weighted gradient
            (`m`) update.
        b2: `float`
            The strength for the weighted gradient squared
            (`v`) update.
        eps: `float`
            Minimum value of the cumulative gradient squared
            (`vhat`) meta paremeter.
        p: `float`
            Meta parameter used by some of the ADAM schemes
        m0: `~numpy.array`
            Initial value of the weighted gradient (`m`) parameter
            for a warm start.
        v0: `~numpy.array`
            Initial value of the weighted gradient squared(`v`) parameter
            for a warm start.
        vhat0: `~numpy.array`
            Initial value of the
            cumulative weighted gradient squared (`vhat`) parameter
            for a warm start.
        scheme: str
            Name of the ADAM scheme to use.
            One of ["adam", "nadam", "adamx", "amsgrad", "padam", "radam"]
        """
        self.x = x

        if not hasattr(b1, "__getitem__"):
            b1 = SingleItemArray(b1)

        self.b1 = b1
        self.b2 = b2
        self.eps = eps
        self.p = p

        if not hasattr(step, "__call__"):
            def _step(x, it):
                return step
            self.step = _step
        else:
            self.step = step
        self.grad = grad
        self.prox = prox
        if m0 is None:
            m0 = np.zeros(x.shape, dtype=x.dtype)
        self.m = m0
        if v0 is None:
            v0 = np.zeros(x.shape, dtype=x.dtype)
        self.v = v0
        if vhat0 is None:
            vhat0 = np.ones(x.shape, dtype=x.dtype) * -np.inf
        self.vhat = vhat0
        self.phi_psi = phi_psi[scheme]
        self.max_prox_iter = max_prox_iter
        self.e_rel = prox_e_rel

    def update(self, it, input_grad, *args):
        """Update the parameter and meta-parameters using the PGM

        See `~LiteParameter` for more.
        """
        # Calculate the gradient
        grad = self.grad(input_grad, self.x, *args)
        # Get the update for the parameter
        phi, psi = self.phi_psi(
            it, grad, self.m, self.v, self.vhat,
            self.b1, self.b2, self.eps, self.p
        )
        # Calculate the step size
        step = self.step(self.x, it)
        if it > 0:
            self.x -= step * phi/psi
        else:
            self.x -= step * phi / psi/10

        # Iterate over the proximal operators until convergence
        if self.prox is not None:
            z = self.x.copy()
            gamma = step/np.max(psi)
            for tau in range(1, self.max_prox_iter + 1):
                _z = self.prox(z - gamma/step * psi * (z-self.x), gamma)
                converged = proxmin.utils.l2sq(_z-z) <= self.e_rel**2 * proxmin.utils.l2sq(z)
                z = _z

                if converged:
                    break

            self.x = z

    def grow(self, new_shape, dist):
        self.x = grow_array(self.x, new_shape, dist)
        self.m = grow_array(self.m, new_shape, dist)
        self.v = grow_array(self.v, new_shape, dist)
        self.vhat = grow_array(self.vhat, new_shape, dist)

    def shrink(self, dist):
        self.x = self.x[dist:-dist, dist:-dist]
        self.m = self.m[dist:-dist, dist:-dist]
        self.v = self.v[dist:-dist, dist:-dist]
        self.vhat = self.vhat[dist:-dist, dist:-dist]

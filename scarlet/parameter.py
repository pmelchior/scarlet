import autograd.numpy as np
from autograd.numpy.numpy_boxes import ArrayBox
from autograd.core import VSpace
from functools import partial
from .constraint import Constraint, ConstraintChain
from .prior import Prior


class Parameter(np.ndarray):
    """Optimization parameter

    Parameters
    ----------
    array: array-like
        numpy array (type float) to hold parameter values
    name: string
        Name to identify parameter
    prior: `~scarlet.Prior`
        Prior distribution for parameter
    constraint: `~scarlet.Constraint`
        Constraint on parameter
    step: float or method
        The step size for the parameter
        If a method is used, it needs to have the signature
            `step(X, it) -> float`
        where `X` is the parameter value and `it` the iteration counter
    std: array-like
        Statistical error estimate, same shape as `array`
    fixed: bool
        Whether parameter is held fixed (excluded) during optimization
    """

    def __new__(
        cls,
        array,
        name="unnamed",
        prior=None,
        constraint=None,
        step=0,
        std=None,
        m=None,
        v=None,
        vhat=None,
        fixed=False,
    ):
        obj = np.asarray(array, dtype=array.dtype).view(cls)
        obj.name = name
        if prior is not None:
            assert isinstance(prior, Prior)
        obj.prior = prior
        if constraint is not None:
            assert isinstance(constraint, Constraint) or isinstance(
                constraint, ConstraintChain
            )
        obj.constraint = constraint
        obj.step = step
        obj.std = std
        obj.m = m
        obj.v = v
        obj.vhat = vhat
        obj.fixed = fixed
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.name = getattr(obj, "name", "unnamed")
        self.prior = getattr(obj, "prior", None)
        self.constraint = getattr(obj, "constraint", None)
        self.step = getattr(obj, "step_size", 0)
        self.std = getattr(obj, "std", None)
        self.m = getattr(obj, "m", None)
        self.v = getattr(obj, "v", None)
        self.vhat = getattr(obj, "vhat", None)
        self.fixed = getattr(obj, "fixed", False)

    def __reduce__(self):
        # Get the parent's __reduce__ tuple
        pickled_state = super().__reduce__()
        # Create our own tuple to pass to __setstate__, but append the __dict__ rather than individual members.
        new_state = pickled_state[2] + (self.__dict__,)
        # Return a tuple that replaces the parent's __setstate__ tuple with our own
        return (pickled_state[0], pickled_state[1], new_state)

    def __setstate__(self, state):
        self.__dict__.update(state[-1])  # Update the internal dict from state
        # Call the parent's __setstate__ with the other tuple elements.
        super().__setstate__(state[0:-1])

    @property
    def _data(self):
        return self.view(np.ndarray)


# autograd needs to consider Parameter a class that in can compute gradients for
# in that regard, it behaves like an ordinary ndarray
ArrayBox.register(Parameter)
VSpace.register(Parameter, vspace_maker=VSpace.mappings[np.ndarray])


def relative_step(X, it, factor=0.1):
    return factor * X.mean(axis=0)

import autograd.numpy as np
from autograd.numpy.numpy_boxes import ArrayBox
from autograd.core import VSpace

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
        Statistical error estimate; set after optimization
    m: array-like
        First moment of the gradient; only set when optimized
        See Kingma & Ba (2015) and Reddi, Kale & Kumar (2018) for details
    v: array-like
        Second moment of the gradient; only set when optimized
        See Kingma & Ba (2015) and Reddi, Kale & Kumar (2018) for details
    vhat: array-like
        Maximal second moment of the gradient; only set when optimized
        See Kingma & Ba (2015) and Reddi, Kale & Kumar (2018) for details
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
        obj.fixed = fixed  # not functional right now
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.name = getattr(obj, "name", "unnamed")
        self.prior = getattr(obj, "prior", None)
        self.constraint = getattr(obj, "constraint", None)
        self.step = getattr(obj, "step", 0)
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

    @property
    def is_finite(self):
        """Check if parameter values are all finite
        """
        return np.isfinite(self._data).all()


# autograd needs to consider Parameter a class that in can compute gradients for
# in that regard, it behaves like an ordinary ndarray
ArrayBox.register(Parameter)
VSpace.register(Parameter, vspace_maker=VSpace.mappings[np.ndarray])


def prepare_param(X, name, fixed=True, step=None):
    if isinstance(X, Parameter):
        assert X.name == name
    else:
        if np.isscalar(X):
            X = (X,)
        X = Parameter(np.array(X, dtype="float"), name=name, fixed=fixed, step=step)
    return X


def relative_step(X, it, factor=0.1, minimum=0, axis=None):
    """Step size set at `factor` times the mean of `X` in direction `axis`
    """
    return np.maximum(minimum, factor * X.mean(axis=axis))

from functools import partial

import numpy as np
import proxmin

from . import interpolation
from . import operator
from . import measurement
from .bbox import trim
from .cache import Cache

class Constraint:
    """Constraint base class

    Constraints encode expected properties of the solution.
    Mathematically, they are the consequence of adding potentially
    non-differentiable penalty functions to the model fitting loss function.

    As we use proximal gradient methods, all constraints act as proxmimal
    operators, i.e. they need to have the following signature:

        f(X, step) -> X'

    where X' is the closest point to X that satisfies the feasibility criterion
    of the penalty function.

    For reference, every operator of the `proxmin` package yields a valid
    `Constraint`.
    """
    def __init__(self, f=None):
        """Constraint base class

        Parameters
        ----------
        f: proximal mapping
            Signature: f(X, step) -> X'
        """
        self.f = f

    def __call__(self, X, step):
        """Proximal mapping

        Parameters
        ----------
        X: array
            Optimimzation parameter
        step: float or array of same shape as X
            Step size for the proximal mapping

        Returns
        -------
        X': closest feasible match to X
        """
        if self.f is not None:
            return self.f(X, step)
        return X

class ConstraintChain:
    """A ordered list of `Constraint`s.

    Uses the concept of alternating projections onto convex sets to find
    solutions that are feasible according to a list of constraints.

    Parameters
    ----------
    constraints: list of `Constraint`
    repeat: int
        How often the constrain chain is repeated to ensure feasibility
    """
    def __init__(self, *constraints, repeat=1):
        assert isinstance(repeat, int) and repeat >= 1
        self.constraints = constraints
        self.repeat = repeat

    def __call__(self, X, step):
        for r in range(self.repeat):
            for c in self.constraints:
                X = c(X, step)
        return X

class PositivityConstraint(Constraint):
    """Allow only non-negative elements.
    """
    def __init__(self):
        super().__init__(proxmin.operators.prox_plus)

class NormalizationConstraint(Constraint):
    def __init__(self, type='sum'):
        """Normalize X to unity.

        Parameters
        ----------
        type: in ['sum', 'max']
            Whether the sum or the maximum is set to unity.
        """
        type = type.lower()
        assert type in ['sum', 'max']
        self.type = type

    def __call__(self, X, step):

        if self.type == "sum":
            X /= X.sum()
        else:
            X /= X.max()
        return X


def L0Constraint(Constraint):
    def __init__(self, thresh):
        """L0 norm (sparsity) penalty

        Parameters
        ----------
        thresh: regularization strength
        """
        super().__init__(partial(proxmin.operators.prox_hard, thresh=thresh))


def L1Constraint(Constraint):
    def __init__(self, thresh):
        """L1 norm (sparsity) penalty

        Parameters
        ----------
        thresh: regularization strength
        """
        super().__init__(partial(proxmin.operators.prox_soft, thresh=thresh))


def ThresholdConstraint(component):
    """Set a cutoff threshold for pixels below the noise

    Use the log histogram of pixel values to determine when the
    source is fitting noise. This function works well to prevent
    faint sources from growing large footprints but for large
    diffuse galaxies with a wide range of pixel values this
    does not work as well.

    The region that contains flux above the threshold is contained
    in `component.bboxes["thresh"]`.
    """
    def __call__(self, X, step):
        thresh, _bins = measurement.threshold(X)
        X[X < thresh] = 0

        # # TODO: bbox treatment must be at source level, not inside of the constraint
        # bbox = trim(component.morph)
        # if not hasattr(component, "bboxes"):
        #     component.bboxes = {}
        # component.bboxes["thresh"] = bbox
        return X

class MonotonicityConstraint(Constraint):
    """Make morphology monotonically decrease from the center

    See `~scarlet.operator.prox_monotonic`
    for a description of the other parameters.
    """
    def __init__(self, pixel_center, use_nearest=False, thresh=0, bbox=None):
        self.pixel_center = pixel_center
        self.use_nearest = use_nearest
        self.thresh = thresh
        self.bbox = bbox

    def __call__(self, X, step):

        if self.bbox is not None:
            # Only apply monotonicity to the pixels inside the bounding box
            morph = X[self.bbox.slices]
            shape = morph.shape
            if shape[0] <= 1 or shape[1] <= 1:
                return X
            center = self.pixel_center[0]-self.bbox.bottom, self.pixel_center[1]-self.bbox.left
        else:
            morph = X
            shape = X.shape[-2:]
            center = self.pixel_center
        morph = morph.copy()

        # get prox from the cache
        prox_name = "operator.prox_strict_monotonic"
        key = (shape, center, self.use_nearest, self.thresh)
        # The creation of this operator is expensive,
        # so load it from memory if possible.
        try:
            prox = Cache.check(prox_name, key)
        except KeyError:
            prox = operator.prox_strict_monotonic(shape, use_nearest=self.use_nearest,
                                                      thresh=self.thresh, center=center)

            Cache.set(prox_name, key, prox)

        # apply the prox
        prox(morph, step)

        # apply the bbox
        if self.bbox is not None:
            X[:] = np.zeros(X.shape, dtype=X.dtype)
            X[self.bbox.slices] = morph
        else:
            X[:] = morph

        return X


class SymmetryConstraint(Constraint):
    """Make the source symmetric about its center

    See `~scarlet.operator.prox_uncentered_symmetry`
    for a description of the parameters.
    """
    def __init__(self, pixel_center, bbox=None, fill=None, shift=None, strength=.5):
        self.pixel_center = pixel_center
        self.bbox = bbox
        self.fill = fill
        self.shift = shift
        self.strength = strength

    def __call__(self, X, step):
        if self.bbox is not None:
            # Only apply monotonicity to the pixels inside the bounding box
            morph = X[self.bbox.slices]
            shape = morph.shape
            if shape[0] <= 1 or shape[1] <= 1:
                return X
            center = self.pixel_center[0]-self.bbox.bottom, self.pixel_center[1]-self.bbox.left
        else:
            morph = X
            shape = X.shape[-2:]
            center = self.pixel_center
        morph = morph.copy()

        # apply the prox
        operator.prox_uncentered_symmetry(morph, step, center, "kspace", self.fill, self.shift, self.strength)

        # apply the bbox
        if self.bbox is not None:
            X[:] = np.zeros(X.shape, dtype=X.dtype)
            X[self.bbox.slices] = morph
        else:
            X[:] = morph

        return X

class CenterOnConstraint(Constraint):
    """Sets the center pixel to a tiny non-zero value
    """
    def __init__(self, pixel_center, tiny=1e-6):
        self.pixel_center = pixel_center
        self.tiny = tiny

    def __call__(self, X, step):
        X[tuple(self.pixel_center)] = self.tiny
        return X

class AllOnConstraint(Constraint):
    """Add to all elements a tiny non-zero value
    """
    def __init__(self, iny=1e-6):
        self.tiny = tiny

    def __call__(self, X, step):
        X += self.tiny
        return X

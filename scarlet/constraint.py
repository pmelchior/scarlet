import proxmin
from . import operator
from . import transformation
from .cache import Cache
from .config import Normalization
from functools import partial


class Constraint(object):
    """A constraint generator for SED and Morphology.
    """
    def __init__(self):
        """Constructor.
        """
        pass

    def prox_sed(self, shape):
        """Return the proximal operator to apply to the SED of a given `shape`.

        Parameters
        ----------
        shape: tuple of int
            The shape of the SED container.
        """
        return proxmin.operators.prox_id

    def prox_morph(self, shape):
        """Return the proximal operator to apply to the morphology of a given `shape`.

        Parameters
        ----------
        shape: tuple of int
            The shape of the morph container.
        """
        return proxmin.operators.prox_id

    def prox_g_sed(self, shape):
        """Return the proximal operator for the SED in the transformed domain.

        Parameters
        ----------
        shape: tuple of int
            The shape of the SED container.
        """
        return None # None or operator

    def prox_g_morph(self, shape):
        """Return the proximal operator for the morphology in the transformed domain.

        Parameters
        ----------
        shape: tuple of int
            The shape of the morph container.
        """
        return None

    def L_sed(self, shape):
        """Return the SED transformation matrix.

        Parameters
        ----------
        shape: tuple of int
            The shape of the SED container.
        """
        return None # None or matrix

    def L_morph(self, shape):
        """Return the morphology transformation matrix.

        Parameters
        ----------
        shape: tuple of int
            The shape of the morphology container.
        """
        return None

class MinimalConstraint(Constraint):
    """The minimal constraint for sources.

    This constraint uses a `proxmin.operators.prox_plus` constraint on the
    morphology (because the flux should always be positive), and a
    `proxmin.operators.prox_unity_plus` constraint on the SED
    (because the SED should always be positive and sum to unity).
    """
    def __init__(self, normalization=Normalization.A):
        self.normalization = normalization

    def prox_sed(self, shape):
        if norm == Normalization.S or norm == Normalization.Smax:
            return proxmin.operators.prox_plus
        return proxmin.operators.prox_unity_plus

    def prox_morph(self, shape):
        if self.normalization == Normalization.S:
            return partial(proxmin.operators.prox_unity_plus, axis=(0,1))
        elif self.normalization == Normalization.Smax:
            return proxmin.operators.AlternatingProjections([
                operator.prox_max,
                proxmin.operators.prox_plus,
            ])
        return proxmin.operators.prox_plus


class SimpleConstraint(Constraint):
    """Effective but still minimally restrictive constraint.

    SED positive and normalized to unity;
    morphology positive and with non-zero center.
    """
    def __init__(self, normalization=Normalization.A):
        self.normalization = normalization

    def prox_sed(self, shape):
        norm = self.normalization
        if norm == Normalization.S or norm == Normalization.Smax:
            return proxmin.operators.AlternatingProjections([
                operator.prox_sed_on, proxmin.operators.prox_plus
            ])
        return proxmin.operators.prox_unity_plus

    def prox_morph(self, shape):
        if self.normalization == Normalization.S:
            return proxmin.operators.AlternatingProjections([
                partial(proxmin.operators.prox_unity, axis=(0,1)),
                operator.prox_center_on,
                proxmin.operators.prox_plus,
            ])
        elif self.normalization == Normalization.Smax:
            return proxmin.operators.AlternatingProjections([
                operator.prox_max,
                operator.prox_center_on,
                proxmin.operators.prox_plus,
            ])
        return proxmin.operators.AlternatingProjections([
                operator.prox_center_on, proxmin.operators.prox_plus])

class L0Constraint(Constraint):
    """L0 sparsity penalty for the morphology
    """
    def __init__(self, thresh):
        """Initialize the constraint

        Parameters
        ----------
        thresh: float
            Threshold to use in `proxmin.operators.prox_hard`
        """
        self.thresh = thresh

    def prox_morph(self, shape):
        return partial(proxmin.operators.prox_hard, thresh=self.thresh)

class L1Constraint(Constraint):
    """L1 sparsity penalty for the morphology
    """
    def __init__(self, thresh):
        """Initialize the constraint

        Parameters
        ----------
        thresh: float
            Threshold to use in `proxmin.operators.prox_soft`
        """
        self.thresh = thresh

    def prox_morph(self, shape):
        return partial(proxmin.operators.prox_soft, thresh=self.thresh)

class DirectMonotonicityConstraint(Constraint):
    """Strict monotonicity constraint

    This creates a $prox_f$ constraint to the morphology that forces it
    to be montonically decreasing from the center.
    """
    def __init__(self, use_nearest=False, exact=False, thresh=0):
        """Initialize the constraint

        Parameters
        ----------
        use_nearest: bool
            If `use_nearest` is `True`, then the nearest pixel in a line between
            the current pixel in the peak is used as a reference.
            Otherwise (the default) a weighted average of all a pixels neighbors
            closer to the peak is used.
            If `exact` is `True`, this argument is ignored.
        exact: bool
            If `exact` is `True` then the exact (but *extremely slow*) monotonic operator is used.
            Otherwise `transformation.getRadialMonotonicOp` is used.
        thresh: float
            Minimum ratio between the current pixel and it's reference pixel.
            When `thresh=0` (default) a flat morphology is allowed.
        """
        self.use_nearest = use_nearest
        self.exact = exact
        self.thresh = thresh

    def prox_morph(self, shape):
        """Build the proximal operator

        Strict monotonicity depends on the shape of the source,
        so this function selects the proper one from a cache.
        """
        prox_name = "DirectMonotonicityConstraint.prox_morph"
        key = shape
        try:
            prox = Cache.check(prox_name, key)
        except KeyError:
            if not self.exact:
                prox = operator.prox_strict_monotonic(shape, use_nearest=self.use_nearest, thresh=self.thresh)
            else:
                # cone method for monotonicity: exact but VERY slow
                G = transformation.getRadialMonotonicOp(shape, useNearest=self.useNearest).toarray()
                prox = partial(operator.prox_cone, G=G)
            Cache.set(prox_name, key, prox)
        return prox

class MonotonicityConstraint(Constraint):
    """$prox_g$ monotonicity constraint

    A radial gradient that is required to monotonically decrease from the peak.
    A warning with this operator is that we have noticed that because monotonicity
    is not strictly enforced as a $prox_g$ operator, a single hot pixel that does not
    meet the constraint can cause pixels further from the peak to also monotonically
    decrease.
    """
    def __init__(self, use_nearest=False):
        """Initialize the constraint

        use_nearest: bool
            If `use_nearest` is `True`, then the nearest pixel in a line between
            the current pixel in the peak is used as a reference.
            Otherwise (the default) a weighted average of all a pixels neighbors
            closer to the peak is used.
        """
        self.use_nearest = use_nearest

    def prox_g_morph(self, shape):
        return proxmin.operators.prox_plus

    def L_morph(self, shape):
        return transformation.getRadialMonotonicOp(shape, useNearest=self.use_nearest)

class SymmetryConstraint(Constraint):
    """$prox_g$ symmetry constraint

    Enforces that the source is 180-degree rotation symmetric with respect to the peak.
    """

    def prox_g_morph(self, shape):
        return proxmin.operators.prox_zero

    def L_morph(self, shape):
        return transformation.getSymmetryOp(shape)

class DirectSymmetryConstraint(Constraint):
    """Soft symmetry constraint

    This creates a :math:`prox_f` constraint to the morphology that
    applies a symmetry constraint using a linear parameter `sigma`
    that can vary from `sigma=0` (no symmetry required) to
    `sigma=1` (perfect symmetry required).
    """
    def __init__(self, sigma=1):
        self.sigma = sigma

    def prox_morph(self, shape):
        return partial(operator.prox_soft_symmetry, sigma=self.sigma)

class TVxConstraint(Constraint):
    """Total Variation (TV) in X

    Penalty used in image processing to denoise an image
    (basically an L1 norm on the X-gradient).
    """
    def __init__(self, thresh=0):
        """Initialize the constraint

        Parameters
        ----------
        thresh: float
            Threshold to use in `proxmin.operators.prox_soft`
        """
        self.thresh = thresh

    def proxs_g_morph(self, shape):
        return partial(proxmin.operators.prox_soft, thresh=self.thresh)

    def L_morph(self, shape):
        name = "TVxConstraint.L_morph"
        key = shape
        try:
            return Cache.check(name, key)
        except KeyError:
            L = proxmin.transformation.get_gradient_x(shape, shape[1])
            _ = L.spectral_norm
            Cache.set(name, key, L)
            return L

class TVyConstraint(Constraint):
    """Total Variation (TV) in Y

    Penalty used in image processing to denoise an image
    (basically an L1 norm on the Y-gradient).
    """
    def __init__(self, thresh=0):
        """Initialize the constraint

        Parameters
        ----------
        thresh: float
            Threshold to use in `proxmin.operators.prox_soft`
        """
        self.thresh = thresh

    def proxs_g_morph(self, shape):
        return partial(proxmin.operators.prox_soft, thresh=self.thresh)

    def L_morph(self, shape):
        name = "TVyConstraint.L_morph"
        key = shape
        try:
            return Cache.check(name, key)
        except KeyError:
            L = proxmin.transformation.get_gradient_y(shape, shape[0])
            _ = L.spectral_norm
            Cache.set(name, key, L)
            return L


class ConstraintAdapter(object):
    """A constraint container for SED and Morphology of a :class:`~scarlet.source.Component`

    Using the adapter, the constraints are always evaluated with the current size
    of the component SED and morphology. It implements the same methods as
    `~scarlet.constraints.Constraint`, but without the `shape` argument.
    """
    def __init__(self, constraint, component):
        """Initialize the constraint adapter.

        Parameters
        ----------
        constraint: `~scarlet.constraints.Constraint` or list thereof
        component: `~scarlet.source.Component`
        """
        if isinstance(constraint, Constraint):
            _C = [constraint]
        else:
            _C = constraint
        if hasattr(_C, '__iter__') and all([isinstance(_c, Constraint) for _c in _C]):
            self.C = _C
        else:
            raise NotImplementedError("argument `constraint` must be constraint or list of constraints")
        self.component = component

    @property
    def prox_sed(self):
        ops = [c.prox_sed(self.component.sed.shape) for c in self.C if c.prox_sed(self.component.sed.shape) is not proxmin.operators.prox_id]
        if len(ops) == 0:
            return proxmin.operators.prox_id
        if len(ops) == 1:
            return ops[0]
        else:
            return proxmin.operators.AlternatingProjections(ops)

    @property
    def prox_morph(self):
        ops = [c.prox_morph(self.component.morph.shape) for c in self.C if c.prox_morph(self.component.morph.shape) is not proxmin.operators.prox_id]
        if len(ops) == 0:
            return proxmin.operators.prox_id
        if len(ops) == 1:
            return ops[0]
        else:
            return proxmin.operators.AlternatingProjections(ops)

    @property
    def prox_g_sed(self):
        return [c.prox_g_sed(self.component.sed.shape) for c in self.C if c.prox_g_sed(self.component.sed.shape) is not None]

    @property
    def prox_g_morph(self):
        return [c.prox_g_morph(self.component.morph.shape) for c in self.C if c.prox_g_morph(self.component.morph.shape) is not None]

    @property
    def L_sed(self):
        return [c.L_sed(self.component.sed.shape) for c in self.C if c.prox_g_sed(self.component.sed.shape) is not None]

    @property
    def L_morph(self):
        return [c.L_morph(self.component.morph.shape) for c in self.C if c.prox_g_morph(self.component.morph.shape) is not None]

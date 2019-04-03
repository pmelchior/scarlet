from enum import Enum
from functools import partial

from . import operator
from . import transformation
from . import utils
from .cache import Cache


from proxmin.operators import prox_id, prox_plus, prox_hard, prox_soft, AlternatingProjections


class Normalization(Enum):
    """Type of normalization to use for A and S

    Due to degeneracies in the product AS it is common to
    normalize one of the two matrices to unity. This
    enumerator is used to define which normalization is
    used to break this degeneracy.

    Attributes
    ----------
    A: 1
        Normalize the A (color) matrix to unity
    S: 2
        Normalize the S (morphology) matrix to unity
    Smax: 3
        Normalize S so that the maximum (peak) value is unity
    """
    A = 1
    S = 2
    Smax = 3


class Constraint(object):
    """A constraint generator for SED and Morphology.
    """
    def __init__(self):
        """Constructor.
        """
        pass

    def prox_sed(self, component):
        """Return the proximal operator to apply to the SED of a given `shape`.

        Parameters
        ----------
        shape: tuple of int
            The shape of the SED container.
        """
        return prox_id

    def prox_morph(self, component):
        """Return the proximal operator to apply to the morphology of a given `shape`.

        Parameters
        ----------
        shape: tuple of int
            The shape of the morph container.
        """
        return prox_id


class MinimalConstraint(Constraint):
    """The minimal constraint for sources.

    This constraint uses a `proxmin.operators.prox_plus` constraint on the
    morphology (because the flux should always be positive), and a
    `proxmin.operators.prox_unity_plus` constraint on the SED
    (because the SED should always be positive and sum to unity).
    """
    def __init__(self, normalization=Normalization.A):
        self.normalization = normalization

    def prox_sed(self, component):
        if self.normalization != Normalization.A:
            return prox_plus
        return operator.prox_unity_plus

    def prox_morph(self, component):
        if self.normalization == Normalization.S:
            return partial(operator.prox_unity_plus, axis=(0, 1))
        elif self.normalization == Normalization.Smax:
            return AlternatingProjections([
                operator.prox_max,
                prox_plus,
            ])
        return prox_plus


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

    def prox_morph(self, component):
        return partial(prox_hard, thresh=self.thresh)


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

    def prox_morph(self, component):
        return partial(prox_soft, thresh=self.thresh)


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

    def prox_morph(self, component):
        """Build the proximal operator

        Strict monotonicity depends on the shape of the source,
        so this function selects the proper one from a cache.
        """
        prox_name = "DirectMonotonicityConstraint.prox_morph"
        shape = component.shape[-2:]
        center = component.center
        key = (shape, center)
        try:
            prox = Cache.check(prox_name, key)
        except KeyError:
            if not self.exact:
                prox = operator.prox_strict_monotonic(shape, use_nearest=self.use_nearest,
                                                      thresh=self.thresh, center=center)
            else:
                raise NotImplementedError("Exact monotonicity is not currently supported")
                # cone method for monotonicity: exact but VERY slow
                G = transformation.getRadialMonotonicOp(shape, useNearest=self.use_nearest).L.toarray()
                prox = partial(operator.prox_cone, G=G)
            Cache.set(prox_name, key, prox)
        return prox


class DirectSymmetryConstraint(Constraint):
    """Soft symmetry constraint

    This creates a :math:`prox_f` constraint to the morphology that
    applies a symmetry constraint using a linear parameter `sigma`
    that can vary from `sigma=0` (no symmetry required) to
    `sigma=1` (perfect symmetry required).
    """
    def __init__(self, sigma=.5, use_soft=True):
        self.sigma = sigma
        self.use_soft = use_soft

    def prox_morph(self, component):
        return partial(operator.prox_uncentered_symmetry, sigma=self.sigma, use_soft=self.use_soft,
                       center=component.center)


class ThresholdConstraint(Constraint):
    """Threshold a component based on the noise in each band
    """
    def __init__(self, cutoffs=0):
        try:
            len(cutoffs)
        except TypeError:
            cutoffs = [cutoffs] * self.B
        self.cutoffs = cutoffs

    def prox_morph(self, component):
        model = component.get_model(trim=False)
        bbox = component.bbox
        if any([utils.flux_at_edge(model[b][bbox.slices], self.cutoffs[b]) for b in range(self.B)]):
            mask = 1-(model < self.cutoffs[:, None, None]).prod(dim=0)
            component.morph.data *= mask
            component._bbox = utils.trim(component.morph, 0)
        return prox_id


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
        ops = [c.prox_sed(self.component) for c in self.C if c.prox_sed(self.component) is not prox_id]
        if len(ops) == 0:
            return prox_id
        if len(ops) == 1:
            return ops[0]
        else:
            return AlternatingProjections(ops)

    @property
    def prox_morph(self):
        ops = [c.prox_morph(self.component) for c in self.C if c.prox_morph(self.component) is not prox_id]
        if len(ops) == 0:
            return prox_id
        if len(ops) == 1:
            return ops[0]
        else:
            return AlternatingProjections(ops)

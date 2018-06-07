import proxmin
from . import operators
from . import transformations
from functools import partial

# global cache to hold all complex proximal operators
cache = {}

def check_cache(name, key):
    global cache
    try:
        cache[name]
    except KeyError:
        cache[name] = {}

    return cache[name][key]

class Constraint(object):
    """A constraint generator for SED and Morphology.
    """
    def __init__(self):
        pass

    def prox_sed(self, shape):
        return proxmin.operators.prox_id

    def prox_morph(self, shape):
        return proxmin.operators.prox_id

    def prox_g_sed(self, shape):
        return None # None or operator

    def prox_g_morph(self, shape):
        return None

    def L_sed(self, shape):
        return None # None or matrix

    def L_morph(self, shape):
        return None

    def __and__(self, c):
        """Combine multiple constraints
        """
        if isinstance(c, Constraint):
            return [self, c]
        elif isinstance(c, list) and all([isinstance(c_, Constraint) for c_ in c]):
            return [self] + c
        else:
            raise NotImplementedError("second argument must be constraint or list of constraints")


class MinimalConstraint(Constraint):
    """The minimal constraint for sources.

    This constraint uses a `proxmin.operators.prox_plus` constraint on the
    morphology (because the flux should always be positive), and a
    `proxmin.operators.prox_unity_plus` constraint on the SED
    (because the SED should always be positive and sum to unity).
    """
    def prox_morph(self, shape):
        return proxmin.operators.prox_plus

    def prox_sed(self, shape):
        return proxmin.operators.prox_unity_plus

class SimpleConstraint(Constraint):
    """Effective but still minimally restrictive constraint.

    SED positive and normalized to unity;
    morphology positive and with non-zero center.
    """
    def prox_sed(self, shape):
        return proxmin.operators.prox_unity_plus

    def prox_morph(self, shape):
        return proxmin.operators.AlternatingProjections([
                operators.prox_center_on, proxmin.operators.prox_plus])

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
            Otherwise `transformations.getRadialMonotonicOp` is used.
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
            prox = check_cache(prox_name, key)
        except KeyError:
            if not self.exact:
                prox = operators.prox_strict_monotonic(shape, use_nearest=self.use_nearest, thresh=self.thresh)
            else:
                # cone method for monotonicity: exact but VERY slow
                G = transformations.getRadialMonotonicOp(shape, useNearest=self.useNearest).toarray()
                prox = partial(operators.prox_cone, G=G)
            cache[prox_name][key] = prox
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
        return transformations.getRadialMonotonicOp(shape, useNearest=self.use_nearest)

class SymmetryConstraint(Constraint):
    """$prox_g$ symmetry constraint

    Requires that the source is symmetric about the peak
    """

    def prox_g_morph(self, shape):
        return proxmin.operators.prox_zero

    def L_morph(self, shape):
        return transformations.getSymmetryOp(shape)

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
        return partial(operators.prox_soft_symmetry, sigma=self.sigma)

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
            return check_cache(name, key)
        except KeyError:
            L = proxmin.transformations.get_gradient_x(shape, shape[1])
            cache[name][key] = L
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
            return check_cache(name, key)
        except KeyError:
            L = proxmin.transformations.get_gradient_y(shape, shape[0])
            cache[name][key] = L
            return L


class ConstraintAdapter(object):
    """A constraint container for SED and Morphology of a :class:`~scarlet.source.Source`
    """
    def __init__(self, C, source):
        """Initialize the constraint adapter.
        """
        if isinstance(C, Constraint) or (isinstance(C, list) and all([isinstance(c_, Constraint) for c_ in C])):
            self.C = C
        else:
            raise NotImplementedError("argument `C` must be constraint or list of constraints")
        self.source = source

    @property
    def prox_sed(self):
        if not isinstance(self.C, list):
            return self.C.prox_sed(self.source.sed[0].shape)
        else:
            return proxmin.operators.AlternatingProjections([c.prox_sed(self.source.sed[0].shape) for c in self.C])

    @property
    def prox_morph(self):
        if not isinstance(self.C, list):
            return self.C.prox_morph(self.source.morph[0].shape)
        else:
            return proxmin.operators.AlternatingProjections([c.prox_morph(self.source.morph[0].shape) for c in self.C])

    @property
    def prox_g_sed(self):
        if not isinstance(self.C, list):
            return self.C.prox_g_sed(self.source.sed[0].shape)
        else:
            return [ c.prox_g_sed(self.source.sed[0].shape) for c in self.C if c.prox_g_sed(self.source.sed[0].shape) is not None ]

    @property
    def prox_g_morph(self):
        if not isinstance(self.C, list):
            return self.C.prox_g_morph(self.source.morph[0].shape)
        else:
            return [ c.prox_g_morph(self.source.morph[0].shape) for c in self.C if c.prox_g_morph(self.source.morph[0].shape) is not None ]

    @property
    def L_sed(self):
        if not isinstance(self.C, list):
            return self.C.L_sed(self.source.sed[0].shape)
        else:
            return [ c.L_sed(self.source.sed[0].shape) for c in self.C if c.prox_g_sed(self.source.sed[0].shape) is not None ]

    @property
    def L_morph(self):
        if not isinstance(self.C, list):
            return self.C.L_morph(self.source.morph[0].shape)
        else:
            return [c.L_morph(self.source.morph[0].shape) for c in self.C if c.prox_g_morph(self.source.morph[0].shape) is not None ]

    def __repr__(self):
        return repr(self.C)

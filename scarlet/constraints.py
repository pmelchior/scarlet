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

class ConstraintAdapter(object):
    """A constraint container for SED and Morphology of a :class:`~scarlet.source.Source`
    """
    def __init__(self, C, source):
        """Initialize the constraint adapter.
        """
        self.C = C
        self.source = source

    @property
    def prox_sed(self):
        return self.C.prox_sed(self.source.sed[0].shape)

    @property
    def prox_morph(self):
        return self.C.prox_morph(self.source.morph[0].shape)

    @property
    def prox_g_sed(self):
        return self.C.prox_g_sed(self.source.sed[0].shape)

    @property
    def prox_g_morph(self):
        return self.C.prox_g_morph(self.source.morph[0].shape)

    @property
    def L_sed(self):
        return self.C.L_sed(self.source.sed[0].shape)

    @property
    def L_morph(self):
        return self.C.L_morph(self.source.morph[0].shape)

    def __repr__(self):
        return repr(self.C)

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
        """Combine two constraints
        """
        if isinstance(c, Constraint):
            return ConstraintList([self, c])
        else:
            raise NotImplementedError


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

class ConstraintList:
    """List of `Constraint` objects

    In general a single component of a source might have multiple
    constraints.
    Some of these might be $prox_f$ strict constraints while others may be
    $prox_g$ linear constraints, which can both apply to either the
    SED or morphology of the component.
    The `ConstraintList` contains all of the constraints on a single component.
    """
    def __init__(self, constraints, repeat=1):
        self.prox_sed = None # might be None, single operator, or AlternatingProjections
        self.prox_morph = None
        self.prox_g_sed = None # None or list of operators
        self.prox_g_morph = None
        self.L_sed = None # None or list of matrices
        self.L_morph = None
        self.constraints = []
        self.repeat = repeat
        for c in constraints:
            self.__iand__(c)

    def __getitem__(self, index):
        """Get the `Constraint` at index `index`.
        """
        return self.constraints[index]

    def __and__(self, c):
        """Combine multiple constraints
        """
        cl = ConstraintList(self.constraints)
        return cl.__iand__(c)

    def __iand__(self, c):
        """Combine multiple constraints
        """
        if isinstance(c, ConstraintList):
            for _c in c.constraints:
                self.__iand__(_c)
            return self

        self.constraints.append(c)
        self._update_projections(c, 'prox_sed')
        self._update_projections(c, 'prox_morph')
        self._update_constraint_list(c, 'prox_g_sed')
        self._update_constraint_list(c, 'prox_g_morph')
        self._update_constraint_list(c, 'L_sed')
        self._update_constraint_list(c, 'L_morph')
        return self

    def _update_projections(self, constraint, prox_name):
        """Update $prox_f$ constraints

        When daisy chaining multiple $prox_f$ constraints,
        which might not commute, we use a
        `proxmin.operators.AlternatingProjections` to combine the
        constraints such that they are alternated in each step of the
        minimization.
        """
        prox = getattr(self, prox_name)
        cprox = getattr(constraint, prox_name)
        if cprox is not None:
            if prox is None or prox is proxmin.operators.prox_id:
                prox = cprox
            elif isinstance(prox, proxmin.operators.AlternatingProjections) is False:
                # self.<prox_name> is single operator
                if isinstance(cprox, proxmin.operators.AlternatingProjections):
                    ops = [prox] + cprox.operators
                else:
                    ops = [prox, cprox]
                prox = proxmin.operators.AlternatingProjections(ops, repeat=self.repeat)
            else:
                # self.<prox_name> is AlternatingProjections
                if isinstance(cprox, proxmin.operators.AlternatingProjections):
                    ops = prox.operators + cprox.operators
                else:
                    ops = prox.operators + [cprox]
                prox = proxmin.operators.AlternatingProjections(ops, repeat=self.repeat)
        setattr(self, prox_name, prox)

    def _update_constraint_list(self, constraint, key):
        """Combine individual constraints

        Each `Constraint` can contain multiple types of constraints.
        When multiple `Constraint`s are combined, this method updates
        the appropriate constraints in the list
        """
        if hasattr(constraint, key):
            clist = getattr(self, key)
            c = getattr(constraint, key)
            if c is not None:
                if clist is None:
                    clist = [c]
                else:
                    clist.append(c)
                setattr(self, key, clist)

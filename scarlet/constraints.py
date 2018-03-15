import proxmin
from . import operators
from . import transformations
from functools import partial

class Constraint(object):
    """A constraint on either the SED or Morphology of a :class:`~scarlet.source.Source`
    """
    def __init__(self):
        """Initialize the properties

        All `Constraint` objects have the properties
        prox_sed, prox_morph, prox_g_sed, prox_g_morph,
        L_sed, L_morph all set to None
        """
        self.smooth = False # Only true for smooth, convex constraints (as opposed to prox)
        self.prox_sed = None  # None, single operator, or AlternatingProjections
        self.prox_morph = None
        self.prox_g_sed = None # None or operator
        self.prox_g_morph = None
        self.L_sed = None      # None or matrix
        self.L_morph = None

    def reset(self, source):
        """Action to perform when the Constraint is reset

        No action is performed in the base class, but inherited classes may
        utilize this method, typically when the `Constraint`
        depends on the shape of the image or source,
        which may not be known when the `Constraint` is initialized.
        """
        pass

    def gradient(self, blend, m, l, block):
        """Return the gradient of a smooth, convex, constraint

        Some constraints are smooth and convex, so instead of using the
        machinery of proximal operators that are not needed, it's gradient
        can be added directly to the likelihood.
        """
        return 0

    def lipschitz_const(self, blend, m, l, block):
        r"""Lipschitz constant of the constraint
        
        For a function `f` that maps :math:`f[a,b] \rightarrow \mathbb{R}`,
        the Lipschitz constant `L` is the smallest constant that obeys the
        Lipschitz condition

        .. math::
            |f(x_2)-f(x_1)| \leq L_f |x_1-x_2|

        which, for for a function that is differentiable everywhere with
        a bounded derivative is
        
        .. math::
            \substack{\textrm{sup} \\ x} |f'(x)|

        For non-smooth or non-convex constraints that require
        proximal operators, the Lipschitz constant is not needed
        and should return `0`.
        """
        return 0

    def __and__(self, c):
        """Combine two constraints
        """
        if isinstance(c, Constraint):
            return ConstraintList([self, c])
        else:
            raise NotImplementedError

class SmoothSymmetryConstraint(Constraint):
    r"""Smooth and convex symmetry constraint

    This class implements symmetry as a smooth convex constraint added
    to the likelihood:
    
    .. math::
        \frac{\sigma}{2} |\Sigma S|_2^2
    
    where :math:`\Sigma \equiv S-S^\dagger` and :math:`S^\dagger`
    is the symmetric version of `S`.
    """
    def __init__(self, sigma=0.1):
        """Initialize the Constraint

        Parameters
        ----------
        sigma: float
            Strength of the constraint
        """
        if sigma<0 or sigma>1:
            raise ValueError("`sigma` must be between 0 and 1")
        super(SmoothSymmetryConstraint, self).__init__()
        self.sigma = sigma
        self.smooth = True

    def gradient(self, blend, m, l, block):
        """Calculate the symmetry gradient
        """
        if block == 0:
            return 0
        return self._symmetry_grad.dot(blend.sources[m].morph[l])

    def lipschitz_const(self, blend, m, l, block):
        """Calculate the Lipschitz constant

        See `Constraint` class for a definition.
        """
        if block == 0:
            return 0
        return 2*self.sigma

    def reset(self, source, sigma=None):
        """Build the symmetry operator

        Symmetry depends on the shape of the source,
        so it cannot be built until after the `source` has been created
        with a bounding box.
        """
        if sigma is not None:
            self.sigma = sigma
        shape = source.shape[1:]
        _symmetry_op = transformations.getSymmetryOp(shape)
        self._symmetry_grad = 0.5 * self.sigma * _symmetry_op.T*_symmetry_op

class MinimalConstraint(Constraint):
    """The minimal constraint required for the result to make sense
    
    This constraint uses a `proxmin.operators.prox_plus` constraint on the
    morphology (because the flux should always be positive), and a
    `proxmin.operators.prox_unity_plus` constraint on the SED
    (because the SED should always be positive and sum to unity).
    """
    def __init__(self):
        super(MinimalConstraint, self).__init__()
        self.prox_morph = proxmin.operators.prox_plus
        self.prox_sed = proxmin.operators.prox_unity_plus

class PositivityConstraint(Constraint):
    """Constrain the morphology to always be positive

    This is slightly more strict than non-negative, where the
    `operators.prox_center_on` proximal operator forces the central
    pixel of the morphology to have a tiny amount of flux, which is necessary for
    the recentering algorithm.
    """
    def __init__(self):
        super(PositivityConstraint, self).__init__()
        self.prox_morph = proxmin.operators.AlternatingProjections([
            operators.prox_center_on, proxmin.operators.prox_plus])

class SimpleConstraint(PositivityConstraint):
    """Constrain the SED to be positive and sum to unity
    """
    def __init__(self):
        super(SimpleConstraint, self).__init__()
        self.prox_sed = proxmin.operators.prox_unity_plus

class L0Constraint(Constraint):
    """Add an L0 sparsity penalty to the morphology
    """
    def __init__(self, thresh):
        """Initialize the constraint

        Parameters
        ----------
        thresh: float
            Threshold to use in `proxmin.operators.prox_hard`
        """
        super(L0Constraint, self).__init__()
        self.prox_morph = partial(proxmin.operators.prox_hard, thresh=thresh)

class L1Constraint(Constraint):
    """Add an L1 sparsity penalty to the morphology
    """
    def __init__(self, thresh):
        """Initialize the constraint

        Parameters
        ----------
        thresh: float
            Threshold to use in `proxmin.operators.prox_soft`
        """
        super(L1Constraint, self).__init__()
        self.prox_morph = partial(proxmin.operators.prox_soft, thresh=thresh)

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
        super(DirectMonotonicityConstraint, self).__init__()
        self.use_nearest = use_nearest
        self.exact = exact
        self.thresh = thresh
        # lazy initialization: wait for the reset to set the source size

    def reset(self, source):
        """Build the proximal operator

        Strict monotonicity depends on the shape of the source,
        so it cannot be built until after the `source` has been created
        with a bounding box.
        """
        shape = source.shape[1:]
        if not self.exact:
            self.prox_morph = operators.prox_strict_monotonic(shape, use_nearest=self.use_nearest,
                                                              thresh=self.thresh)
        else:
            # cone method for monotonicity: exact but VERY slow
            G = transformations.getRadialMonotonicOp(shape, useNearest=self.useNearest).toarray()
            self.prox_morph = partial(operators.prox_cone, G=G)

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
        super(MonotonicityConstraint, self).__init__()
        # positive radial gradients:
        self.prox_g_morph = proxmin.operators.prox_plus
        # lazy initialization: wait for the reset to set the source size
        self.use_nearest = use_nearest

    def reset(self, source):
        """Build the proximal operator

        Monotonicity depends on the shape of the source,
        so it cannot be built until after the `source` has been created
        with a bounding box.
        """
        shape = source.shape[1:]
        self.L_morph = transformations.getRadialMonotonicOp(shape, useNearest=self.use_nearest)

class SoftSymmetryConstraint(Constraint):
    """Soft symmetry constraint

    This creates a :math:`prox_f` constraint to the morphology that
    applies a symmetry constraint using a linear parameter `sigma`
    that can vary from `sigma=0` (no symmetry required) to
    `sigma=1` (perfect symmetry required).
    """
    def __init__(self, sigma=1):
        super(SoftSymmetryConstraint, self).__init__()
        self.sigma = sigma

    def reset(self, source):
        """Build the proximal operator

        Symmetry depends on the shape of the source,
        so it cannot be built until after the `source` has been created
        with a bounding box.
        """
        shape = source.shape[1:]
        self.prox_morph = partial(operators.prox_soft_symmetry, sigma=self.sigma)

class SymmetryConstraint(Constraint):
    """$prox_g$ symmetry constraint

    Requires that the source is symmetric about the peak
    """
    def __init__(self):
        super(SymmetryConstraint, self).__init__()
        self.prox_g_morph = proxmin.operators.prox_zero
        # lazy initialization: wait for the reset to set the source size

    def reset(self, source):
        """Build the proximal operator

        Symmetry depends on the shape of the source,
        so it cannot be built until after the `source` has been created
        with a bounding box.
        """
        shape = source.shape[1:]
        self.L_morph = transformations.getSymmetryOp(shape)

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
        super(DirectMonotonicityConstraint, self).__init__()
        self.proxs_g_morph = partial(proxmin.operators.prox_soft, thresh=thresh)
        # lazy initialization: wait for the reset to set the source size

    def reset(self, source):
        shape = source.shape[1:]
        self.L_morph = proxmin.transformations.get_gradient_x(shape, source.Nx)

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
        super(DirectMonotonicityConstraint, self).__init__()
        self.proxs_g_morph = partial(proxmin.operators.prox_soft, thresh=thresh)
        # lazy initialization: wait for the reset to set the source size

    def reset(self, source):
        shape = source.shape[1:]
        self.L_morph = proxmin.transformations.get_gradient_y(shape, source.Ny)


class ConstraintList:
    """List of `Constraint` objects

    In general a single component of a source might have multiple
    constraints.
    Some of these might be $prox_f$ strict constraints while others may be
    $prox_g$ linear constraints, which can both apply to either the
    SED or morphology of the component.
    The `ConstraintList` contains all of the constraints on a single component.
    """
    def __init__(self, constraints):
        self.smooth = False
        self.smooth_constraints = []
        self.prox_sed = None # might be None, single operator, or AlternatingProjections
        self.prox_morph = None
        self.prox_g_sed = None # None or list of operators
        self.prox_g_morph = None
        self.L_sed = None # None or list of matrices
        self.L_morph = None
        self.constraints = []
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
        self._update_smooth_list(c)
        return self

    def reset(self, source):
        for c in self.constraints:
            c.reset(source)
        self.__init__(self.constraints)

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
            if prox is None:
                prox = cprox
            elif isinstance(prox, proxmin.operators.AlternatingProjections) is False:
                # self.<prox_name> is single operator
                if isinstance(cprox, proxmin.operators.AlternatingProjections):
                    ops = [prox] + cprox.operators
                else:
                    ops = [prox, cprox]
                prox = proxmin.operators.AlternatingProjections(ops)
            else:
                # self.<prox_name> is AlternatingProjections
                if isinstance(cprox, proxmin.operators.AlternatingProjections):
                    ops = prox.operators + cprox.operators
                else:
                    ops = prox.operators + [cprox]
                prox = proxmin.operators.AlternatingProjections(ops)
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

    def _update_smooth_list(self, constraint):
        """Add a new smooth, convex constraint to the likelihood
        """
        if constraint.smooth:
            self.smooth_constraints.append(constraint)
            self.smooth = True

    def gradient(self, blend, m, l, block):
        """Apply any smooth and convex constraints
        """
        grad = 0
        for c in self.smooth_constraints:
            grad += c.gradient(blend, m, l, block)
        return grad

    def lipschitz_const(self, blend, m, l, block):
        """Calculate the Lipschitz constant

        See `Constraint` class for a better description.
        """
        lipschitz_const = 0
        for c in self.smooth_constraints:
            lipschitz_const += c.lipschitz_const(blend, m, l, block)
        return lipschitz_const
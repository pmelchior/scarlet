import proxmin
from . import operators
from . import transformations
from functools import partial

class Constraint(object):
    """A constraint on either the SED or Morphology of a :class:`~scarlet.source.Source`
    """
    def __init__(self):
        self.prox_sed = None  # None, single operator, or AlternatingProjections
        self.prox_morph = None
        self.prox_g_sed = None # None or operator
        self.prox_g_morph = None
        self.L_sed = None      # None or matrix
        self.L_morph = None

    def reset(self, source):
        pass

    def __and__(self, c):
        if isinstance(c, Constraint):
            return ConstraintList([self, c])
        else:
            raise NotImplementedError

class MinimalConstraint(Constraint):
    def __init__(self):
        super(MinimalConstraint, self).__init__()
        self.prox_morph = proxmin.operators.prox_plus
        self.prox_sed = proxmin.operators.prox_unity_plus

class PositivityConstraint(Constraint):
    def __init__(self):
        super(PositivityConstraint, self).__init__()
        self.prox_morph = proxmin.operators.AlternatingProjections([operators.prox_center_on, proxmin.operators.prox_plus])

class SimpleConstraint(PositivityConstraint):
    def __init__(self):
        super(SimpleConstraint, self).__init__()
        self.prox_sed = proxmin.operators.prox_unity_plus

class L0Constraint(Constraint):
    def __init__(self, thresh):
        super(L0Constraint, self).__init__()
        self.prox_morph = partial(proxmin.operators.prox_hard, thresh=thresh)

class L1Constraint(Constraint):
    def __init__(self, thresh):
        super(L1Constraint, self).__init__()
        self.prox_morph = partial(proxmin.operators.prox_soft, thresh=thresh)

class DirectMonotonicityConstraint(Constraint):
    def __init__(self, use_nearest=False, exact=False, thresh=0):
        super(DirectMonotonicityConstraint, self).__init__()
        self.use_nearest = use_nearest
        self.exact = exact
        self.thresh = thresh
        # lazy initialization: wait for the reset to set the source size

    def reset(self, source):
        shape = source.shape[1:]
        if not self.exact:
            self.prox_morph = operators.prox_strict_monotonic(shape, use_nearest=self.use_nearest, thresh=self.thresh)
        else:
            # cone method for monotonicity: exact but VERY slow
            G = transformations.getRadialMonotonicOp(shape, useNearest=self.useNearest).toarray()
            self.prox_morph = partial(operators.prox_cone, G=G)

class MonotonicityConstraint(Constraint):
    def __init__(self, use_nearest=False):
        super(MonotonicityConstraint, self).__init__()
        # positive radial gradients:
        self.prox_g_morph = proxmin.operators.prox_plus
        # lazy initialization: wait for the reset to set the source size
        self.use_nearest = use_nearest

    def reset(self, source):
        shape = source.shape[1:]
        self.L_morph = transformations.getRadialMonotonicOp(shape, useNearest=self.use_nearest)

class SymmetryConstraint(Constraint):
    def __init__(self):
        super(SymmetryConstraint, self).__init__()
        self.prox_g_morph = proxmin.operators.prox_zero
        # lazy initialization: wait for the reset to set the source size

    def reset(self, source):
        shape = source.shape[1:]
        self.L_morph = transformations.getSymmetryOp(shape)

class TVxConstraint(Constraint):
    def __init__(self, thresh=0):
        super(DirectMonotonicityConstraint, self).__init__()
        self.proxs_g_morph = partial(proxmin.operators.prox_soft, thresh=thresh)
        # lazy initialization: wait for the reset to set the source size

    def reset(self, source):
        shape = source.shape[1:]
        self.L_morph = proxmin.transformations.get_gradient_x(shape, source.Nx)

class TVyConstraint(Constraint):
    def __init__(self, thresh=0):
        super(DirectMonotonicityConstraint, self).__init__()
        self.proxs_g_morph = partial(proxmin.operators.prox_soft, thresh=thresh)
        # lazy initialization: wait for the reset to set the source size

    def reset(self, source):
        shape = source.shape[1:]
        self.L_morph = proxmin.transformations.get_gradient_y(shape, source.Ny)


class ConstraintList:
    def __init__(self, constraints):
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
        return self.constraints[index]

    def __and__(self, c):
        cl = ConstraintList(self.constraints)
        return cl.__iand__(c)

    def __iand__(self, c):
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

    def reset(self, source):
        for c in self.constraints:
            c.reset(source)
        self.__init__(self.constraints)

    def _update_projections(self, constraint, prox_name):
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
        if hasattr(constraint, key):
            clist = getattr(self, key)
            c = getattr(constraint, key)
            if c is not None:
                if clist is None:
                    clist = [c]
                else:
                    clist.append(c)
                setattr(self, key, clist)

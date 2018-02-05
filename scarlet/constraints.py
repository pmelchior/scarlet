import proxmin
from . import operators
from . import transformations
from functools import partial

class Constraint(object):
    def __init__(self):
        self.prox_sed = None # might be None, single operator, or AlternatingProjections
        self.prox_morph = None
        self.prox_g_sed = None
        self.prox_g_morph = None
        self.L_sed = None
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
            self.prox_morphh = partial(operators.prox_cone, G=G)

class MonotonicityConstraint(Constraint):
    def __init__(self, use_nearest=False):
        super(MonotonicityConstraint, self).__init__()
        # positive radial gradients:
        self.prox_g_morph = proxmin.operators.prox_plus
        # lazy initialization: wait for the reset to set the source size
        self.use_nearest = use_nearest

    def resize(self, source):
        shape = source.shape[1:]
        self.L_morph = transformations.getRadialMonotonicOp(shape, useNearest=self.use_nearest)

class SymmetryConstraint(Constraint):
    def __init__(self):
        super(SymmetryConstraint, self).__init__()
        self.prox_g_morph = proxmin.operators.prox_zero
        # lazy initialization: wait for the reset to set the source size

    def resize(self, source):
        shape = source.shape[1:]
        self.L_morph = transformations.getSymmetryOp(shape)

class TVxConstraint(Constraint):
    def __init__(self, thresh=0):
        super(DirectMonotonicityConstraint, self).__init__()
        self.proxs_g_morph = partial(proxmin.operators.prox_soft, thresh=thresh)
        # lazy initialization: wait for the reset to set the source size

    def resize(self, source):
        shape = source.shape[1:]
        self.L_morph = proxmin.transformations.get_gradient_x(shape, source.Nx)

class TVyConstraint(Constraint):
    def __init__(self, thresh=0):
        super(DirectMonotonicityConstraint, self).__init__()
        self.proxs_g_morph = partial(proxmin.operators.prox_soft, thresh=thresh)
        # lazy initialization: wait for the reset to set the source size

    def resize(self, source):
        shape = source.shape[1:]
        self.L_morph = proxmin.transformations.get_gradient_y(shape, source.Ny)


class ConstraintList:
    def __init__(self, constraints):
        self.prox_sed = None # might be None, single operator, or AlternatingProjections
        self.prox_morph = None
        self.prox_g_sed = None
        self.prox_g_morph = None
        self.L_sed = None
        self.L_morph = None
        self.constraints = []
        for c in constraints:
            self.__iand__(c)

    def __and__(self, c):
        cl = ConstraintList(self.constraints)
        return cl.__iand__(c)

    def __iand__(self, c):
        if isinstance(c, ConstraintList):
            for _c in c.constraints:
                self.__iand__(_c)
            return self

        self.constraints.append(c)
        if c.prox_sed is not None:
            if self.prox_sed is None:
                self.prox_sed = c.prox_sed
            elif isinstance(self.prox_sed, proxmin.operators.AlternatingProjections) is False:
                # self.prox_sed is single operator
                if isinstance(c.prox_sed, proxmin.operators.AlternatingProjections):
                    ops = [self.prox_sed] + c.prox_sed.operators
                else:
                    ops = [self.prox_sed, c.prox_sed]
                self.prox_sed = proxmin.operators.AlternatingProjections(ops)
            else:
                # self.prox_sed is AlternatingProjections
                if isinstance(c.prox_sed, proxmin.operators.AlternatingProjections):
                    self.prox_sed.operators += c.prox_sed.operators
                else:
                    self.prox_sed.operators.append(c.prox_sed)

        if c.prox_morph is not None:
            if self.prox_morph is None:
                self.prox_morph = c.prox_morph
            elif isinstance(self.prox_morph, proxmin.operators.AlternatingProjections) is False:
                # self.prox_morph is single operator
                if isinstance(c.prox_morph, proxmin.operators.AlternatingProjections):
                    ops = [self.prox_morph] + c.prox_morph.operators
                else:
                    ops = [self.prox_morph, c.prox_morph]
                self.prox_morph = proxmin.operators.AlternatingProjections(ops)
            else:
                # self.prox_sed is AlternatingProjections
                if isinstance(c.prox_morph, proxmin.operators.AlternatingProjections):
                    self.prox_morph.operators += c.prox_morph.operators
                else:
                    self.prox_morph.operators.append(c.prox_morph)

        if c.prox_g_sed is not None:
            if self.prox_g_sed is None:
                self.prox_g_sed = [c.prox_g_sed]
            else:
                self.prox_g_sed.append(c.prox_g_sed)

        if c.prox_g_morph is not None:
            if self.prox_g_morph is None:
                self.prox_g_morph = [c.prox_g_morph]
            else:
                self.prox_g_morph.append(c.prox_g_morph)

        if c.L_sed is not None:
            if self.L_sed is None:
                self.L_sed = [c.L_sed]
            else:
                self.L_sed.append(c.L_sed)

        if c.L_morph is not None:
            if self.L_morph is None:
                self.L_morph = [c.L_morph]
            else:
                self.L_morph.append(c.L_morph)
        return self

    def reset(self, source):
        for c in self.constraints:
            c.reset(source)
        self.__init__(self.constraints)

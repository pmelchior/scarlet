import autograd.numpy as np
from autograd.numpy.numpy_boxes import ArrayBox
from autograd.core import VSpace

class Parameter(np.ndarray):
    def __new__(cls, array, prior=None, constraint=None, step=0, converged=False, fixed=False, **kwargs):
        obj = np.asarray(array, dtype=array.dtype).view(cls)
        obj.prior = prior
        obj.constraint = constraint
        obj.step = step
        obj.converged = converged
        obj.fixed = fixed
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        self.prior = getattr(obj, 'prior', None)
        self.constraint = getattr(obj, 'constraint', None)
        self.step = getattr(obj, 'step_size', 0)
        self.converged = getattr(obj, 'converged', False)
        self.fixed = getattr(obj, 'fixed', False)

    @property
    def _data(self):
        return self.view(np.ndarray)

ArrayBox.register(Parameter)
VSpace.register(Parameter, vspace_maker=VSpace.mappings[np.ndarray])

default_step = lambda X, it: 0.1*X.mean(axis=0)

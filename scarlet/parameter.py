import autograd.numpy as np
from autograd.numpy.numpy_boxes import ArrayBox
from autograd.core import VSpace
from functools import partial

class Parameter(np.ndarray):
    def __new__(cls, array, prior=None, constraint=None, step=0, converged=False, std=None, fixed=False, **kwargs):
        obj = np.asarray(array, dtype=array.dtype).view(cls)
        obj.prior = prior
        obj.constraint = constraint
        obj.step = step
        obj.converged = converged
        obj.std = std
        obj.fixed = fixed
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        self.prior = getattr(obj, 'prior', None)
        self.constraint = getattr(obj, 'constraint', None)
        self.step = getattr(obj, 'step_size', 0)
        self.converged = getattr(obj, 'converged', False)
        self.std = getattr(obj, 'std', None)
        self.fixed = getattr(obj, 'fixed', False)

    @property
    def _data(self):
        return self.view(np.ndarray)

ArrayBox.register(Parameter)
VSpace.register(Parameter, vspace_maker=VSpace.mappings[np.ndarray])

relative_step = lambda X, it, factor: factor*X.mean(axis=0)
default_step = partial(relative_step, factor=0.1)

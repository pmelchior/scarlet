import autograd.numpy as np
from functools import partial

from .model import Model


class Spectrum(Model):
    pass


class TabulatedSpectrum(Spectrum):
    def __init__(self, frame, spectrum, bbox=None, **kwargs):
        if isinstance(spectrum, Parameter):
            assert spectrum.name == "spectrum"
        else:
            constraint = PositivityConstraint()
            step = partial(relative_step, factor=1e-2)
            spectrum = Parameter(
                spectrum, name="spectrum", step=step, constraint=constraint
            )

        if bbox is None:
            assert frame.bbox[0].shape == spectrum.shape
            bbox = Box(spectrum.shape)
        else:
            assert bbox.shape == spectrum.shape

        super().__init__(frame, spectrum, bbox=bbox, **kwargs)

    def get_model(self, *parameters):
        spectrum = self.get_parameter(0, *parameters)
        return spectrum

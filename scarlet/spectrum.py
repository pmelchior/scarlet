from functools import partial

from .model import Model
from .frame import Frame
from .bbox import Box
from .parameter import Parameter, relative_step
from .constraint import PositivityConstraint
from .bbox import Box

# make sure that import * above doesn't import its own stock numpy
import autograd.numpy as np


class Spectrum(Model):
    def __init__(self, frame, *parameters, bbox=None):
        assert isinstance(frame, Frame)
        self.frame = frame
        assert isinstance(bbox, Box)
        self.bbox = bbox
        super().__init__(*parameters)


class TabulatedSpectrum(Spectrum):
    def __init__(self, frame, spectrum, bbox=None):
        if isinstance(spectrum, Parameter):
            assert spectrum.name == "spectrum"
        else:
            constraint = PositivityConstraint(threshold=1e-20)
            step = partial(relative_step, factor=1e-2)
            spectrum = Parameter(
                spectrum, name="spectrum", step=step, constraint=constraint
            )

        if bbox is None:
            assert frame.bbox[0].shape == spectrum.shape
            bbox = Box(spectrum.shape)
        else:
            assert bbox.shape == spectrum.shape

        super().__init__(frame, spectrum, bbox=bbox)

    def get_model(self, *parameters):
        spectrum = self.get_parameter(0, *parameters)
        return spectrum

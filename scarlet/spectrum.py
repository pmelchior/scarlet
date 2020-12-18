import autograd.numpy as np
from functools import partial

from .bbox import Box
from .constraint import PositivityConstraint
from .frame import Frame
from .model import Model
from .parameter import Parameter, relative_step


class Spectrum(Model):
    """Spectrum base class

    The class describes the 1D spectral dependence of`~scarlet.FactorizedComponent`.

    Parameters
    ----------
    frame: `~scarlet.Frame`
        Characterization of the model
    parameters: list of `~scarlet.Parameter`
    bbox: `~scarlet.Box`
        2D bounding box of this model
    """

    def __init__(self, frame, *parameters, bbox=None):
        assert isinstance(frame, Frame)
        self.frame = frame
        assert isinstance(bbox, Box)
        self.bbox = bbox
        super().__init__(*parameters)


class TabulatedSpectrum(Spectrum):
    """Spectrum from a array/table

    The class uses an arbitrary array as non-parametric model.

    Parameters
    ----------
    frame: `~scarlet.Frame`
        Characterization of the model
    spectrum: 1D array or `~scarlet.Parameter`
        Spectrum parameter
    bbox: `~scarlet.Box`
        1D bounding box for focation of the spectrum in `frame`
    """

    def __init__(self, frame, spectrum, bbox=None, noise_rms=None):
        if isinstance(spectrum, Parameter):
            assert spectrum.name == "spectrum"
        else:
            # slightly positive values
            constraint = PositivityConstraint(zero=1e-20)
            # steps of 1% of mean amplitude, minimum set by noise_rms
            factor = 1e-2
            if noise_rms is None:
                noise_rms = 0
            else:
                noise_rms = noise_rms.mean()
            step = partial(relative_step, factor=factor, minimum=noise_rms)
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

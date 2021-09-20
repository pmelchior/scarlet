from functools import partial

from .bbox import Box
from .constraint import PositivityConstraint, CorellationConstraint, ConstraintChain
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

import numpy as np
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
    min_step: float
        Minimum absolute step size for spectrum element updates.
    """

    def __init__(self, frame, spectrum, bbox=None, min_step=0):
        if isinstance(spectrum, Parameter):
            assert spectrum.name == "spectrum"
        else:
            # slightly positive values
            constraint = PositivityConstraint(0.1*  np.max(spectrum))
            # steps of 1% of mean amplitude, minimum set by noise_rms
            step = partial(relative_step, factor=1e-2, minimum=min_step)
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

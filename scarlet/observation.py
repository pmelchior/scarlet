import autograd.numpy as np

from . import interpolation
from .bbox import Box, overlapped_slices
from .frame import Frame
from .renderer import Renderer, NullRenderer, ConvolutionRenderer, ResolutionRenderer


class Observation(Frame):
    """Data and metadata for a single set of observations

    Attributes
    ----------
    data: array
        3D data cube (channels, Ny, Nx) of the image in each band.
    weights: array or tensor
        Weight for each pixel in `data`.
        If a set of masks exists for the observations then
        then any masked pixels should have their `weight` set
        to zero.
    """

    def __init__(self, data, channels, psf=None, weights=None, wcs=None, padding=10):
        """Create an Observation

        Parameters
        ---------
        data: array or tensor
            3D data cube (Channel, Height, Width) of the image in each band.
        psf: `scarlet.PSF` or its arguments
            PSF in each channel. Can be 3D cube of data stacked in channel direction.
        weights: array or tensor
            Weight for each pixel in `data`.
            If a set of masks exists for the observations then
            then any masked pixels should have their `weight` set
            to zero.
        wcs: TBD
            World Coordinate System associated with the data.
        channels: list of hashable elements
            Names/identifiers of spectral channels
        padding: int
            Number of pixels to pad each side with, in addition to
            half the width of the PSF, for FFTs. This is needed to
            prevent artifacts from the FFT.
        """
        super().__init__(
            data.shape, wcs=wcs, psf=psf, channels=channels, dtype=data.dtype
        )

        self.data = data
        if weights is not None:
            self.weights = weights
        else:
            self.weights = np.ones(data.shape, dtype=data.dtype)
        assert (
            self.weights.shape == self.data.shape
        ), "Weights needs to have same shape as data"

    def match(self, model_frame, renderer=None):
        """Match the frame of the model to the frame of this observation.

        The method sets up the mappings in spectral and spatial coordinates,
        which includes a spatial selection, computing PSF difference kernels
        and filter transformations.

        Parameters
        ---------
        model_frame: a `scarlet.Frame` instance
            The frame of `Blend` to match
        renderer: a `scarlet.Renderer` instance
            The transformation from model to observation

        Returns
        -------
        None
        """
        self.model_frame = model_frame

        # check dtype consistency
        if self.dtype != model_frame.dtype:
            self.dtype = model_frame.dtype
            self.data = self.data.astype(model_frame.dtype)
            if type(self.weights) is np.ndarray:
                self.weights = self.weights.astype(model_frame.dtype)

        # choose the renderer
        if renderer is None:
            if self.psf is model_frame.psf:
                self.renderer = NullRenderer()
            else:
                assert self.psf is not None and model_frame.psf is not None
                if self.wcs is model_frame.wcs:
                    # same or None wcs: ConvolutionRenderer
                    self.renderer = ConvolutionRenderer(
                        self, model_frame, convolution_type="fft"
                    )
                else:
                    # if wcs shows changes in resolution or orientation:
                    # use ResolutionRenderer
                    assert self.wcs is not None and model_frame.wcs is not None
                    angle, h = interpolation.get_angles(self.wcs, model_frame.wcs)
                    same_res = abs(h - 1) < np.finfo(float).eps
                    same_rot = (np.abs(angle[1]) ** 2) < np.finfo(float).eps
                    if same_res and same_rot:
                        self.renderer = ConvolutionRenderer(
                            self, model_frame, convolution_type="fft"
                        )
                    else:
                        self.renderer = ResolutionRenderer(self, model_frame)
        else:
            assert isinstance(renderer, Renderer)
            self.renderer = renderer

        return self

    @property
    def noise_rms(self):
        if not hasattr(self, "_noise_rms"):
            import numpy.ma as ma

            self._noise_rms = 1 / np.sqrt(ma.masked_equal(self.weights, 0))
            ma.set_fill_value(self._noise_rms, np.inf)

        return self._noise_rms

    @property
    def parameters(self):
        # data is immutable, but renderer might be parameterized
        return self.renderer.parameters

    def render(self, model, *parameters):
        """Convolve a model to the observation frame

        Parameters
        ----------
        model: array
            The hyperspectral model
        parameters: tuple of optimization parameters

        Returns
        -------
        model_: array
            `model` mapped into the observation frame
        """
        return self.renderer(model, self.parameters)

    def get_log_likelihood(self, model, *parameters, noise_factor=0):
        """Computes the log-Likelihood of a given model wrt to the observation

        Parameters
        ----------
        model: array
            The model from `Blend`
        parameters: tuple of optimization parameters

        Returns
        -------
        logL: float
        """
        model_ = self.render(model, *parameters)
        data_ = self.data
        weights_ = self.weights

        # noise injection to soften the gradient
        if noise_factor > 0:
            noise = np.random.normal(loc=0, scale=self.noise_rms)
            data_ = data_ + noise
            weights_ = weights_ / (noise_factor + 1)

        return -self.log_norm - np.sum(weights_ * (model_ - data_) ** 2) / 2

    @property
    def log_norm(self):
        if not hasattr(self, "_log_norm"):
            # normalization of the single-pixel likelihood:
            # 1 / [(2pi)^1/2 (sigma^2)^1/2]
            # with inverse variance weights: sigma^2 = 1/weight
            # full likelihood is sum over all data samples: pixel in data
            D = np.prod(self.data.shape) - self.noise_rms.mask.sum()
            self._log_norm = D / 2 * np.log(2 * np.pi)

            # noise_rms has 0s masked, but log still throws warning...
            with np.errstate(divide="ignore"):
                self._log_norm += np.log(self.noise_rms).sum()

        return self._log_norm

    def _to_frame(self, frame, data=None):
        """Project this observation into another frame

        Note: the frame must have the same sampling and rotation,
        but can differ in the shape and origin of the `Frame`.
        This method is a convenience function for now but should
        not be considered supported and could be removed in a later version
        """
        frame_slices, observation_slices = overlapped_slices(frame.bbox, self.bbox)

        if data is None:
            data = self.data

        if hasattr(frame, "dtype"):
            dtype = frame.dtype
        else:
            dtype = data.dtype
        result = np.zeros(frame.shape, dtype=dtype)
        result[frame_slices] = data[observation_slices]
        return result

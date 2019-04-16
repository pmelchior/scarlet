import numpy as np
import torch
from . import convolution

import logging
logger = logging.getLogger("scarlet.observation")

def build_detection_coadd(sed, bg_rms, observation, scene, thresh=1):
    """Build a band weighted coadd to use for source detection

    Parameters
    ----------
    sed: array
        SED at the center of the source.
    bg_rms: array
        Background RMS in each band in observation.
    observation: `~scarlet.observation.Observation`
        Observation to use for the coadd.
    scene: `scarlet.observation.Scene`
        The scene that the model lives in.
    thresh: `float`
        Multiple of the backround RMS used as a
        flux cutoff.

    Returns
    -------
    detect: array
        2D image created by weighting all of the bands by SED
    bg_cutoff: float
        The minimum value in `detect` to include in detection.
    """
    B = observation.B
    images = observation.get_scene(scene)
    weights = np.array([sed[b]/bg_rms[b]**2 for b in range(B)])
    jacobian = np.array([sed[b]**2/bg_rms[b]**2 for b in range(B)]).sum()
    detect = np.einsum('i,i...', weights, images) / jacobian

    # thresh is multiple above the rms of detect (weighted variance across bands)
    bg_cutoff = thresh * np.sqrt((weights**2 * bg_rms**2).sum()) / jacobian
    return detect, bg_cutoff


class Scene():
    """Extent and characteristics of the modeled scence

    Attributes
    ----------
    shape: tuple
        (bands, Ny, Nx) shape of the model image.
    wcs: TBD
        World Coordinates
    psfs: array or tensor
        PSF in each band
    filtercurve: TBD
        Filter curve used for unpacking spectral information.
    dtype: `~numpy.dtype`
        Data type of the model.
    """
    #
    def __init__(self, shape, wcs=None, psfs=None, filtercurve=None):
        self._shape = tuple(shape)
        self.wcs = wcs

        assert psfs is None or shape[0] == len(psfs)
        if psfs is not None:
            psfs = torch.Tensor(psfs)
        self._psfs = psfs
        assert filtercurve is None or shape[0] == len(filtercurve)
        self.filtercurve = filtercurve

    @property
    def B(self):
        """Number of bands in the model
        """
        return self.shape[0]

    @property
    def Ny(self):
        """Number of pixel in the y-direction
        """
        return self.shape[1]

    @property
    def Nx(self):
        """Number of pixels in the x-direction
        """
        return self.shape[2]

    @property
    def shape(self):
        """Shape of the model.
        """
        return self._shape

    @property
    def psfs(self):
        if self._psfs is None:
            return None
        return self._psfs.data.detach().numpy()

    def get_pixel(self, sky_coord):
        """Get the pixel coordinate from a world coordinate

        If there is no WCS associated with the `Scene`,
        meaning the data frame and model frame are the same,
        then this just returns the `sky_coord`
        """
        if self.wcs is not None:
            return self.wcs.radec2pix(sky_coord).int()
        return (int(sky_coord[0]), int(sky_coord[1]))


class Observation(Scene):
    """Data and metadata for a single set of observations

    Attributes
    ----------
    images: array or tensor
        3D data cube (bands, Ny, Nx) of the image in each band.
        These images must be resampled to the same pixel scale and
        same target WCS (for now).
    psfs: array or tensor
        PSF for each band in `images`.
    weights: array or tensor
        Weight for each pixel in `images`.
        If a set of masks exists for the observations then
        then any masked pixels should have their `weight` set
        to zero.
    wcs: TBD
        World Coordinate System associated with the images.
    padding: int
        Number of pixels to pad each side with, in addition to
        half the width of the PSF, for FFT's. This is needed to
        prevent artifacts due to the FFT.
    """
    def __init__(self, images, psfs=None, weights=None, wcs=None, filtercurve=None, padding=3):
        super().__init__(images.shape, wcs=wcs, psfs=psfs, filtercurve=filtercurve)

        self._images = torch.Tensor(images)
        self.padding = padding

        if weights is not None:
            self._weights = torch.Tensor(weights)
        else:
            self._weights = 1

        # Calculate and store the PSFs in Fourier space
        if self._psfs is not None:
            ipad, ppad = convolution.get_common_padding(images, psfs, padding=padding)
            self.image_padding, self.psf_padding = ipad, ppad
            _psfs = torch.nn.functional.pad(self._psfs, self.psf_padding)
            self.psfs_fft = torch.rfft(_psfs, 2)

    @property
    def images(self):
        return self._images.data.detach().numpy()

    @property
    def weights(self):
        if self._weights is None:
            return None
        elif self._weights == 1:
            return self._weights
        return self._weights.data.detach().numpy()

    def get_model(self, model, numpy=True):
        """Resample and convolve a model to the observation frame

        Parameters
        ----------
        model: `~torch.Tensor`
            The model in some other data frame.
        numpy: bool
            Whether to return the model as a numpy array
            (`numpy=True`) or a `torch.tensor`.

        Returns
        -------
        model: `~torch.Tensor`
            The convolved and resampled `model` in the observation frame.
        """
        if self.wcs is not None or scene.shape != self.shape:
            msg = "get_model is currently only supported when the observation frame matches the scene"
            raise NotImplementedError(msg)

        def _convolve_band(model, psf):
            """Convolve the model in a single band
            """
            _model = torch.nn.functional.pad(model, self.image_padding)
            Image = torch.rfft(_model, 2)
            Convolved = convolution.complex_mul(Image, psf)
            convolved = torch.irfft(Convolved, 2, signal_sizes=_model.shape)
            result = convolution.ifftshift(convolved)
            bottom, top, left, right = self.image_padding
            result = result[bottom:-top, left:-right]
            return result
        model = torch.stack([_convolve_band(model[b], self.psfs_fft[b]) for b in range(self.B)])
        if numpy:
            model = model.detach().numpy()
        return model

    def get_loss(self, model):
        """Calculate the loss function for the model

        Parameters
        ----------
        model: `~torch.Tensor`
            The model in some other data frame.

        Returns
        -------
        result: `~torch.Tensor`
            Scalar tensor with the likelihood of the model
            given the image data.
        """
        if self._psfs is not None:
            model = self.get_model(model, False)
        model *= self._weights
        return 0.5 * torch.nn.MSELoss(reduction='sum')(model, self._images*self._weights)

    def get_scene(self, scene, numpy=True):
        """Reproject and resample the image in some other data frame

        This is currently only supported to return `images` when the data
        scene and target scene are the same.

        Parameters
        ----------
        scene: `~scarlet.observation.Scene`
            The target data frame.

        Returns
        -------
        images: `~torch.Tensor`
            The image cube in the target `scene`.
        """
        if self.wcs is not None or scene.shape != self.shape:
            msg = "get_scene is currently only supported when the observation frame matches the scene"
            raise NotImplementedError(msg)
        if numpy:
            return self.images
        return self._images

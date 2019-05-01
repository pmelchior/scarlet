import numpy as np
import torch

from . import convolution
from . import resampling
from . import psf_match

import logging
logger = logging.getLogger("scarlet.observation")


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

        assert psfs is None or shape[0] == len(psfs) or len(psfs.shape) == 2
        if psfs is not None:
            psfs = torch.Tensor(psfs)
            psfs /= psfs.sum()
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

    def match(self, scene):


        #Get pixel coordinates in each frame
        mask,over_lr, over_hr  = resampling.match_patches(scene.shape, self.shape,scene.wcs, self.wcs)

        #Compute diff kernel at hr
        if scene._psfs != None:
            target_psf = scene._psf[0,:,:]
            coord_phr = np.where(target_psf*0.==0)
            coord_plr = np.where(target_psf[0,:,:] * 0. == 0)
            #This is going to assume that the spatial span of the psfs matches. In practice, we actually need their wcs
            #In which case both previous lines will be replaced by:
            # mask_p,coord_plr, coord_phr  = resampling.match_patches(np.shape(scene.psf), np.shape(self.psf),scene.pwcs, self.pwcs)
            interp_diff = []
            for _psf_self in self._psfs:
                interp_psf = resampling.interp2D(coord_phr, coord_plr,_psf_self)

            #Here we need to choose a reference PSF I choose the first one for now, but it might be a degraded version of all the high resolution PSFs.

                diff_psf, psf_blend = psf_match.build_diff_kernels(interp_psf,target_psf, l0_thresh=0.000001)
                interp_diff.appen(diff_psf)

        # Computes the resampling/convolution matrix
        self.resample = resampling.make_mat(mask.shape(), over_lr, diff_psf)

        # 3) compute obs.psf in the frame of scene, store in Fourier space
        # A few notes on this procedure:
        # a) This assumes that scene.psfs and self.psfs have the same spatial shape,
        #    which will need to be modified for multi-resolution datasets
        # b)Currently pytorch does not have complex tensor type
        #   (see https://github.com/pytorch/pytorch/issues/755),
        #   so in the meantime we convert the PSFs to numpy, perform the
        #   deconvolution there, and then convert back to pytorch. Once
        #   proper support is added by pytorch we can use Tensors all the way through.
        if self._psfs is not None:
            ipad, ppad = convolution.get_common_padding(self._images, self._psfs, padding=self.padding)
            self.image_padding, self.psf_padding = ipad, ppad
            _psfs = torch.nn.functional.pad(self._psfs, self.psf_padding)
            _target = torch.nn.functional.pad(scene._psfs, self.psf_padding)

            new_kernel_fft = []
            # Deconvolve the target PSF
            target_fft = np.fft.fft2(np.fft.ifftshift(_target.detach().numpy()))

            for _psf in _psfs:
                observed_fft = np.fft.fft2(np.fft.ifftshift(_psf.detach().numpy()))
                # Create the matching kernel
                kernel_fft = observed_fft / target_fft
                # Take the inverse Fourier transform to normalize the result
                # Trials without this operation are slow to converge, but in the future
                # we may be able to come up with a method to normalize in the Fourier Transform
                # and avoid this step.
                kernel = np.fft.ifft2(kernel_fft)
                kernel = np.fft.fftshift(np.real(kernel))
                kernel /= kernel.sum()
                kernel = torch.Tensor(kernel)
                # Store the Fourier transform of the matching kernel
                new_kernel_fft.append(torch.rfft(kernel, 2))
            self.psfs_fft = torch.stack(new_kernel_fft)

        # 4) divide obs.psf from scene.psf in Fourier space

        # [ 5) compute sparse representation of interpolation * convolution ]

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
        if self.wcs is not None:
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

        def _resample_band(model, M, coord):
            """applies joint resampling and convolution in a single band (no factorisation)
            """
            return np.dot(model.flatten(), M)


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

    def get_scene(self, scene, M, coord_lr, numpy=True):
        """Reproject and resample  images in some other data frame

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
        result = torch.zeros((self.shape))
        for b in range(scene.B):
            result[b,coord_lr] =  np.dot(scene[b],M)

        if numpy:
            return result
        return result

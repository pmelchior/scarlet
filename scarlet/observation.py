import autograd.numpy as np
from scipy import fftpack

from . import interpolation

from . import resampling

import logging

logger = logging.getLogger("scarlet.observation")


def _centered(arr, newshape):
    """Return the center newshape portion of the array.

    This function is used by `fft_convolve` to remove
    the zero padded region of the convolution.
    """
    newshape = np.asarray(newshape)
    currshape = np.array(arr.shape[1:])
    startind = (currshape - newshape) // 2
    endind = startind + newshape
    myslice = np.concatenate(([slice(0,arr.shape[0])],[slice(startind[k], endind[k]) for k in range(len(endind))]))
    return arr[tuple(myslice)]


class Frame():
    """Spatial and spectral characteristics of the data

    Attributes
    ----------
    shape: tuple
        (channels, Ny, Nx) shape of the model image
    wcs: TBD
        World Coordinates
    psfs: array or tensor
        PSF in each band
    channels: list of hashable elements
        Names/identifiers of spectral channels
    """
    def __init__(self, shape, wcs=None, psfs=None, channels=None):
        assert len(shape) == 3
        self._shape = tuple(shape)
        self.wcs = wcs

        if psfs is None:
            logger.warning('No PSFs specified. Possible, but dangerous!')
        else:
            assert len(psfs) == 1 or len(psfs) == shape[0], 'PSFs need to have shape (1,Ny,Nx) for Blend and (B,Ny,Nx) for Observation'
            if not np.allclose(psfs.sum(axis=(1, 2)), 1):
                logger.warning('PSFs not normalized. Normalizing now..')
                psfs /= psfs.sum(axis=(1, 2))[:,None,None]
        self._psfs = psfs

        assert channels is None or len(channels) == shape[0]
        self.channels = channels

    @property
    def C(self):
        """Number of channels in the model
        """
        return self._shape[0]

    @property
    def Ny(self):
        """Number of pixel in the y-direction
        """
        return self._shape[1]

    @property
    def Nx(self):
        """Number of pixels in the x-direction
        """
        return self._shape[2]

    @property
    def shape(self):
        """Shape of the model.
        """
        return self._shape

    @property
    def psfs(self):
        return self._psfs

    def get_pixel(self, sky_coord):
        """Get the pixel coordinate from a world coordinate
        If there is no WCS associated with the `Scene`,
        meaning the data frame and model frame are the same,
        then this just returns the `sky_coord`
        """
        if self.wcs is not None:
            if self.wcs.naxis == 3:
                coord = self.wcs.wcs_world2pix(sky_coord[0], sky_coord[1], 0, 0)
            elif self.wcs.naxis == 2:
                coord = self.wcs.wcs_world2pix(sky_coord[0], sky_coord[1], 0)
            else:
                raise ValueError("Invalid number of wcs dimensions: {0}".format(self.wcs.naxis))
            return (int(coord[0].item()), int(coord[1].item()))

        return tuple(int(coord) for coord in sky_coord)


class Observation():
    """Data and metadata for a single set of observations

    Attributes
    ----------
    images: array or tensor
        3D data cube (channels, Ny, Nx) of the image in each band.
    frame: a `scarlet.Frame` instance
        The spectral and spatial characteristics of these data
    weights: array or tensor
        Weight for each pixel in `images`.
        If a set of masks exists for the observations then
        then any masked pixels should have their `weight` set
        to zero.
    padding: int
        Number of pixels to pad each side with, in addition to
        half the width of the PSF, for FFTs. This is needed to
        prevent artifacts from the FFT.
    """

    def __init__(self, images, psfs=None, weights=None, wcs=None, channels=None,
                 padding=10):
        """Create an Observation

        Parameters
        ---------
        images: array or tensor
            3D data cube (channels, Ny, Nx) of the image in each band.
        psfs: array or tensor
            PSF for each band in `images`.
        weights: array or tensor
            Weight for each pixel in `images`.
            If a set of masks exists for the observations then
            then any masked pixels should have their `weight` set
            to zero.
        wcs: TBD
            World Coordinate System associated with the images.
        channels: list of hashable elements
            Names/identifiers of spectral channels
        padding: int
            Number of pixels to pad each side with, in addition to
            half the width of the PSF, for FFTs. This is needed to
            prevent artifacts from the FFT.
        """
        self.frame = Frame(images.shape, wcs=wcs, psfs=psfs, channels=channels)

        self.images = np.array(images)
        if weights is not None:
            self.weights = np.array(weights)
        else:
            self.weights = 1

        self._padding = padding

    def match(self, model_frame):
        """Match the frame of `Blend` to the frame of this observation.

        The method sets up the mappings in spectral and spatial coordinates,
        which includes a spatial selection, computing PSF difference kernels
        and filter transformations.

        Parameters
        ---------
        model_frame: a `scarlet.Frame` instance
            The frame of `Blend` to match

        Returns
        -------
        None
        """

        #  channels of model that are represented in this observation
        self._band_slice = slice(None)
        if self.frame.channels is not model_frame.channels:
            assert self.frame.channels is not None and model_frame.channels is not None
            bmin = model_frame.channels.index(self.frame.channels[0])
            bmax = model_frame.channels.index(self.frame.channels[-1])
            self._band_slice = slice(bmin, bmax+1)

        self._diff_kernels_fft = None
        if self.frame.psfs is not model_frame.psfs:
            assert self.frame.psfs is not None and model_frame.psfs is not None
            # First we setup the parameters for the model -> observation FFTs
            # Make the PSF stamp wider due to errors when matching PSFs
            psf_shape = np.array(self.frame.psfs[0].shape) + self._padding
            shape = np.array(model_frame.shape[1:]) + psf_shape - 1
            # Choose the optimal shape for FFTPack DFT
            self._fftpack_shape = [fftpack.helper.next_fast_len(d) for d in shape]
            # Store the pre-fftpack optimization slices
            self.slices = tuple(np.concatenate(([slice(self.C)],[slice(s) for s in shape])))

            # Now we setup the parameters for the psf -> kernel FFTs
            shape = np.array(model_frame.psfs[0].shape) + np.array(self.frame.psfs[0].shape) - 1
            _fftpack_shape = [fftpack.helper.next_fast_len(d) for d in shape]
            # Deconvolve the target PSF
            target_fft = np.fft.rfftn(model_frame.psfs[0], _fftpack_shape)

            # Match the PSF in each band
            _psf_fft = np.fft.rfftn(self.psfs, _fftpack_shape, axes=(1, 2))


            kernels = np.fft.ifftshift(np.fft.irfftn(_psf_fft / target_fft, _fftpack_shape, axes=(1, 2)), axes=(1, 2))

            if kernels.shape[1] % 2 == 0:
                kernels = kernels[:, 1:, 1:]

            kernels = _centered(kernels, psf_shape)
            _diff_kernels_fft = np.fft.rfftn(kernels, self.fftpack_shape, axes=(1, 2))

            self._diff_kernels_fft = np.array(_diff_kernels_fft)

        return self

    def _convolve_band(self, model, diff_kernel_fft):
        """Convolve the model in a single band
        """
        model_fft = np.fft.rfftn(model, self._fftpack_shape)
        convolved = np.fft.irfftn(model_fft * diff_kernel_fft, self._fftpack_shape)[self._slices]
        return _centered(convolved, model.shape)

    def render(self, model):
        """Convolve a model to the observation frame

        Parameters
        ----------
        model: array
            The model from `Blend`

        Returns
        -------
        model_: array
            The convolved `model` in the observation frame
        """
        model_ = model[self._band_slice,:,:]

        if self._diff_kernels_fft is not None:
            model_ = np.array([self._convolve_band(model_[c], self._diff_kernels_fft[c]) for c in range(self.frame.C)])

        return model_

    def get_loss(self, model):
        """Computes the loss/fidelity of a given model wrt to the observation

        Parameters
        ----------
        model: array
            The model from `Blend`

        Returns
        -------
        result: array
            Scalar tensor with the likelihood of the model
            given the image data
        """

        model = self.render(model)

        return 0.5 * np.sum((self.weights * (model - self.images)) ** 2)


class LowResObservation(Observation):

    def __init__(self, images, wcs=None, psfs=None, weights=None, channels=None, padding=3):

        assert wcs is not None, "WCS is necessary for LowResObservation"
        assert psfs is not None, "PSFs are necessary for LowResObservation"

        self.frame = Frame(images.shape, wcs=wcs, psfs=psfs, channels=channels)
        self.images = np.array(images)
        self._padding = padding

        if weights is not None:
            self.weights = np.array(weights)
        else:
            self.weights = 1

    def make_operator(self, shape, psfs):
        '''Builds the resampling and convolution operator

        Builds the matrix that expresses the linear operation of resampling a function evaluated on a grid with coordinates
        'coord_lr' to a grid with shape 'shape', and convolving by a kernel p

        Parameters
        ------
        shape: tuple
            shape of the high resolution scene
        coord_lr: array
            coordinates of the overlapping pixels from the low resolution grid in the high resolution grid frame.
        p: array
            convolution kernel (PSF)
        Returns
        -------
        operator: array
            the convolution-resampling matrix
        '''
        B, Ny, Nx = shape
        Bpsf = psfs.shape[0]

        ker = np.zeros((Ny,Nx))
        y_hr, x_hr = np.where(ker == 0)
        y_lr, x_lr = self._coord_lr

        Nlr = x_lr.size

        h = y_hr[1] - y_hr[0]
        if h == 0:
            h = x_hr[1] - x_hr[0]
        assert h != 0

        # sinc interpolant:
        #ker_mat = interpolation.sinc2D((y_lr[:, np.newaxis] - y_hr[np.newaxis, :]) / h,
        #                           (x_lr[:, np.newaxis] - x_hr[np.newaxis, :]) / h)#.reshape(Nlr, Ny, Nx)
        #print(ker_mat.shape)
        import matplotlib.pyplot as plt

        print(self.fftpack_shape[0])
        operator = []
        for m in range(Nlr):
            ker[y_hr,x_hr] = interpolation.sinc2D((y_lr[m] - y_hr) / h,
                                       (x_lr[m] - x_hr) / h)#.reshape(Ny, Nx)
            print(x_lr, x_hr)

            plt.imshow(np.log(ker), cmap = 'gist_stern')
            plt.show()
            ker_fft = np.fft.rfftn(ker, self.fftpack_shape)
            operator_fft = ker_fft[np.newaxis, :, :] * psfs
            op_ifft = np.fft.ifftshift(np.fft.irfftn(operator_fft, axes=(1, 2)), axes=(1, 2))
            operator.append(_centered(op_ifft, (Bpsf, Ny,Nx)))
        print(np.shape(operator_fft))
        #operator_fft = np.reshape(operator_fft, (Bpsf*Nlr,self.fftpack_shape[0],self.fftpack_shape[1]))

        operator = _centered(operator, (Bpsf*Nlr, Ny,Nx))

#        import scipy.signal as scp
#        operator = scp.fftconvolve(psfs[None,:,:,:], ker[:,None,:,:], mode = 'same', axes = (2,3))
        #    print(np.shape(operator))
        #    plt.imshow(np.array(operator)[0, 1, :, :], cmap = 'gist_stern'); plt.show()
        print(np.shape(operator))

        # FFTs of the psf and sinc
        #ker_fft = np.fft.rfftn(ker, self.fftpack_shape, axes=(1, 2))
        #
        #operator = np.fft.irfftn(operator_fft, axes=(2,3)) * Nlr / (Nx * Ny) / np.pi

        #A little trimming
        #operator = _centered(operator, (Bpsf, Ny, Nx)).reshape(Bpsf, Nlr, Nx * Ny)

        return np.array(operator).reshape(Bpsf, Nx*Ny, Nlr)*Nlr/(Nx*Ny*np.pi)

    def match(self, model_frame):
        '''Matches the observation with a scene

        Builds the tools to project images of this observation to a scene at a different resolution and psf.

        Parameters
        ----------
        scene: Scene object
            A scene in which to project the images from this observation
        Returns
        -------
        None
            instanciates new attributes of the object

        '''
        #  channels of model that are represented in this observation
        self._band_slice = slice(None)
        if self.frame.channels is not model_frame.channels:
            bmin = model_frame.channels.index(self.frame.channels[0])
            bmax = model_frame.channels.index(self.frame.channels[-1])
            self._band_slice = slice(bmin, bmax+1)
        # Get pixel coordinates in each frame.
        mask, coord_lr, coord_hr = resampling.match_patches(model_frame.shape, self.frame.shape, model_frame.wcs, self.frame.wcs)
        self._coord_lr = coord_lr
        self._coord_hr = coord_hr
        self._mask = mask

        # Compute diff kernel at hr

        whr = model_frame.wcs
        wlr = self.frame.wcs

        # Reference PSF
        #This definition of a target is a little arbitrary
        # I will probably be the job of `band to point the taget psf
        _target = model_frame.psfs[0, :, :]
        _shape = model_frame.shape

        resconv_op = []
        target_kernels = []
        observed_kernels = []
        for _psf in self.psfs:
            # Computes spatially matching observation and target psfs. The observation psf is also resampled to the scene's resolution
            new_target, observed_psf = resampling.match_psfs(_target, _psf, whr, wlr)
            target_kernels.append(new_target)
            observed_kernels.append(observed_psf)


        # First we setup the parameters for the model -> observation FFTs
        # Make the PSF stamp wider due to errors when matching PSFs
        psf_shape = np.array(new_target[1].shape) + 10
        shape = np.array(model_frame.shape[1:]) + psf_shape - 1
        # Choose the optimal shape for FFTPack DFT
        self.fftpack_shape = [fftpack.helper.next_fast_len(d) for d in shape]
        # Store the pre-fftpack optimization slices
        self.slices = tuple([slice(s) for s in shape])

        # Computes the diff kernel in Fourier
        target_fft = np.fft.rfftn(np.fft.ifftshift(target_kernels, axes = (1, 2)), self.fftpack_shape, axes=(1, 2))
        observed_fft = np.fft.rfftn(np.fft.ifftshift(observed_kernels, axes=(1, 2)), self.fftpack_shape, axes=(1, 2))

        sel = (target_fft == 0)
        diff_fft = observed_fft / target_fft
        diff_fft[sel] = 0
        diff_fft = diff_fft / diff_fft.sum(axis = (1, 2))[:, np.newaxis,np.newaxis]
        #diff_psf = np.fft.ifftshift(np.fft.irfftn(diff_fft, self.fftpack_shape, axes=(1, 2)), axes=(1, 2))
        #diff_psf = _centered(diff_psf, _shape[1:])

        # Computes the resampling/convolution matrix
        resconv_op = self.make_operator(_shape, diff_fft)

        self._resconv_op = np.array(resconv_op)

        return self

    @property
    def matching_mask(self):
        return self._mask

    def _render(self, model):
        """Resample and convolve a model in the observation frame

        Parameters
        ----------
        model: array
            The model in some other data frame.

        Returns
        -------
        model_: array
            The convolved and resampled `model` in the observation frame.
        """
        model_ = model[self._band_slice,:,:]
        model_ = np.array([np.dot(model_[c].flatten(), self._resconv_op[c]) for c in range(self.frame.C)])

        return model_

    def render(self, model):
        """Resample and convolve a model in the observation frame

        Parameters
        ----------
        model: array
            The model in some other data frame.

        Returns
        -------
        model_: array
            The convolved and resampled `model` in the observation frame.
        """
        img = np.zeros(self.frame.shape)
        img[:, self._coord_lr[0], self._coord_lr[1]] = self._render(model)
        return img

    def get_loss(self, model):
        """Computes the loss/fidelity of a given model wrt to the observation

        Parameters
        ----------
        model: array
            A model from `Blend`

        Returns
        -------
        loss: float
            Loss of the model
        """

        model_ = self._render(model)

        return 0.5 * np.sum((self.weights * (
                model_ - self.images[:, self._coord_lr[0].astype(int), self._coord_lr[1].astype(int)])) ** 2)

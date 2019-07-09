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
    currshape = np.array(arr.shape)
    startind = (currshape - newshape) // 2
    endind = startind + newshape
    myslice = [slice(startind[k], endind[k]) for k in range(len(endind))]

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
            assert len(psfs) == 1 or len(psfs) == shape[
                0], 'PSFs need to have shape (1,Ny,Nx) for Blend and (B,Ny,Nx) for Observation'
            if not np.allclose(psfs.sum(axis=(1, 2)), 1):
                logger.warning('PSFs not normalized. Normalizing now..')
                psfs /= psfs.sum(axis=(1, 2))[:, None, None]
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

    def __init__(self, images, psfs=None, weights=None, wcs=None, channels=None, padding=10):
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
            self._band_slice = slice(bmin, bmax + 1)

        self._diff_kernels_fft = None
        if self.frame.psfs is not model_frame.psfs:
            assert self.frame.psfs is not None and model_frame.psfs is not None
            # First we setup the parameters for the model -> observation FFTs
            # Make the PSF stamp wider due to errors when matching PSFs
            psf_shape = np.array(self.frame.psfs.shape)
            psf_shape[1:] += self._padding
            shape = np.array(model_frame.shape) + psf_shape - 1
            shape[0] = model_frame.shape[0]

            # Choose the optimal shape for FFTPack DFT
            self._fftpack_shape = [fftpack.helper.next_fast_len(d) for d in shape[1:]]
            # autograd.numpy.fft does not currently work
            # if the last dimension is odd
            while self._fftpack_shape[-1] % 2 != 0:
                _shape = self._fftpack_shape[-1] + 1
                self._fftpack_shape[-1] = fftpack.helper.next_fast_len(_shape)

            # Store the pre-fftpack optimization slices
            self.slices = tuple(([slice(s) for s in shape]))

            # Now we setup the parameters for the psf -> kernel FFTs
            shape = np.array(model_frame.psfs.shape)
            shape[1:] += np.array(self.frame.psfs[0].shape)

            _fftpack_shape = [fftpack.helper.next_fast_len(d) for d in shape[1:]]

            # fft of the target psf
            target_fft = np.fft.rfftn(model_frame.psfs, _fftpack_shape, axes=(1, 2))

            # fft of the observation's PSFs in each band
            _psf_fft = np.fft.rfftn(self.frame.psfs, _fftpack_shape, axes=(1, 2))

            # Diff kernel between observation and target psf in Fourrier
            kernels = np.fft.ifftshift(np.fft.irfftn(_psf_fft / target_fft, _fftpack_shape, axes=(1, 2)), axes=(1, 2))

            if kernels.shape[1] % 2 == 0:
                kernels = kernels[:, 1:, 1:]

            kernels = _centered(kernels, psf_shape)

            self._diff_kernels_fft = np.fft.rfftn(kernels, self._fftpack_shape, axes=(1, 2))

        return self

    def _convolve(self, model):
        """Convolve the model in a single band
        """
        model_fft = np.fft.rfftn(model, self._fftpack_shape, axes=(1, 2))
        convolved = np.fft.irfftn(model_fft * self._diff_kernels_fft, self._fftpack_shape, axes=(1, 2))[self.slices]
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
        model_ = model[self._band_slice, :, :]
        if self._diff_kernels_fft is not None:
            model_ = self._convolve(model_)

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

    def make_operator(self, shape, psf):
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
        mat: array
            the convolution-resampling matrix
        '''
        B, Ny, Nx = shape
        y_hr, x_hr = np.where(np.zeros((Ny, Nx)) == 0)
        y_lr, x_lr = self._coord_lr
        mat = np.zeros((Ny * Nx, x_lr.size))

        import scipy.signal as scp

        for m in range(np.size(x_lr)):
            mat[:, m] = scp.fftconvolve(self._ker[m], psf, mode='same').flatten() / np.pi * y_lr.size / y_hr.size
        return mat

    def match_psfs(self, psf_hr, wcs_hr):
        '''psf matching between different dataset
        Matches PSFS at different resolutions by interpolating psf_lr on the same grid as psf_hr
        Parameters
        ----------
        psf_hr: array
            centered psf of the high resolution scene
        psf_lr: array
            centered psf of the low resolution scene
        wcs_hr: WCS object
            wcs of the high resolution scene
        wcs_lr: WCS object
            wcs of the low resolution scene
        Returns
        -------
        psf_match_hr: array
            high rresolution psf at mactching size
        psf_match_lr: array
            low resolution psf at matching size and resolution
        '''

        psf_lr = self.frame.psfs
        wcs_lr = self.frame.wcs

        ny_hr, nx_hr = psf_hr.shape
        npsf, ny_lr, nx_lr = psf_lr.shape

        # Createsa wcs for psfs centered around the frame center
        psf_wcs_hr = wcs_hr.deepcopy()
        psf_wcs_lr = wcs_lr.deepcopy()

        if psf_wcs_hr.naxis == 2:
            psf_wcs_hr.wcs.crval = 0., 0.
            psf_wcs_hr.wcs.crpix = ny_hr / 2., nx_hr / 2.
        elif psf_wcs_hr.naxis == 3:
            psf_wcs_hr.wcs.crval = 0., 0., 0.
            psf_wcs_hr.wcs.crpix = ny_hr / 2., nx_hr / 2., 0.
        if psf_wcs_lr.naxis == 2:
            psf_wcs_lr.wcs.crval = 0., 0.
            psf_wcs_lr.wcs.crpix = ny_lr / 2., nx_lr / 2.
        elif psf_wcs_lr.naxis == 3:
            psf_wcs_lr.wcs.crval = 0., 0., 0.
            psf_wcs_lr.wcs.crpix = ny_lr / 2., nx_lr / 2., 0

        mask, p_lr, p_hr = resampling.match_patches(psf_hr.shape, psf_lr.data.shape[1:], psf_wcs_hr, psf_wcs_lr)

        cmask = np.where(mask == 1)

        n_p = np.int((np.size(cmask[0])) ** 0.5)

        psf_match_lr = interpolation.sinc_interp(cmask, p_hr[::-1],
                                                 psf_lr.reshape(npsf, ny_lr * nx_lr)).reshape(npsf, n_p, n_p)


        psf_match_hr = psf_hr[np.int((ny_hr - n_p) / 2):np.int((ny_hr + n_p) / 2),
                       np.int((nx_hr - n_p) / 2):np.int((nx_hr + n_p) / 2)]

        psf_match_hr /= np.max(psf_match_hr)
        psf_match_lr /= np.max(psf_match_lr)
        return psf_match_hr[np.newaxis, :], psf_match_lr

    def match(self, model_frame):

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

        # Reference PSF
        _target = model_frame.psfs[0, :, :]
        _shape = model_frame.shape


        self._fftpack_shape = [fftpack.helper.next_fast_len(d) for d in _target.shape]

        # Interpolation kernel for resampling
        self._ker = resampling.conv2D_fft(_shape, self._coord_hr)
        # Computes spatially matching observation and target psfs. The observation psf is also resampled to the model frame resolution
        new_target, observed_psf = self.match_psfs(_target, whr)
        target_fft = np.fft.rfftn(new_target[0], self._fftpack_shape)
        sel = target_fft == 0
        observed_fft = np.fft.rfftn(observed_psf, self._fftpack_shape, axes=(1, 2))

        # Computes the diff kernel in Fourier
        kernel_fft = observed_fft / target_fft
        kernel_fft[:,sel] = 0
        kernel = np.fft.irfftn(kernel_fft, self._fftpack_shape, axes = (1,2))
        kernel = np.fft.ifftshift(kernel, axes = (1,2))

        kernel = _centered(kernel, observed_psf.shape)
        diff_psf = kernel / kernel.max()

        # Computes the resampling/convolution matrix
        resconv_op = []
        for dpsf in diff_psf:
            resconv_op.append(self.make_operator(_shape, dpsf))

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
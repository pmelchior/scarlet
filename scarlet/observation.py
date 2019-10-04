import autograd.numpy as np
from scipy import fftpack

from . import interpolation
from . import fft
from . import resampling
from .psf import generate_psf_image, gaussian, fit_target_psf

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


def sinc_shift_1D(img, shift, axis, fast_size):
    '''Performs 1D sinc convolutions and shifting in Fourier space

    Parameters
    ----------
    img: array
        the 2D image to sinc convolve and shift
    shift: array
        an array of the shift values for each line of img
    axis: int (0 or 1)
        the axis along which the convolution and shifting happens
    fast_size: int
        numpy's fast fft size for Fourier transform
    Returns
    -------
    result: array
        the shifted and sinc convolved array in configuration space
    '''

    assert np.size(img.shape) == 2
    assert axis in [0,1]

    #fft convolution along the desired axis
    img_fft = np.fft.rfftn(img, [fast_size], axes = [axis])

    #frequency sampling
    nu = np.fft.rfftfreq(fast_size)

    if axis == 0:
        # shifting operator
        shift_op = np.exp(-1j *2*np.pi* shift[:, np.newaxis] * nu[np.newaxis, :])
        # convolution by sinc: setting to zero all coefficients > n//2
        shift_op[fast_size // 2 :, :] = 0
        img_shiftfft = img_fft[np.newaxis, :, :]*shift_op[:,:,np.newaxis]

    else:
        # shifting operator
        shift_op = np.exp(-1j *2*np.pi* nu[:, np.newaxis] * shift[np.newaxis, :])
        # convolution by sinc: setting to zero all coefficients > n//2
        shift_op[:,fast_size // 2 :] = 0
        img_shiftfft = img_fft[:, :, np.newaxis]*shift_op[np.newaxis, :,:]

    return np.fft.fftshift(np.fft.irfftn(img_shiftfft, [fast_size], axes = [1]), axes = [1])


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
    dtype: `numpy.dtype`
        Dtype to represent the data.
    """
    def __init__(self, shape, wcs=None, psfs=None, channels=None, dtype=np.float32):
        assert len(shape) == 3
        self._shape = tuple(shape)
        self.wcs = wcs

        if psfs is None:
            logger.warning('No PSFs specified. Possible, but dangerous!')
        else:
            msg = 'PSFs need to have shape (1,Ny,Nx) for Blend and (B,Ny,Nx) for Observation'
            assert len(psfs) == 1 or len(psfs) == shape[0], msg
            if not isinstance(psfs, fft.Fourier):
                psfs = fft.Fourier(psfs, axes=(1, 2))
            if not np.allclose(psfs.sum(axis=(1, 2)), 1):
                logger.warning('PSFs not normalized. Normalizing now..')
                psfs.normalize()

            if dtype != psfs.image.dtype:
                msg = "Dtypes of PSFs and Frame different. Casting PSFs to {}".format(dtype)
                logger.warning(msg)
                psfs.update_dtype(dtype)

        self._psfs = psfs

        assert channels is None or len(channels) == shape[0]
        self.channels = channels
        self.dtype = dtype

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

    def __init__(self, images, psfs=None, weights=None, wcs=None, channels=None, padding=10,
                 inverse_kernel=None):
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
        self.frame = Frame(images.shape, wcs=wcs, psfs=psfs, channels=channels, dtype=images.dtype)

        self.images = np.array(images)
        if weights is not None:
            self.weights = np.array(weights)
        else:
            self.weights = 1

        self._padding = padding
        self._inverse = inverse_kernel

    def match(self, model_frame, inverse=True, window=None):
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
        self.window = window
        if self.frame.dtype != model_frame.dtype:
            msg = "Dtypes of model and observation different. Casting observation to {}"
            msg = msg.format(model_frame.dtype)
            logger.warning(msg)
            self.frame.dtype = model_frame.dtype
            self.images = self.images.astype(model_frame.dtype)
            if type(self.weights) is np.ndarray:
                self.weights = self.weights.astype(model_frame.dtype)
            if self.frame._psfs is not None:
                self.frame.psfs.update_dtype(model_frame.dtype)

        #  channels of model that are represented in this observation
        self._band_slice = slice(None)
        if self.frame.channels is not model_frame.channels:
            assert self.frame.channels is not None and model_frame.channels is not None
            bmin = model_frame.channels.index(self.frame.channels[0])
            bmax = model_frame.channels.index(self.frame.channels[-1])
            self._band_slice = slice(bmin, bmax + 1)

        self._diff_kernels = None
        if self.frame.psfs is not model_frame.psfs:
            assert self.frame.psfs is not None and model_frame.psfs is not None
            self._diff_kernels = fft.match_psfs(self.frame.psfs, model_frame.psfs)

            # Also invert the difference kernel to allow deconvolutions.
            # This can be used for initialization to create a better
            # initial model.
            if inverse:
                self._inverse_kernels = fft.match_psfs(model_frame.psfs, self.frame.psfs, window=window)
            else:
                self._inverse_kernels = None

        return self

    def _convolve(self, model):
        """Convolve the model in a single band
        """
        return fft.convolve(fft.Fourier(model, axes=(1, 2)), self._diff_kernels, window=self.window).image

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
        if self._diff_kernels is not None:
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

    def __init__(self, images, wcs=None, psfs=None, weights=None, channels=None, padding=3, operator = 'exact'):

        assert wcs is not None, "WCS is necessary for LowResObservation"
        assert psfs is not None, "PSFs are necessary for LowResObservation"
        assert operator in ['exact', 'bilinear', 'SVD']

        super().__init__(images, wcs=wcs, psfs=psfs, weights=weights, channels=channels, padding=padding)

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

        psf_lr = self.frame.psfs.image
        wcs_lr = self.frame.wcs

        ny_hr, nx_hr = psf_hr.shape
        npsf_bands, ny_lr, nx_lr = psf_lr.shape

        # Createsa wcs for psfs centered around the frame center
        psf_wcs_hr = wcs_hr.deepcopy()
        psf_wcs_lr = wcs_lr.deepcopy()

        if psf_wcs_hr.naxis == 2:
            psf_wcs_hr.wcs.crval = 0., 0.
            psf_wcs_hr.wcs.crpix = ny_hr / 2.+1, nx_hr / 2.+1
        elif psf_wcs_hr.naxis == 3:
            psf_wcs_hr.wcs.crval = 0., 0., 0.
            psf_wcs_hr.wcs.crpix = ny_hr / 2+1, nx_hr / 2.+1, 0.
        if psf_wcs_lr.naxis == 2:
            psf_wcs_lr.wcs.crval = 0., 0.
            psf_wcs_lr.wcs.crpix = ny_lr / 2.+1, nx_lr / 2.+1
        elif psf_wcs_lr.naxis == 3:
            psf_wcs_lr.wcs.crval = 0., 0., 0.
            psf_wcs_lr.wcs.crpix = ny_lr / 2.+1, nx_lr / 2.+1, 0

        pcoordlr_lr, pcoordlr_hr, coordover_hr = resampling.match_patches(psf_hr.shape, psf_lr.data.shape[1:],
                                                                    psf_wcs_hr, psf_wcs_lr, isrot = False, psf = True)

        npsf_y = np.max(coordover_hr[0])-np.min(coordover_hr[0])+1
        npsf_x = np.max(coordover_hr[1])-np.min(coordover_hr[1])+1

        psf_match_lr = interpolation.sinc_interp(coordover_hr, pcoordlr_hr,
                                            psf_lr[:,pcoordlr_lr[0].min():pcoordlr_lr[1].max()+1,pcoordlr_lr[1].min():
                                            pcoordlr_lr[1].max()+1]).reshape(npsf_bands, npsf_y, npsf_x)
        psf_match_hr = psf_hr[coordover_hr[0].min():coordover_hr[0].max()+1,
                       coordover_hr[1].min():coordover_hr[1].max()+1].reshape(npsf_y, npsf_x)

        assert np.shape(psf_match_lr[0]) == np.shape(psf_match_hr)

        psf_match_hr /= np.sum(psf_match_hr)
        psf_match_lr /= np.sum(psf_match_lr)
        return psf_match_hr[np.newaxis, :], psf_match_lr

    def build_diffkernel(self, model_frame):
        '''Builds the differential convolution kernel between the observation and the frame psfs

        Parameters
        ----------
        model_frame: Frame object
            the frame of the model (hehehe)
        Returns
        -------
        diff_psf: array
            the differential psf between observation and frame psfs.
        '''
        # Compute diff kernel at hr
        whr = model_frame.wcs

        # Reference PSF
        _target = model_frame.psfs.image[0, :, :]

        _fftpack_shape = [fftpack.helper.next_fast_len(d) for d in _target.shape]

        while _fftpack_shape[-1] % 2 != 0:
            k_shape = np.array(_fftpack_shape) + 1
            _fftpack_shape = [fftpack.helper.next_fast_len(k_s) for k_s in k_shape]

        # Computes spatially matching observation and target psfs. The observation psf is also resampled \\
        # to the model frame resolution
        new_target, observed_psfs = self.match_psfs(_target, whr)
        target_fft = np.fft.rfftn(new_target[0], _fftpack_shape)
        sel = target_fft == 0
        observed_ffts = np.fft.rfftn(observed_psfs, _fftpack_shape, axes=(1, 2))

        # Computes the diff kernel in Fourier
        kernel_fft = observed_ffts / target_fft
        kernel_fft[:, sel] = 0
        kernel = np.fft.irfftn(kernel_fft, _fftpack_shape, axes=(1, 2))
        kernel = np.fft.ifftshift(kernel, axes=(1, 2))

        if kernel.shape[1] % 2 == 0:
            kernel = kernel[:, 1:, 1:]

        kernel = _centered(kernel, observed_psfs.shape)
        diff_psf = kernel / kernel.sum()

        return diff_psf

    def match(self, model_frame):

        if self.frame.dtype != model_frame.dtype:
            msg = "Dtypes of model and observation different. Casting observation to {}"
            logger.warning(msg.format(model_frame.dtype))
            self.frame.dtype = model_frame.dtype
            self.images = self.images.astype(model_frame.dtype)
            if type(self.weights) is np.ndarray:
                self.weights = self.weights.astype(model_frame.dtype)
            if self.frame._psfs is not None:
                self.frame._psfs.update_dtype(model_frame.dtype)

        #  channels of model that are represented in this observation
        self._band_slice = slice(None)
        if self.frame.channels is not model_frame.channels:
            bmin = model_frame.channels.index(self.frame.channels[0])
            bmax = model_frame.channels.index(self.frame.channels[-1])
            self._band_slice = slice(bmin, bmax+1)

        # Angle between datasets
        rot = np.cross(np.sum(self.frame.wcs.wcs.pc, axis=0)[:2], np.sum(model_frame.wcs.wcs.pc, axis=0)[:2])
        self.isrot = (np.abs(rot) % np.pi) > np.finfo(float).eps

        # Get pixel coordinates in each frame.
        coord_lr, coord_hr, coordhr_over = resampling.match_patches(model_frame.shape, self.frame.shape,
                                                                    model_frame.wcs, self.frame.wcs, isrot = self.isrot)

        #shape of the low resolutino image in the overlap or union
        self.lr_shape = (np.max(coord_lr[0])-np.min(coord_lr[0])+1,np.max(coord_lr[1])-np.min(coord_lr[1])+1)

        #Coordinates of overlapping low resolutions pixels at low resolution
        self._coord_lr = coord_lr
        #Coordinates of overlaping low resolution pixels in high resolution frame
        self._coord_hr = coord_hr
        #Coordinates for all model frame pixels
        self.frame_coord = (np.array(range(model_frame.Ny)), np.array(range(model_frame.Nx)))

        diff_psf = self.build_diffkernel(model_frame)

        #Padding the psf to the frame size
        #At the moment, this only handles the case where size(psf)<size(frame), which,
        # I think is the only one that should arise anyway
        pady = np.ceil((model_frame.shape[1] - diff_psf.shape[1]) / 2.).astype(int)
        padx = np.ceil((model_frame.shape[2] - diff_psf.shape[2]) / 2.).astype(int)
        diff_psf = np.pad(diff_psf, ((0, 0), (pady, pady), (padx, padx)), 'constant')

        # 1D convolutions convolutions of the model are done along the smaller axis, therefore,
        # psf is convolved along the frame's longer axis.
        #the smaller frame axis:
        self.small_axis = (self.frame.Nx <= self.frame.Ny)

        # Coordinates for all psf pixels in model frame (centered on the frame's centre)
        # Choose the optimal shape for FFTPack DFT
        psf_fast = fftpack.helper.next_fast_len(diff_psf.shape[1 + (not self.small_axis)])

        # autograd.numpy.fft does not currently work
        # if the last dimension is odd
        while psf_fast % 2 != 0:
            psf_fast = fftpack.helper.next_fast_len(psf_fast+1)

        self.obs_fast = fftpack.helper.next_fast_len(model_frame.shape[1 + (self.small_axis)])

        # autograd.numpy.fft does not currently work
        # if the last dimension is odd
        while psf_fast % 2 != 0:
            self.obs_fast = fftpack.helper.next_fast_len(self.obs_fast + 1)

        # Computes the resampling/convolution matrix
        resconv_op = []
        for dpsf in diff_psf:

            resconv_temp = sinc_shift_1D(dpsf, self._coord_hr[(not self.small_axis)], (not self.small_axis), psf_fast)
            resconv_shape = np.shape(resconv_temp)

            if (not self.small_axis):
                resconv_op.append(np.reshape(resconv_temp,(resconv_shape[0] * resconv_shape[1], resconv_shape[2])))
            else:
                resconv_op.append(np.reshape(resconv_temp, (resconv_shape[0], resconv_shape[1] * resconv_shape[2])))


        self._resconv_op = np.array(resconv_op, dtype=self.frame.dtype)*(model_frame.Ny*model_frame.Nx)/(self.frame.Ny*
                                                                                                         self.frame.Nx)

        return self


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
        model_image = []
        for c in range(self.frame.C):

            model_conv1d = sinc_shift_1D(model_[c], self._coord_hr[self.small_axis], self.small_axis, self.obs_fast)
            model_shape = np.shape(model_conv1d)

            # I dont KNOW why this stupid axis is inverted and it KILLS ME!!!!

            if (self.small_axis):
                model_image.append(np.dot(self._resconv_op[c],model_conv1d.reshape(model_shape[0]*model_shape[1],
                                                                                model_shape[2]))[:,::-1])
            else:
                model_image.append(np.dot(model_conv1d.reshape(model_shape[0]*model_shape[1], model_shape[2]),
                                        self._resconv_op[c])[::-1,:])

        model_image = np.array(model_image, dtype=self.frame.dtype)

        return model_image

    def render(self, model):
        """Resample and convolve a model in the observation frame for display only!
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


        img[:, np.min(self._coord_lr[0]).astype(int):np.max(self._coord_lr[0]).astype(int)+1,
                     np.min(self._coord_lr[1]).astype(int):np.max(self._coord_lr[1]).astype(int)+1] = \
        self._render(model)
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
                model_ - self.images[:, np.min(self._coord_lr[0]).astype(int):np.max(self._coord_lr[0]).astype(int)+1,
                         np.min(self._coord_lr[1]).astype(int):np.max(self._coord_lr[1]).astype(int)+1])) ** 2)

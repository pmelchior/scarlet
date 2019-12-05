import autograd.numpy as np

from . import interpolation
from . import fft
from . import resampling

import logging

logger = logging.getLogger("scarlet.observation")


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
                psfs = fft.Fourier(psfs)
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
        self.frame = Frame(images.shape, wcs=wcs, psfs=psfs, channels=channels, dtype=images.dtype)

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

        return self

    def _convolve(self, model):
        """Convolve the model in a single band
        """
        return fft.convolve(fft.Fourier(model), self._diff_kernels, axes=(1, 2)).image

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

    def match_psfs(self, psf_hr, wcs_hr, angle):
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
        angle: tuple
            the cos and sin of the rotation angle between frames
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

        psf_valid = psf_lr[:,pcoordlr_lr[0].min():pcoordlr_lr[1].max()+1,pcoordlr_lr[1].min():pcoordlr_lr[1].max()+1]\

        psf_match_lr = interpolation.sinc_interp(psf_valid, coordover_hr, pcoordlr_hr, angle = angle)

        nx_target = ny_lr/self.h
        if (nx_target % 2) == 0:
            nx_target += 1

        assert np.shape(psf_match_lr[0]) == np.shape(psf_hr)

        psf_hr /= np.sum(psf_hr)
        psf_match_lr /= np.sum(psf_match_lr)

        return psf_hr, psf_match_lr

    def build_diffkernel(self, model_frame, angle):
        '''Builds the differential convolution kernel between the observation and the frame psfs

        Parameters
        ----------
        model_frame: Frame object
            the frame of the model (hehehe)
        angle: tuple
            tuple of the cos and sin of the rotation angle between frames.
        Returns
        -------
        diff_psf: array
            the differential psf between observation and frame psfs.
        '''
        # Compute diff kernel at hr
        whr = model_frame.wcs

        # Reference PSF
        _target = model_frame.psfs.image[0, :, :]

        # Computes spatially matching observation and target psfs. The observation psf is also resampled \\
        # to the model frame resolution

        new_target, observed_psfs = self.match_psfs(_target, angle)
        diff_psf = fft.match_psfs(fft.Fourier(observed_psfs), fft.Fourier(new_target))

        return diff_psf

    def sinc_shift(self, imgs, shifts, axes):
        '''Performs 2 1D sinc convolutions and shifting along one rotated axis in Fourier space.

        Parameters
        ----------
        imgs: Fourier
            a Fourier object of 2D images to sinc convolve and shift
            to the adequate shape.
        shifts: array
            an array of the shift values for each line and columns of images in imgs
        axes: array
            Optional argument that specifies the axes along which to apply sinc convolution.
        Returns
        -------
        result: array
            the shifted and sinc convolved array in configuration space
        '''
        # fft
        axes = tuple(np.array(axes)-1)
        fft_shape = np.array(self._fft_shape)[tuple([axes])]
        imgs_fft = imgs.fft(fft_shape, np.array(axes)+1)
        transformed_shape = np.array(imgs_fft.shape[1:])
        transformed_shape[tuple([axes])] = fft_shape

        # frequency sampling
        if len(axes) == 1:
            shifter = np.array(interpolation.mk_shifter(self._fft_shape, real = True))
        else:
            shifter = np.array(interpolation.mk_shifter(self._fft_shape))

        # Shift
        if 0 in axes:
            # Fourier shift
            shishift = np.exp(shifter[0][np.newaxis, :] * shifts[0][:, np.newaxis])
            imgs_shiftfft = imgs_fft[:, np.newaxis, :, :] * shishift[np.newaxis, :, :, np.newaxis]
            fft_axes = [len(imgs_shiftfft.shape) - 2]
            # Shift along the x-axis
            if 1 in axes:
                # Fourier shift
                shishift = np.exp(shifter[1][np.newaxis, :] * shifts[1][:,np.newaxis])
                imgs_shiftfft = imgs_shiftfft * shishift[np.newaxis, :, np.newaxis, :]
                fft_axes = np.array(axes)+len(imgs_shiftfft.shape)-2
            inv_shape = tuple(imgs_shiftfft.shape[:2]) + tuple(transformed_shape)


        elif 1 in axes:
            # Fourier shift
            shishift = np.exp(shifter[1][:, np.newaxis] * shifts[1][np.newaxis, :])
            imgs_shiftfft = imgs_fft[:, :, :, np.newaxis] * shishift[np.newaxis, np.newaxis, :, :]
            inv_shape = tuple([imgs_shiftfft.shape[0]]) + tuple(transformed_shape) + tuple([imgs_shiftfft.shape[-1]])
            fft_axes = [len(imgs_shiftfft.shape)-2]

        # Inverse Fourier transform.
        # The n-dimensional transform could pose problem for very large images
        op = fft.Fourier.from_fft(imgs_shiftfft, fft_shape, inv_shape, fft_axes).image
        return op

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

        # channels of model that are represented in this observation
        self._band_slice = slice(None)
        if self.frame.channels is not model_frame.channels:
            bmin = model_frame.channels.index(self.frame.channels[0])
            bmax = model_frame.channels.index(self.frame.channels[-1])
            self._band_slice = slice(bmin, bmax+1)

        # Affine transform
        try :
            model_affine = model_frame.wcs.wcs.pc
        except AttributeError:
            model_affine = model_frame.wcs.cd
        try:
            self_affine = self.frame.wcs.wcs.pc
        except AttributeError:
            self_affine = self.frame.wcs.cd

        model_pix = np.sqrt(np.abs(model_affine[0,0])*np.abs(model_affine[1,1]-model_affine[0,1]*model_affine[1,0]))
        self_pix = np.sqrt(np.abs(self_affine[0,0])*np.abs(self_affine[1,1]-self_affine[0,1]*self_affine[1,0]))

        # Pixel scale ratio
        self.h = self_pix/model_pix
        #Vector giving the direction of the x-axis of each frame
        self_framevector = np.sum(self_affine, axis=0)[:2]/self_pix
        model_framevector = np.sum(model_affine, axis=0)[:2]/model_pix
        # normalisation
        self_framevector /= np.sum(self_framevector**2)**0.5
        model_framevector /= np.sum(model_framevector**2)**0.5

        # sin of the angle between datasets (normalised cross product)
        self.sin_rot = np.cross(self_framevector, model_framevector)
        # cos of the angle. (normalised scalar product)
        self.cos_rot = np.dot(self_framevector, model_framevector)
        # Is the angle larger than machine precision?
        self.isrot = (np.abs(self.sin_rot)**2) > np.finfo(float).eps
        if not self.isrot:
            self.sin_rot = 0
            self.cos_rot = 1
            angle = None
        else:
            angle = (self.cos_rot, self.sin_rot)

        # Get pixel coordinates in each frame.
        coord_lr, coord_hr, coordhr_over = resampling.match_patches(model_frame.shape, self.frame.shape,
                                                                    model_frame.wcs, self.frame.wcs,
                                                                    perimeter = 'union', isrot = self.isrot)

        # shape of the low resolutino image in the overlap or union
        self.lr_shape = (np.max(coord_lr[0])-np.min(coord_lr[0])+1,np.max(coord_lr[1])-np.min(coord_lr[1])+1)

        # Coordinates of overlapping low resolutions pixels at low resolution
        self._coord_lr = coord_lr
        # Coordinates of overlaping low resolution pixels in high resolution frame
        self._coord_hr = coord_hr
        # Coordinates for all model frame pixels
        self.frame_coord = (np.array(range(model_frame.Ny)), np.array(range(model_frame.Nx)))

        diff_psf = self.build_diffkernel(model_frame, angle)

        # 1D convolutions convolutions of the model are done along the smaller axis, therefore,
        # psf is convolved along the frame's longer axis.
        # the smaller frame axis:
        self.small_axis = (self.frame.Nx <= self.frame.Ny)

        self._fft_shape = fft._get_fft_shape(model_frame.psfs, np.zeros(model_frame.shape), padding=3,
                                             axes=[-2, -1], max=True)

        center_y = np.int(self._fft_shape[0]/2.-(self._fft_shape[0]-model_frame.Ny)/2.) - \
                   ((self._fft_shape[0] % 2) != 0) * ((model_frame.Ny % 2) == 0)
        center_x = np.int(self._fft_shape[1]/2.-(self._fft_shape[1]-model_frame.Nx)/2.) - \
                   ((self._fft_shape[1] % 2) != 0) * ((model_frame.Nx % 2) == 0)

        self.diff_psf = fft.Fourier(fft._pad(diff_psf.image, self._fft_shape, axes = (1,2)))
        if self.isrot:

            #Unrotated coordinates:
            Y_unrot = ((self._coord_hr[0] - center_y) * self.cos_rot +
                       (self._coord_hr[1] - center_x) * self.sin_rot).reshape(self.lr_shape)
            X_unrot = ((self._coord_hr[1] - center_x) * self.cos_rot -
                       (self._coord_hr[0] - center_y) * self.sin_rot).reshape(self.lr_shape)

            #Removing redundancy
            self.Y_unrot = Y_unrot[:, 0]
            self.X_unrot = X_unrot[0, :]

            if self.small_axis:
                self.shifts = np.array([self.Y_unrot * self.cos_rot, self.Y_unrot * self.sin_rot])
                self.other_shifts = np.array([-self.sin_rot * self.X_unrot, self.cos_rot * self.X_unrot])
            else:
                self.shifts = np.array([-self.sin_rot * self.X_unrot, self.cos_rot * self.X_unrot])
                self.other_shifts = np.array([self.Y_unrot * self.cos_rot, self.Y_unrot * self.sin_rot])

            axes = (1,2)

        #aligned case.
        else:

            axes = [int(not self.small_axis)+1]
            self.shifts = np.array(self._coord_hr)

            self.shifts[0] -= center_y
            self.shifts[1] -= center_x

            self.other_shifts = np.copy(self.shifts)

        # Computes the resampling/convolution matrix
        resconv_op = self.sinc_shift(self.diff_psf, self.shifts, axes)

        resconv_op = self.sinc_shift(self.diff_psf, self.shifts, axes)


        self._resconv_op = np.array(resconv_op, dtype=self.frame.dtype)*self.h**2

        if self.isrot:
            self._resconv_op = self._resconv_op.reshape(*self._resconv_op.shape[:2], -1)
            return self
        if self.small_axis:
            self._resconv_op = self._resconv_op.reshape(*self._resconv_op.shape[:2], -1)
            return self
        else:
            self._resconv_op = self._resconv_op.reshape(self._resconv_op.shape[0], -1, self._resconv_op.shape[-1])
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
        # Padding the psf to the fast_shape size
        model_ = fft.Fourier(fft._pad(model[self._band_slice, :, :], self._fft_shape, axes=(-2, -1)))

        model_image = []
        if self.isrot:
            axes = (1,2)

        else:
            axes = [int(self.small_axis)+1]

        model_conv = self.sinc_shift(model_, -self.other_shifts, axes)
        #Transposes are all over the place to make arrays F-contiguous
        if self.isrot:
            if self.small_axis:
                model_conv = model_conv.reshape(*model_conv.shape[:2], -1)
                for c in range(self.frame.C):
                    model_image.append((self._resconv_op[c] @ model_conv[c].T))
                return np.array(model_image, dtype=self.frame.dtype)
            else:
                model_conv = model_conv.reshape(*model_conv.shape[:2], -1)
                for c in range(self.frame.C):
                    model_image.append((self._resconv_op[c] @ model_conv[c].T).T)
                return np.array(model_image, dtype=self.frame.dtype)


        if self.small_axis:
            model_conv = model_conv.reshape(model_conv.shape[0],-1,model_conv.shape[-1])
            for c in range(self.frame.C):
                model_image.append((model_conv[c].T @ self._resconv_op[c].T).T)
            return np.array(model_image, dtype=self.frame.dtype)
        else:
            model_conv = model_conv.reshape(*model_conv.shape[:2], -2)
            for c in range(self.frame.C):
                model_image.append((self._resconv_op[c].T @ model_conv[c].T).T)
            return np.array(model_image, dtype=self.frame.dtype)


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

        img[:, np.min(self._coord_lr[0]).astype(int):np.max(self._coord_lr[0]).astype(int) + 1,
        np.min(self._coord_lr[1]).astype(int):np.max(self._coord_lr[1]).astype(int) + 1] = \
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

        min_lr = [np.min(self._coord_lr[0]).astype(int), np.min(self._coord_lr[1]).astype(int)]
        max_lr = [np.max(self._coord_lr[0]).astype(int), np.max(self._coord_lr[1]).astype(int)]

        return 0.5 * np.sum((self.weights * (
                model_ -
                self.images[:, min_lr[0]:max_lr[0] + 1, min_lr[1]:max_lr[1] + 1])) ** 2)

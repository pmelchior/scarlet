import autograd.numpy as np

from .frame import Frame
from . import interpolation
from . import fft
from . import resampling
from .bbox import Box


class Observation:
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

    def __init__(
            self, images, psfs=None, weights=None, wcs=None, channels=None, padding=10
    ):
        """Create an Observation

        Parameters
        ---------
        images: array or tensor
            3D data cube (Channel, Height, Width) of the image in each band.
        psfs: `scarlet.PSF` or its arguments
            PSF in each channel. Can be 3D cube of images stacked in channel direction.
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
        self.frame = Frame(
            images.shape, wcs=wcs, psfs=psfs, channels=channels, dtype=images.dtype
        )

        self.images = images
        if weights is not None:
            self.weights = weights
        else:
            self.weights = np.ones(images.shape)
        assert (
                self.weights.shape == self.images.shape
        ), "Weights needs to have same shape as images"

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
        # find the box that contained this obs in model_frame
        shape = self.images.shape
        yx0 = model_frame.get_pixel(self.frame.get_sky_coord((-model_frame.origin[-2], -model_frame.origin[-1])))
        #  channels of model that are represented in this observation
        if self.frame.channels is model_frame.channels:
            origin = (0, *yx0)
        else:
            assert self.frame.channels is not None and model_frame.channels is not None
            cmin = list(model_frame.channels).index(self.frame.channels[0])
            origin = (cmin, *yx0)
        self.bbox = Box(shape, origin=origin)
        self.model_slices = self.bbox.slices_for(model_frame.shape)

        # check dtype consistency
        if self.frame.dtype != model_frame.dtype:
            self.frame.dtype = model_frame.dtype
            self.images = self.images.copy().astype(model_frame.dtype)
            if type(self.weights) is np.ndarray:
                self.weights = self.weights.copy().astype(model_frame.dtype)

        # constrcut diff kernels
        self._diff_kernels = None
        if self.frame.psf is not model_frame.psf:
            assert self.frame.psf is not None and model_frame.psf is not None
            psf = fft.Fourier(self.frame.psf.update_dtype(model_frame.dtype).image)
            model_psf = fft.Fourier(
                model_frame.psf.update_dtype(model_frame.dtype).image
            )
            self._diff_kernels = fft.match_psfs(psf, model_psf)

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
        image_model: array
            `model` mapped into the observation frame
        """

        if self._diff_kernels is not None:
            image_model = self._convolve(model)[self.model_slices]
        else:
            image_model = model[self.model_slices]

        return image_model

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

        model_ = self.render(model)
        images_ = self.images
        weights_ = self.weights

        # normalization of the single-pixel likelihood:
        # 1 / [(2pi)^1/2 (sigma^2)^1/2]
        # with inverse variance weights: sigma^2 = 1/weight
        # full likelihood is sum over all data samples: pixel in images
        # NOTE: this assumes that all pixels are used in likelihood!
        log_sigma = np.zeros(self.weights.shape, dtype=self.weights.dtype)
        cuts = self.weights > 0
        log_sigma[cuts] = np.log(1 / self.weights[cuts])
        log_norm = (
                np.prod(images_.shape) / 2 * np.log(2 * np.pi)
                + np.sum(log_sigma) / 2
        )

        return log_norm + np.sum(weights_ * (model_ - images_) ** 2) / 2

    def get_LowRes(self):
        """ Creates a LowResObservation object from an Observation object

        """
        return LowResObservation(self.images,
                                 psfs=self.frame._psfs,
                                 weights=self.weights,
                                 wcs=self.frame.wcs,
                                 channels=self.frame.channels)


class LowResObservation(Observation):
    def __init__(
            self,
            images,
            wcs=None,
            psfs=None,
            weights=None,
            channels=None,
            padding=3,
    ):

        assert wcs is not None, "WCS is necessary for LowResObservation"
        assert psfs is not None, "PSF is necessary for LowResObservation"

        super().__init__(
            images,
            wcs=wcs,
            psfs=psfs,
            weights=weights,
            channels=channels,
            padding=padding,
        )

    def match_psfs(self, psf_hr, wcs_hr):
        """psf matching between different dataset
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
        """
        psf_lr = self.frame._psfs.image
        # Odd pad shape
        pad_shape = np.array((self.images.shape[-2:] + np.array(psf_lr.shape[-2:]))/2).astype(int) * 2 + 1
        wcs_lr = self.frame.wcs

        h_lr = interpolation.get_pixel_size(interpolation.get_affine(wcs_lr))
        h_hr = interpolation.get_pixel_size(interpolation.get_affine(wcs_hr))

        # Interpolation of the low res psf
        angle, h_ratio = interpolation.get_angles(wcs_hr, wcs_lr)
        psf_match_lr = interpolation.sinc_interp_inplace(psf_lr, h_lr, h_hr, angle, pad_shape = pad_shape)

        # Normalisation
        psf_hr /= np.sum(psf_hr)
        psf_match_lr /= np.sum(psf_match_lr)

        return psf_hr, psf_match_lr

    def build_diffkernel(self, model_frame):
        """Builds the differential convolution kernel between the observation and the frame psfs

        Parameters
        ----------
        model_frame: Frame object
            the frame of the model
        Returns
        -------
        diff_psf: array
            the differential psf between observation and frame psfs.
        """
        # Compute diff kernel at hr
        whr = model_frame.wcs

        # Reference PSF
        _target = model_frame._psfs.image

        # Computes spatially matching observation and target psfs. The observation psf is also resampled \\
        # to the model frame resolution
        new_target, observed_psfs = self.match_psfs(_target, whr)
        diff_psf = fft.match_psfs(fft.Fourier(observed_psfs), fft.Fourier(new_target))

        return diff_psf

    def sinc_shift(self, imgs, shifts, axes):
        """Performs 2 1D sinc convolutions and shifting along one rotated axis in Fourier space.

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
        """
        # fft
        axes = tuple(np.array(axes) - 1)
        fft_shape = np.array(self._fft_shape)[tuple([axes])]
        imgs_fft = imgs.fft(fft_shape, np.array(axes) + 1)
        transformed_shape = np.array(imgs_fft.shape[1:])
        transformed_shape[tuple([axes])] = fft_shape

        # frequency sampling
        if len(axes) == 1:
            shifter = np.array(interpolation.mk_shifter(self._fft_shape, real=True))
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
                shishift = np.exp(shifter[1][np.newaxis, :] * shifts[1][:, np.newaxis])
                imgs_shiftfft = imgs_shiftfft * shishift[np.newaxis, :, np.newaxis, :]
                fft_axes = np.array(axes) + len(imgs_shiftfft.shape) - 2
            inv_shape = tuple(imgs_shiftfft.shape[:2]) + tuple(transformed_shape)


        elif 1 in axes:
            # Fourier shift
            shishift = np.exp(shifter[1][:, np.newaxis] * shifts[1][np.newaxis, :])
            imgs_shiftfft = imgs_fft[:, :, :, np.newaxis] * shishift[np.newaxis, np.newaxis, :, :]
            inv_shape = tuple([imgs_shiftfft.shape[0]]) + tuple(transformed_shape) + tuple([imgs_shiftfft.shape[-1]])
            fft_axes = [len(imgs_shiftfft.shape) - 2]

        # Inverse Fourier transform.
        # The n-dimensional transform could pose problem for very large images
        op = fft.Fourier.from_fft(imgs_shiftfft, fft_shape, inv_shape, fft_axes).image
        return op

    def match(self, model_frame):
        """ matches the observation to a frame

        Parameters
        ----------
        model_frame: `Frame`
            Frame to match to the observation
        coord: `array`
            coordinates of the pixels in the frame to fit
        """
        if self.frame.dtype != model_frame.dtype:
            self.images = self.images.copy().astype(model_frame.dtype)
            if type(self.weights) is np.ndarray:
                self.weights = self.weights.copy().astype(model_frame.dtype)
            if self.frame._psfs is not None:
                self.frame._psfs.update_dtype(model_frame.dtype)

        self.angle, self.h = interpolation.get_angles(self.frame.wcs, model_frame.wcs)

        # Is the angle larger than machine precision?
        self.isrot = (np.abs(self.angle[1]) ** 2) > np.finfo(float).eps
        if not self.isrot:
            self.angle = None

        # Get pixel coordinates in each frame.
        coord_lr, coord_hr = resampling.match_patches(self, model_frame,isrot=self.isrot)
        # shape of the low resolution image in the intersection or union
        self.lr_shape = (
            np.max(coord_lr[0]) - np.min(coord_lr[0]) + 1,
            np.max(coord_lr[1]) - np.min(coord_lr[1]) + 1,
        )

        # BBox of the low resolution pixels in model frame
        #  1) channels of model that are represented in this observation
        if self.frame.channels is not model_frame.channels:
            cmin = model_frame.channels.index(self.frame.channels[0])
            cmax = model_frame.channels.index(self.frame.channels[-1])
        else:
            cmin, cmax = 0, self.frame.C
        # 2) use the bounds of coord_hr
        self.bbox = Box(
            (cmax + 1 - cmin,
            np.around(np.max(coord_lr[0]) + 1 - np.min(coord_lr[0])).astype(int),
            np.around(np.max(coord_lr[1]) + 1 - np.min(coord_lr[1])).astype(int),
        ))
        # Slice of the frame that contains the observation
        self.model_slices = self.bbox.slices_for(model_frame.shape)
        # Coordinates for all model frame pixels
        self.frame_coord = (
            np.array(range(model_frame.Ny)),
            np.array(range(model_frame.Nx)),
        )
        diff_psf = self.build_diffkernel(model_frame)

        # 1D convolutions convolutions of the model are done along the smaller axis, therefore,
        # psf is convolved along the frame's longer axis.
        # the smaller frame axis:
        self.small_axis = self.frame.Nx <= self.frame.Ny

        self._fft_shape = fft._get_fft_shape(
            diff_psf,
            np.zeros(model_frame.shape),
            padding=3,
            axes=[-2, -1],
            max=True,
        )
        self.diff_psf = fft.Fourier(fft._pad(diff_psf.image, self._fft_shape, axes=(-2,-1)))

        center_y = np.int(self._fft_shape[0] / 2. - (self._fft_shape[0] - model_frame.Ny) / 2.) + \
                   ((self._fft_shape[0] % 2) != 0) * ((model_frame.Ny % 2) == 0) + model_frame.origin[-2]
        center_x = np.int(self._fft_shape[1] / 2. - (self._fft_shape[1] - model_frame.Nx) / 2.) - \
                   ((self._fft_shape[1] % 2) != 0) * ((model_frame.Nx % 2) == 0) + model_frame.origin[-1]
        if self.isrot:

            # Unrotated coordinates:
            Y_unrot = (
                    (coord_hr[0] - center_y) * self.angle[0]
                    - (coord_hr[1] - center_x) * self.angle[1]
            ).reshape(self.lr_shape)
            X_unrot = (
                    (coord_hr[1] - center_x) * self.angle[0]
                    + (coord_hr[0] - center_y) * self.angle[1]
            ).reshape(self.lr_shape)

            # Removing redundancy
            self.Y_unrot = Y_unrot[:, 0]
            self.X_unrot = X_unrot[0, :]

            if self.small_axis:
                self.shifts = np.array([self.Y_unrot * self.angle[0], self.Y_unrot * self.angle[1]])
                self.other_shifts = np.array([
                    -self.angle[1] * self.X_unrot,
                    self.angle[0] * self.X_unrot,
                ])
            else:
                self.shifts = np.array([
                    -self.angle[1] * self.X_unrot,
                    self.angle[0] * self.X_unrot,
                ])
                self.other_shifts = np.array([
                    self.Y_unrot * self.angle[0],
                    self.Y_unrot * self.angle[1],
                ])

            axes = (1, 2)

        # aligned case.
        else:

            axes = [int(not self.small_axis) + 1]
            self.shifts = np.array(coord_hr)

            self.shifts[0] -= center_y
            self.shifts[1] -= center_x

            self.other_shifts = np.copy(self.shifts)
        # Computes the resampling/convolution matrix
        resconv_op = self.sinc_shift(self.diff_psf, self.shifts, axes)

        self._resconv_op = np.array(resconv_op, dtype=self.frame.dtype) * self.h ** 2

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
        image_model: array
            `model` mapped into the observation frame
        """
        # Padding the psf to the fast_shape size
        model_ = fft.Fourier(
            fft._pad(model, self._fft_shape, axes=(-2, -1))
        )

        model_image = []
        if self.isrot:
            axes = (1, 2)

        else:
            axes = [int(self.small_axis) + 1]

        model_conv = self.sinc_shift(model_, -self.other_shifts, axes)
        # Transposes are all over the place to make arrays F-contiguous
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
            model_conv = model_conv.reshape(model_conv.shape[0], -1, model_conv.shape[-1])
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
        image_model: array
            `model` mapped into the observation frame
        """

        image_model = self._render(model)[self.model_slices]
        return image_model

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
        images_ = self.images[:, self.model_slices[-2], self.model_slices[-1]]
        weights_ = self.weights[:, self.model_slices[-2], self.model_slices[-1]]

        # properly normalized likelihood
        log_sigma = np.zeros(weights_.shape, dtype=weights_.dtype)
        cuts = weights_ > 0
        log_sigma[cuts] = np.log(1 / weights_[cuts])
        log_norm = (
                np.prod(images_.shape) / 2 * np.log(2 * np.pi)
                + np.sum(log_sigma) / 2
        )

        return log_norm + 0.5 * np.sum(weights_ * (model_ - images_) ** 2)

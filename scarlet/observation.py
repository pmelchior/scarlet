import autograd.numpy as np
from autograd.extend import defvjp, primitive

from .frame import Frame
from . import interpolation
from . import fft
from .bbox import Box, overlapped_slices
from scarlet.operators_pybind11 import apply_filter


@primitive
def convolve(image, psf, bounds):
    """Convolve an image with a PSF in real space
    """
    result = np.empty(image.shape, dtype=image.dtype)
    for band in range(len(image)):
        if hasattr(image[band], "_value"):
            # This is an ArrayBox
            img = image[band]._value
        else:
            img = image[band]
        apply_filter(
            img,
            psf[band].reshape(-1),
            bounds[0],
            bounds[1],
            bounds[2],
            bounds[3],
            result[band],
        )
    return result


def _grad_convolve(convolved, image, psf, slices):
    """Gradient of a real space convolution
    """
    return lambda input_grad: convolve(input_grad, psf[:, ::-1, ::-1], slices)


# Register this function in autograd
defvjp(convolve, _grad_convolve)


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

    def __init__(self, images, channels, psfs=None, weights=None, wcs=None, padding=10):
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
            self.weights = np.ones(images.shape, dtype=images.dtype)
        assert (
            self.weights.shape == self.images.shape
        ), "Weights needs to have same shape as images"
        self._padding = padding
        self._diff_kernels = None
        self._slices_for_model = slice(None)
        self._slices_for_images = slice(None)
        self._parameters = ()

    def match(self, model_frame, convolution_type="fft"):
        """Match the frame of `Blend` to the frame of this observation.

        The method sets up the mappings in spectral and spatial coordinates,
        which includes a spatial selection, computing PSF difference kernels
        and filter transformations.

        Parameters
        ---------
        model_frame: a `scarlet.Frame` instance
            The frame of `Blend` to match
        diff_kernels: array
            The difference kernel for each band.
            If `diff_kernels` is `None` then they are
            calculated automatically.
        convolution_type: str
            The type of convolution to use.
            - `real`: Use a real space convolution and gradient
            - `fft`: Use a DFT to do the convolution and gradient

        Returns
        -------
        None
        """
        self.model_frame = model_frame

        # check dtype consistency
        if self.frame.dtype != model_frame.dtype:
            self.frame.dtype = model_frame.dtype
            self.images = self.images.astype(model_frame.dtype)
            if type(self.weights) is np.ndarray:
                self.weights = self.weights.astype(model_frame.dtype)

        # determine which channels are covered by data
        # this has to be done first because convolution is only determined for these
        self._channel_map = self.get_channel_map_for(model_frame)

        # same of the 2D spatial region covered by data
        pixel_in_model_frame = self.frame.convert_pixel_to(model_frame)
        # since there cannot be rotation or scaling, it's only translation
        ll = np.round(pixel_in_model_frame.min(axis=0)).astype("int")
        ur = np.round(pixel_in_model_frame.max(axis=0)).astype("int") + 1
        bounds = (ll[0], ur[0]), (ll[1], ur[1])
        data_box = model_frame.bbox[0] @ Box.from_bounds(*bounds)
        # properly treats truncation in both boxes
        slices = overlapped_slices(data_box, model_frame.bbox)
        self._slices_for_images = slices[0]
        self._slices_for_model = slices[1]

        # construct diff kernels
        if self.frame.psf is not model_frame.psf:
            assert self.frame.psf is not None and model_frame.psf is not None
            psf = fft.Fourier(self.frame.psf.get_model().astype(model_frame.dtype))
            model_psf = fft.Fourier(
                model_frame.psf.get_model().astype(model_frame.dtype)
            )
            self._diff_kernels = fft.match_psfs(psf, model_psf)

            # construct deconvolved image for detection:
            # divide Fourier transform of images by Fourier transform of diff kernel
            self.prerender_images = fft._kspace_operation(
                fft.Fourier(self.images),
                self._diff_kernels,
                3,
                fft.operator.truediv,
                self.images.shape,
                axes=(-2, -1),
            ).image  # then get the image from Fourier

            # deconvolve noise field to estimate its noise level
            noise = np.random.normal(scale=1 / np.sqrt(self.weights))
            noise_deconv = fft._kspace_operation(
                fft.Fourier(noise),
                self._diff_kernels,
                3,
                fft.operator.truediv,
                self.images.shape,
                axes=(-2, -1),
            ).image
            # spatially flat noise, ignores any variation in self.weights
            self.prerender_sigma = noise_deconv.std(axis=(1, 2))
        else:
            self._diff_kernels = None
            self.prerender_images = self.images
            self.prerender_sigma = np.mean(1 / np.sqrt(self.weights), axis=(1, 2))

        assert convolution_type in [
            "real",
            "fft",
        ], "`convolution` must be either 'real' or 'fft'"
        self._convolution_type = convolution_type
        return self

    @property
    def convolution_bounds(self):
        """Build the slices needed for convolution in real space
        """
        try:
            return self._convolution_bounds
        except AttributeError:
            coords = interpolation.get_filter_coords(self._diff_kernels[0])
            self._convolution_bounds = interpolation.get_filter_bounds(
                coords.reshape(-1, 2)
            )
        return self._convolution_bounds

    def convolve(self, model, convolution_type=None):
        """Convolve the model in a single band
        """
        if convolution_type is None:
            convolution_type = self._convolution_type
        if convolution_type == "real":
            result = convolve(model, self._diff_kernels.image, self._convolution_bounds)
        elif convolution_type == "fft":
            result = fft.convolve(
                fft.Fourier(model), self._diff_kernels, axes=(1, 2)
            ).image
        else:
            raise ValueError(
                "`convolution` must be either 'real' or 'fft', got {}".format(
                    convolution_type
                )
            )
        return result

    @property
    def parameters(self):
        return self._parameters

    def get_channel_map_for(self, target):
        """Compute the mapping between channels in the target frame and this observation

        Parameters
        ----------
        target: a `scarlet.Frame` instance
            The frame to match

        Returns
        -------
        channel_map: None, slice, or array
            None for identical channels in both frames; slice for concatenated channels;
            array for linear mapping of target channels onto observation channels
        """

        if list(self.frame.channels) == list(target.channels):
            return None

        channel_map = [
            list(target.channels).index(c) for c in list(self.frame.channels)
        ]
        min_channel = min(channel_map)
        max_channel = max(channel_map)
        if max_channel + 1 - min_channel == len(channel_map):
            channel_map = slice(min_channel, max_channel + 1)
        return channel_map

        # full-fledged linear mixing model to allow for spectrophotometry later
        channel_map = np.zeros((self.C, target.C))
        for i, c in enumerate(list(self.frame.channels)):
            j = list(target.channels).index(c)
            assert j != -1, f"Could not find channel {c} in target frame"
            channel_map[i, j] = 1
            # TODO: for overlap computation:
            # * turn channels into dict channel['g'] = (lambdas, R_lambdas)
            # * extrapolate obs R_lambda onto model lambdas
            # * compute np.dot(model_R_lambda, obs_R_lambda) for every
            #   combination of obs and model channels
        return channel_map

    def map_channels(self, model):
        """Map to model channels onto the observation channels

        Parameters
        ----------
        model: array
            The hyperspectral model

        Returns
        -------
        obs_model: array
            `model` mapped onto the observation channels
        """
        if self._channel_map is None:
            return model
        if isinstance(self._channel_map, slice):
            return model[self._channel_map]
        return np.dot(self._channel_map, model)

    def render(self, model, *parameters):
        """Convolve a model to the observation frame

        Parameters
        ----------
        model: array
            The hyperspectral model
        parameters: tuple of optimization parameters

        Returns
        -------
        image_model: array
            `model` mapped into the observation frame
        """
        # restrict to observed portion
        model = self.map_channels(model)
        if self._diff_kernels is not None:
            model_ = self.convolve(model)
        else:
            model_ = model
        return model_[self._slices_for_model]

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
        images_ = self.images[self._slices_for_images]
        weights_ = self.weights[self._slices_for_images]

        # noise injection to soften the gradient
        if noise_factor > 0:
            std = 1 / np.sqrt(weights_)
            noise = np.random.normal(loc=0, scale=std)
            images_ = images_.copy() + noise
            weights_ = weights_.copy() / (noise_factor + 1)

        return -self.log_norm - np.sum(weights_ * (model_ - images_) ** 2) / 2

    @property
    def log_norm(self):
        try:
            return self._log_norm
        except AttributeError:
            images_ = self.images[self._slices_for_images]
            weights_ = self.weights[self._slices_for_images]

            # normalization of the single-pixel likelihood:
            # 1 / [(2pi)^1/2 (sigma^2)^1/2]
            # with inverse variance weights: sigma^2 = 1/weight
            # full likelihood is sum over all data samples: pixel in images
            # NOTE: this assumes that all pixels are used in likelihood!
            log_sigma = np.zeros(weights_.shape, dtype=self.weights.dtype)
            cuts = weights_ > 0
            log_sigma[cuts] = np.log(1 / weights_[cuts])
            self._log_norm = (
                np.prod(images_.shape) / 2 * np.log(2 * np.pi) + np.sum(log_sigma) / 2
            )
        return self._log_norm

    def get_LowRes(self):
        """ Creates a LowResObservation object from an Observation object

        """
        return LowResObservation(
            self.images,
            psfs=self.frame._psfs,
            weights=self.weights,
            wcs=self.frame.wcs,
            channels=self.frame.channels,
        )

    def _to_frame(self, frame, images=None):
        """Project this observation into another frame

        Note: the frame must have the same sampling and rotation,
        but can differ in the shape and origin of the `Frame`.
        This method is a convenience function for now but should
        not be considered supported and could be removed in a later version
        """
        frame_slices, observation_slices = overlapped_slices(
            frame.bbox, self.frame.bbox
        )

        if images is None:
            images = self.images

        if hasattr(frame, "dtype"):
            dtype = frame.dtype
        else:
            dtype = images.dtype
        result = np.zeros(frame.shape, dtype=dtype)
        result[frame_slices] = images[observation_slices]
        return result


class LowResObservation(Observation):
    def __init__(
        self, images, channels, wcs=None, psfs=None, weights=None, padding=3,
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
        psf_lr = self.frame.psf.get_model().astype(psf_hr.dtype)
        # Odd pad shape
        pad_shape = (
            np.array((self.images.shape[-2:] + np.array(psf_lr.shape[-2:])) / 2).astype(
                int
            )
            * 2
            + 1
        )
        wcs_lr = self.frame.wcs

        h_lr = interpolation.get_pixel_size(interpolation.get_affine(wcs_lr))
        h_hr = interpolation.get_pixel_size(interpolation.get_affine(wcs_hr))

        # Interpolation of the low res psf
        angle, h_ratio = interpolation.get_angles(wcs_hr, wcs_lr)
        psf_match_lr = interpolation.sinc_interp_inplace(
            psf_lr, h_lr, h_hr, angle, pad_shape=pad_shape
        )

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
        _target = model_frame.psf.get_model()

        # Computes spatially matching observation and target psfs. The observation psf is also resampled \\
        # to the model frame resolution
        new_target, observed_psfs = self.match_psfs(_target, whr)
        diff_psf = fft.match_psfs(fft.Fourier(observed_psfs), fft.Fourier(new_target))

        return diff_psf, new_target

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
            imgs_shiftfft = (
                imgs_fft[:, np.newaxis, :, :] * shishift[np.newaxis, :, :, np.newaxis]
            )
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
            imgs_shiftfft = (
                imgs_fft[:, :, :, np.newaxis] * shishift[np.newaxis, np.newaxis, :, :]
            )
            inv_shape = (
                tuple([imgs_shiftfft.shape[0]])
                + tuple(transformed_shape)
                + tuple([imgs_shiftfft.shape[-1]])
            )
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
        self.model_frame = model_frame

        # check dtype consistency
        if self.frame.dtype != model_frame.dtype:
            self.frame.dtype = model_frame.dtype
            self.images = self.images.astype(model_frame.dtype)
            if type(self.weights) is np.ndarray:
                self.weights = self.weights.astype(model_frame.dtype)

        # determine which channels are covered by data
        # this has to be done first because convolution is only determined for these
        self._channel_map = self.get_channel_map_for(model_frame)

        # check if data is rotated wrt to model_frame
        self.angle, self.h = interpolation.get_angles(self.frame.wcs, model_frame.wcs)
        self.isrot = (np.abs(self.angle[1]) ** 2) > np.finfo(float).eps

        # Get pixel coordinates alinged with x and y axes  of this observation
        # in model frame
        lr_shape = self.frame.shape[1:]
        pixels = np.stack((np.arange(lr_shape[0]), np.arange(lr_shape[1])), axis=1)
        coord_hr = self.frame.convert_pixel_to(model_frame, pixel=pixels)

        # TODO: should coords define a _slices_for_model/images?
        # lr_inside_hr = model_frame.bbox.contains(coord_hr)

        # compute diff kernel in model_frame pixels
        diff_psf, target = self.build_diffkernel(model_frame)

        # 1D convolutions convolutions of the model are done along the smaller axis, therefore,
        # psf is convolved along the frame's longer axis.
        # the smaller frame axis:
        self.small_axis = self.frame.Nx <= self.frame.Ny

        self._fft_shape = fft._get_fft_shape(
            target, np.zeros(model_frame.shape), padding=3, axes=[-2, -1], max=False,
        )
        # Cutting diff_psf if needded and keeping the parity
        if (self._fft_shape[-2] < diff_psf.shape[-2]) or (
            self._fft_shape[-1] < diff_psf.shape[-1]
        ):
            diff_psf = fft._centered(
                diff_psf, np.array([diff_psf.shape[0] + 1, *self._fft_shape]) - 1
            )

        self._diff_kernels = fft.Fourier(
            fft._pad(diff_psf.image, self._fft_shape, axes=(-2, -1))
        )

        center_y = np.int(
            self._fft_shape[0] / 2.0 - (self._fft_shape[0] - model_frame.Ny) / 2.0
        ) + ((self._fft_shape[0] % 2) != 0) * ((model_frame.Ny % 2) == 0)
        center_x = np.int(
            self._fft_shape[1] / 2.0 - (self._fft_shape[1] - model_frame.Nx) / 2.0
        ) - ((self._fft_shape[1] % 2) != 0) * ((model_frame.Nx % 2) == 0)

        if self.isrot:
            # Unrotated coordinates:
            Y_unrot = (
                (coord_hr[:, 0] - center_y) * self.angle[0]
                - (coord_hr[:, 1] - center_x) * self.angle[1]
            ).reshape(lr_shape)
            X_unrot = (
                (coord_hr[:, 1] - center_x) * self.angle[0]
                + (coord_hr[:, 0] - center_y) * self.angle[1]
            ).reshape(lr_shape)

            # Removing redundancy
            self.Y_unrot = Y_unrot[:, 0]
            self.X_unrot = X_unrot[0, :]

            if self.small_axis:
                self.shifts = np.array(
                    [self.Y_unrot * self.angle[0], self.Y_unrot * self.angle[1]]
                )
                self.other_shifts = np.array(
                    [-self.angle[1] * self.X_unrot, self.angle[0] * self.X_unrot,]
                )
            else:
                self.shifts = np.array(
                    [-self.angle[1] * self.X_unrot, self.angle[0] * self.X_unrot,]
                )
                self.other_shifts = np.array(
                    [self.Y_unrot * self.angle[0], self.Y_unrot * self.angle[1],]
                )

            axes = (1, 2)

        # aligned case.
        else:
            axes = [int(not self.small_axis) + 1]
            self.shifts = coord_hr.T

            self.shifts[0] -= center_y
            self.shifts[1] -= center_x

            self.other_shifts = np.copy(self.shifts)
        # Computes the resampling/convolution matrix
        resconv_op = self.sinc_shift(self._diff_kernels, self.shifts, axes)

        self._resconv_op = np.array(resconv_op, dtype=self.frame.dtype) * self.h ** 2

        if self.isrot:
            self._resconv_op = self._resconv_op.reshape(*self._resconv_op.shape[:2], -1)
        elif self.small_axis:
            self._resconv_op = self._resconv_op.reshape(*self._resconv_op.shape[:2], -1)
        else:
            self._resconv_op = self._resconv_op.reshape(
                self._resconv_op.shape[0], -1, self._resconv_op.shape[-1]
            )

        # # TODO: DOES NOT WORK!!!
        # # construct deconvolved image for detection:
        # # * interpolate to model frame
        # # * divide Fourier transform of images by Fourier transform of diff kernel
        # self.prerender_images = interpolation.interpolate_observation(self, model_frame)
        # self.prerender_images = fft._kspace_operation(
        #     fft.Fourier(self.prerender_images),
        #     self._diff_kernels,
        #     3,
        #     fft.operator.truediv,
        #     self.prerender_images.shape,
        #     axes=(-2, -1),
        # ).image  # then get the image from Fourier
        #
        # # do same thing with noise field
        # noise = np.random.normal(scale=1 / np.sqrt(self.weights))
        # # temporarily replace obs.images with noise
        # images = self.images
        # self.images = noise
        # noise = interpolation.interpolate_observation(self, model_frame)
        # self.images = images
        # noise_deconv = fft._kspace_operation(
        #     fft.Fourier(noise),
        #     self._diff_kernels,
        #     3,
        #     fft.operator.truediv,
        #     self.images.shape,
        #     axes=(-2, -1),
        # ).image
        # # spatially flat noise, ignores any variation in self.weights
        # self.prerender_sigma = noise_deconv.std(axis=(1, 2))

        return self

    def render(self, model, *parameters):
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
        # restrict to observed portion
        model = self.map_channels(model)
        # FFT of model, padding the psf to the fast_shape size
        model_ = fft.Fourier(fft._pad(model, self._fft_shape, axes=(-2, -1)))

        if self.isrot:
            axes = (1, 2)
        else:
            axes = [int(self.small_axis) + 1]
        model_conv = self.sinc_shift(model_, -self.other_shifts, axes)

        # Transposes are all over the place to make arrays F-contiguous
        # -> faster than np.einsum
        if self.isrot:
            model_conv = model_conv.reshape(*model_conv.shape[:2], -1)

            if self.small_axis:
                return np.array(
                    [
                        np.dot(self._resconv_op[c], model_conv[c].T)
                        for c in range(self.frame.C)
                    ],
                    dtype=self.frame.dtype,
                )
            else:
                return np.array(
                    [
                        np.dot(self._resconv_op[c], model_conv[c].T).T
                        for c in range(self.frame.C)
                    ],
                    dtype=self.frame.dtype,
                )

        if self.small_axis:
            model_conv = model_conv.reshape(
                model_conv.shape[0], -1, model_conv.shape[-1]
            )
            return np.array(
                [
                    np.dot(model_conv[c].T, self._resconv_op[c].T).T
                    for c in range(self.frame.C)
                ],
                dtype=self.frame.dtype,
            )
        else:
            model_conv = model_conv.reshape(*model_conv.shape[:2], -1)
            return np.array(
                [
                    np.dot(self._resconv_op[c].T, model_conv[c].T).T
                    for c in range(self.frame.C)
                ],
                dtype=self.frame.dtype,
            )

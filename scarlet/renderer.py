import autograd.numpy as np
from autograd.extend import defvjp, primitive

from .model import Model
from . import interpolation
from .parameter import Parameter
from . import fft
from .bbox import Box, overlapped_slices
from scarlet.operators_pybind11 import apply_filter


class Renderer(Model):
    def __init__(self, data_frame, model_frame, *parameters):
        self.data_frame = data_frame
        self.model_frame = model_frame
        # mapping of model to data frame channels
        self.channel_map = self.get_channel_map(data_frame, model_frame)

        super().__init__(*parameters)

    # renderer is a parameterized transformation function
    def __call__(self, model, *parameters):
        self.transform = self.get_model(*parameters)
        return self.transform(model)

    def get_channel_map(self, data_frame, model_frame):
        """Compute the mapping between channels in the model frame and this observation

        Parameters
        ----------
        model_frame: a `scarlet.Frame` instance
            The frame to match

        Returns
        -------
        channel_map: None, slice, or array
            None for identical channels in both frames; slice for concatenated channels;
            array for linear mapping of model channels onto observation channels
        """

        if list(data_frame.channels) == list(model_frame.channels):
            return None

        channel_map = [
            list(model_frame.channels).index(c) for c in list(data_frame.channels)
        ]
        min_channel = min(channel_map)
        max_channel = max(channel_map)
        if max_channel + 1 - min_channel == len(channel_map):
            channel_map = slice(min_channel, max_channel + 1)
        return channel_map

        # full-fledged linear mixing model to allow for spectrophotometry later
        channel_map = np.zeros((data_frame.C, model_frame.C))
        for i, c in enumerate(list(data_frame.channels)):
            j = list(model_frame.channels).index(c)
            assert j != -1, f"Could not find channel {c} in model frame"
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
        if self.channel_map is None:
            return model
        if isinstance(self.channel_map, slice):
            return model[self.channel_map]
        return np.dot(self.channel_map, model)


class NullRenderer(Renderer):
    def __init__(self, data_frame, model_frame):
        super().__init__(data_frame, model_frame)

    def get_model(*parameters):
        def nothing(model):
            return model

        return nothing


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

# match the spatial shapes of model and data
@primitive
def match_shape(model, data_frame, slices):
    data_slices, model_slices = slices
    data_shape = data_frame.shape

    # check if data get sliced
    if any(
        [
            data_slices[d].stop - data_slices[d].start != data_shape[d]
            for d in range(-2, 0)
        ]
    ):
        matched = np.zeros(data_frame.shape, dtype=data_frame.dtype)
        matched[data_slices] = model[model_slices]
        return matched

    return model[model_slices]


def _grad_match_shape(upstream_grad, model, data_frame, slices):
    # just slices gradients like the model
    data_slices, model_slices = slices

    def result(upstream_grad):
        _result = np.zeros(model.shape, dtype=model.dtype)
        _result[model_slices] = upstream_grad[data_slices]
        return _result

    return result

defvjp(match_shape, _grad_match_shape)

class ConvolutionRenderer(Renderer):

    def __init__(self, data_frame, model_frame, *parameters, convolution_type="fft", padding=10, psf_shift=None):

        if psf_shift is not None:
            psf_shift = Parameter(
                psf_shift, name="psf_shift", step=1.e-2)
            parameters = (*parameters, psf_shift)

        super().__init__(data_frame, model_frame, *parameters)

        assert convolution_type in [
            "real",
            "fft",
        ], "`convolution` must be either 'real' or 'fft'"
        self._convolution_type = convolution_type

        # 2D spatial region covered by data
        pixel_in_model_frame = data_frame.convert_pixel_to(model_frame)
        # since there cannot be rotation or scaling, it's only translation
        ll = np.round(pixel_in_model_frame.min(axis=0)).astype("int")
        ur = np.round(pixel_in_model_frame.max(axis=0)).astype("int") + 1
        bounds = (ll[0], ur[0]), (ll[1], ur[1])
        # properly treats truncation in both boxes
        data_box = model_frame.bbox[0] @ Box.from_bounds(*bounds)
        self.slices = overlapped_slices(data_box, model_frame.bbox)

        # construct diff kernel
        psf_fft = fft.Fourier(data_frame.psf.get_model().astype(model_frame.dtype))
        model_psf_fft = fft.Fourier(
            model_frame.psf.get_model().astype(model_frame.dtype)
        )
        self.diff_kernel = fft.match_psf(psf_fft, model_psf_fft, padding=padding)

    @property
    def convolution_bounds(self):
        """Build the slices needed for convolution in real space
        """
        if not hasattr(self, "_convolution_bounds"):
            coords = interpolation.get_filter_coords(self.diff_kernel[0])
            self._convolution_bounds = interpolation.get_filter_bounds(
                coords.reshape(-1, 2)
            )
        return self._convolution_bounds

    def convolve(self, model, convolution_type=None, psf_shift=None):
        """Convolve the model in a single band
        """
        if convolution_type is None:
            convolution_type = self._convolution_type
        if psf_shift is not None:
            kernel = fft.shift(self.diff_kernel.image,
                           psf_shift,
                           fft_shape=None,
                           axes=(-2, -1),
                           return_Fourier=True)
        else:
            kernel = self.diff_kernel.image
        if convolution_type == "real":
            result = convolve(model, kernel, self.convolution_bounds)
        elif convolution_type == "fft":
            result = fft.convolve(
                fft.Fourier(model), kernel, axes=(1, 2)
            ).image
        else:
            raise ValueError(
                "`convolution` must be either 'real' or 'fft', got {}".format(
                    convolution_type
                )
            )

        return result

    def __call__(self, model, *parameters):
        self.transform = self.get_model(*parameters)
        return self.transform(model, *parameters)


    def get_model(self, *parameters):

        def transform(model, *parameters):
            # restrict to observed channels
            model_ = self.map_channels(model)
            #get the shift
            shift = self.get_parameter("psf_shift", *parameters)
            if len(shift)==0:
                shift = None
            # convolve observed channels
            model_ = self.convolve(model_, psf_shift=shift)
            # adjust spatial shapes
            model_ = match_shape(model_, self.data_frame, self.slices)
            return model_


        return transform


class ResolutionRenderer(Renderer):
    def __init__(self, data_frame, model_frame, padding=10):

        super().__init__(data_frame, model_frame)

        # check if data is rotated wrt to model_frame
        self.angle, self.h = interpolation.get_angles(data_frame.wcs, model_frame.wcs)
        self.isrot = (np.abs(self.angle[1]) ** 2) > np.finfo(float).eps

        # Get pixel coordinates alinged with x and y axes  of this observation
        # in model frame
        lr_shape = data_frame.shape[1:]
        pixels = np.stack((np.arange(lr_shape[0]), np.arange(lr_shape[1])), axis=1)
        coord_hr = data_frame.convert_pixel_to(model_frame, pixel=pixels)

        # TODO: should coords define a _slices_for_model/data?
        # lr_inside_hr = model_frame.bbox.contains(coord_hr)

        # compute diff kernel in model_frame pixels
        diff_psf, psf_lr_hr = self.build_diffkernel(data_frame, model_frame)

        # 1D convolutions convolutions of the model are done along the smaller axis, therefore,
        # psf is convolved along the frame's longer axis.
        # the smaller frame axis:
        self.small_axis = data_frame.Nx <= data_frame.Ny

        self._fft_shape = fft._get_fft_shape(
            psf_lr_hr, np.zeros(model_frame.shape), padding=3, axes=[-2, -1], max=False,
        )
        # Cutting diff_psf if needded and keeping the parity
        if (self._fft_shape[-2] < diff_psf.shape[-2]) or (
            self._fft_shape[-1] < diff_psf.shape[-1]
        ):
            diff_psf = fft._centered(
                diff_psf, np.array([diff_psf.shape[0] + 1, *self._fft_shape]) - 1
            )

        self.diff_kernel = fft.Fourier(
            fft._pad(diff_psf.image, self._fft_shape, axes=(-2, -1))
        )

        # Center of the FFT shape for matched diff kernel
        center_y = int(
            self._fft_shape[0] / 2.0 - (self._fft_shape[0] - model_frame.Ny) / 2.0
        ) + ((self._fft_shape[0] % 2) != 0) * ((model_frame.Ny % 2) == 0)
        center_x = int(
            self._fft_shape[1] / 2.0 - (self._fft_shape[1] - model_frame.Nx) / 2.0
        ) - ((self._fft_shape[1] % 2) != 0) * ((model_frame.Nx % 2) == 0)

        # Compute the shifts of all LR pixels into centered HR coord
        # 1 ) aligned case
        if not self.isrot:
            axes = [int(not self.small_axis) + 1]
            self.shifts = coord_hr.T
            self.shifts[0] -= center_y
            self.shifts[1] -= center_x
            self.other_shifts = np.copy(self.shifts)
        # 2) rotated case
        else:
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

        # Computes the resampling/convolution matrix
        resconv_op = self.sinc_shift(self.diff_kernel, self.shifts, axes)
        self._resconv_op = np.array(resconv_op, dtype=model_frame.dtype) * self.h ** 2

        if self.isrot:
            self._resconv_op = self._resconv_op.reshape(*self._resconv_op.shape[:2], -1)
        elif self.small_axis:
            self._resconv_op = self._resconv_op.reshape(*self._resconv_op.shape[:2], -1)
        else:
            self._resconv_op = self._resconv_op.reshape(
                self._resconv_op.shape[0], -1, self._resconv_op.shape[-1]
            )

    def build_diffkernel(self, data_frame, model_frame):
        """Builds the differential convolution kernel between the observation and the model psf

        Parameters
        ----------
        model_frame: Frame object
            the frame of the model
        Returns
        -------
        diff_psf: array
            the differential psf between observation and frame psf.
        """
        # Compute diff kernel at hr
        wcs_hr = model_frame.wcs
        wcs_lr = data_frame.wcs

        # PSF models
        psf_hr = model_frame.psf.get_model()
        psf_lr = data_frame.psf.get_model().astype(model_frame.dtype)

        # Computes spatially matching observation and model psf. The observation psf is also resampled \\
        # to the model frame resolution
        # Odd pad shape
        pad_shape = (
            np.array(
                (self.data_frame.shape[-2:] + np.array(psf_lr.shape[-2:])) / 2
            ).astype(int)
            * 2
            + 1
        )
        h_lr = interpolation.get_pixel_size(interpolation.get_affine(wcs_lr))
        h_hr = interpolation.get_pixel_size(interpolation.get_affine(wcs_hr))

        # Interpolation of the low res psf
        # TODO: isn't that just inverse of self.angle, self.h?
        angle, h_ratio = interpolation.get_angles(wcs_hr, wcs_lr)
        psf_lr_hr = interpolation.sinc_interp_inplace(
            psf_lr, h_lr, h_hr, angle, pad_shape=pad_shape
        )

        # Normalisation
        psf_hr /= np.sum(psf_hr)
        psf_lr_hr /= np.sum(psf_lr_hr)

        # build diff kernel in Fourier space
        diff_psf = fft.match_psf(fft.Fourier(psf_lr_hr), fft.Fourier(psf_hr))

        return diff_psf, psf_hr

    def sinc_shift(self, imgs, shifts, axes):
        """Performs 2 1D sinc convolutions and shifting along one rotated axis in Fourier space.

        Parameters
        ----------
        imgs: Fourier
            a Fourier object of 2D data to sinc convolve and shift
            to the adequate shape.
        shifts: array
            an array of the shift values for each line and columns of data in imgs
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
        # The n-dimensional transform could pose problem for very large data
        op = fft.Fourier.from_fft(imgs_shiftfft, fft_shape, inv_shape, fft_axes).image
        return op

    def get_model(self, *parameters):
        def transform(model):
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
            model_ = self.map_channels(model)
            C = model_.shape[0]
            dtype = model_.dtype

            # FFT of model, padding the psf to the fast_shape size
            model_ = fft.Fourier(fft._pad(model_, self._fft_shape, axes=(-2, -1)))

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
                            for c in range(C)
                        ],
                        dtype=dtype,
                    )
                else:
                    return np.array(
                        [
                            np.dot(self._resconv_op[c], model_conv[c].T).T
                            for c in range(C)
                        ],
                        dtype=dtype,
                    )

            if self.small_axis:
                model_conv = model_conv.reshape(
                    model_conv.shape[0], -1, model_conv.shape[-1]
                )
                return np.array(
                    [
                        np.dot(model_conv[c].T, self._resconv_op[c].T).T
                        for c in range(C)
                    ],
                    dtype=dtype,
                )
            else:
                model_conv = model_conv.reshape(*model_conv.shape[:2], -1)
                return np.array(
                    [
                        np.dot(self._resconv_op[c].T, model_conv[c].T).T
                        for c in range(C)
                    ],
                    dtype=dtype,
                )

        return transform

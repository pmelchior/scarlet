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

        if psfs is not None:
            psfs = np.array(psfs)
            # Make sure that psf is always 3D
            if len(psfs.shape) == 2:
                psfs = psfs[None]
            psfs = psfs / psfs.sum(axis=(1, 2))[:, None, None]
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


    def __init__(self, images, psfs=None, weights=None, wcs=None, filtercurve=None, structure=None,
                 padding=10):

        super().__init__(images.shape, wcs=wcs, psfs=psfs, filtercurve=filtercurve)

        self.images = np.array(images)
        if weights is not None:
            self.weights = np.array(weights)
        else:
            self.weights = 1

        self.structure = structure
        if structure is not None:
            self.structure = np.array(self.structure)
        self.padding = padding

    def match(self, scene):
        """Match the psf in each observed band to the target PSF
        """
        if self.psfs is not None:
            # First we setup the parameters for the model -> observation FFTs
            # Make the PSF stamp wider due to errors when matching PSFs
            psf_shape = np.array(self.psfs[0].shape) + self.padding
            shape = np.array(scene.shape[1:]) + psf_shape - 1

            # Choose the optimal shape for FFTPack DFT
            self.fftpack_shape = [fftpack.helper.next_fast_len(d) for d in shape]
            # Store the pre-fftpack optimization slices
            self.slices = tuple(np.concatenate(([slice(self.B)],[slice(s) for s in shape])))

            # Now we setup the parameters for the psf -> kernel FFTs
            shape = np.array(scene.psfs[0].shape) + np.array(self.psfs[0].shape) - 1
            fftpack_shape = [fftpack.helper.next_fast_len(d) for d in shape]
            # Deconvolve the target PSF
            target_fft = np.fft.rfftn(scene.psfs[0], fftpack_shape)

            # Match the PSF in each band
            _psf_fft = np.fft.rfftn(self.psfs, fftpack_shape, axes=(1, 2))


            kernels = np.fft.ifftshift(np.fft.irfftn(_psf_fft / target_fft, fftpack_shape, axes=(1, 2)), axes=(1, 2))


            #kernels *= scene.psfs[0].sum()
            if kernels.shape[1] % 2 == 0:
                kernels = kernels[:, 1:, 1:]

            kernels = _centered(kernels, psf_shape)
            new_kernel_fft = np.fft.rfftn(kernels, self.fftpack_shape, axes=(1, 2))

            self.psfs_fft = np.array(new_kernel_fft)
            self.kernels = np.array(kernels)

        return self

    def _convolve_bands(self, model, psf_fft):
        """Convolve the model in a single band
        """
        model_fft = np.fft.rfftn(model, self.fftpack_shape, axes=(1, 2))

        convolved = np.fft.irfftn(model_fft * psf_fft, self.fftpack_shape, axes=(1, 2))[self.slices]

        return _centered(convolved, model[0].shape)


    def get_model(self, model):
        """Resample and convolve a model to the observation frame
        Parameters
        ----------
        model: array
            The model in some other data frame.
        Returns
        -------
        model: array
            The convolved and resampled `model` in the observation frame.
        """
        if self.structure is not None:
            assert self.structure.size == model.shape[0]
            model = model[self.structure == 1]
        if self.psfs is not None:
            model = self._convolve_bands(model, self.psfs_fft)

        return model

    def get_loss(self, model):
        """Calculate the loss function for the model
        Parameters
        ----------
        model: array
            The model in some other data frame.
        Returns
        -------
        result: array
            Scalar tensor with the likelihood of the model
            given the image data.
        """

        model = self.get_model(model)

        return 0.5 * np.sum((self.weights * (model - self.images)) ** 2)


class LowResObservation(Scene):
    """Data and metadata for a set of observations to resample on a different grid

    Attributes
    ----------
    images: array or tensor
        3D data cube (bands, Ny, Nx) of the image in each band.
        These images must be resampled to the same pixel scale and
        same target WCS (for now).
        wcs: WCS object
        World Coordinate System associated with the images.
    psfs: array or tensor
        PSF for each band in `images`.
    weights: array or tensor
        Weight for each pixel in `images`.
        If a set of masks exists for the observations then
        then any masked pixels should have their `weight` set
        to zero.
    padding: int
        Number of pixels to pad each side with, in addition to
        half the width of the PSF, for FFT's. This is needed to
        prevent artifacts due to the FFT.
    structure: array
        An array that encodes the position of the images of this observation into a high resolution
        (spatial and spectral) scene.
    """

    def __init__(self, images, wcs=None, psfs=None, weights=None, filtercurve=None, padding=3, target_psf=None,
                 structure=None):
        super().__init__(images.shape, wcs=wcs, psfs=psfs, filtercurve=filtercurve)

        self.images = np.array(images)
        self.padding = padding

        if weights is not None:
            self.weights = np.array(weights)
        else:
            self.weights = 1

        self.target_psf = target_psf

        self.structure = structure

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

        y_hr, x_hr = np.where(np.zeros((Ny, Nx)) == 0)
        y_lr, x_lr = self._coord_lr

        Nlr = x_lr.size

        h = y_hr[1] - y_hr[0]
        if h == 0:
            h = x_hr[1] - x_hr[0]
        assert h != 0

        # sinc interpolant:
        ker_mat = interpolation.sinc2D((y_lr[:, np.newaxis] - y_hr[np.newaxis, :]) / h,
                                   (x_lr[:, np.newaxis] - x_hr[np.newaxis, :]) / h).reshape(Nlr, Ny, Nx)
        #import matplotlib.pyplot as plt
        print(self.fftpack_shape[0])
        operator = []
        for m in range(Nlr):
            ker = ker_mat[m]#interpolation.sinc2D((y_lr[m] - y_hr) / h,
                                       #(x_lr[m] - x_hr) / h).reshape(Ny, Nx)

            import matplotlib.pyplot as plt
            plt.imshow(np.log(ker), cmap = 'gist_sttern')
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

    def match(self, scene):
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

        if self.wcs == None:
            raise TypeError('WCS is actually mandatory, please provide one (tbdiscussed)')
        if self.psfs is not None:
            # Get pixel coordinates in each frame.
            mask, coord_lr, coord_hr = resampling.match_patches(scene.shape, self.shape, scene.wcs, self.wcs)
            self._coord_lr = coord_lr
            self._coord_hr = coord_hr
            self._mask = mask

            # Compute diff kernel at hr

            whr = scene.wcs
            wlr = self.wcs

            # Reference PSF
            if self.target_psf is None:

                _target = scene.psfs[0, :, :]
                _shape = scene.shape

            else:
                _target = self.target_psf

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
            shape = np.array(scene.shape[1:]) + psf_shape - 1
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

            self.resconv_op = np.array(resconv_op)

        else:
            class InitError(Exception):
                '''
                'Observation PSF needed: unless you are not dealing with astronomical data, you are doing something wrong'
                '''
                pass

            raise InitError

        return self

    @property
    def matching_mask(self):
        return self._mask

    def get_model(self, model):
        """Resample and convolve a model to the observation frame
        Parameters
        ----------
        model: array
            The model in some other data frame.
        Returns
        -------
        model: array
            The convolved and resampled `model` in the observation frame.
        """
        if self.structure is not None:
            assert self.structure.size == model.shape[0]
            model = model[self.structure == 1]
        obs = np.array([np.dot(model[b].flatten(), self.resconv_op[b]) for b in range(self.B)])

        return obs

    def model_to_frame(self, model):
        """Resample and convolve a model to the observation frame
        Parameters
        ----------
        model: array
            The model in some other data frame.
        Returns
        -------
        model: array
            The convolved and resampled `model` in the observation frame.
        """
        img = np.zeros(self.shape)
        img[:, self._coord_lr[0], self._coord_lr[1]] = self.get_model(model)
        return img

    def get_loss(self, model):
        '''Computes the loss of a given model compared to the object's images

        Parameters
        ----------
        model: array
            A model for the data as computed from get_model method
        Return
        loss: float
            Loss of the model
        ------
        '''

        if self._psfs is not None:
            model = self.get_model(model)

        return 0.5 * np.sum((self.weights * (
                model - self.images[:, self._coord_lr[0].astype(int), self._coord_lr[1].astype(int)])) ** 2)

import pytest
import numpy as np
from numpy.testing import assert_array_equal, assert_almost_equal

import scarlet
from scarlet.cache import Cache
from scarlet.config import Config


def get_kernel_info(shape=(5, 3), coords=None, center=None):
    # Use a normalized kernel with no symmetries
    kernel = np.arange(shape[0]*shape[1]).reshape(shape) + 1
    kernel = kernel / np.max(kernel)

    if center is None:
        center = shape[0]//2, shape[1]//2

    if coords is None:
        x = np.arange(shape[1])
        y = np.arange(shape[0])
        x, y = np.meshgrid(x, y)
        x -= center[1]
        y -= center[0]
        coords = np.dstack([y, x])
    else:
        coords = coords
    slices = scarlet.transformation.get_filter_slices(coords.reshape(-1, 2))
    return kernel, coords, slices


def test_get_filter_slices():
    # get_filter_slices is used to generate true values for other tests,
    # so we should first make sure that it isn't broken by hard coding the
    # results
    coords = np.array([
        [-2, -2, -2, -2, -2, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2],
        [-2, -1, 0, 1, 2, -2, -1, 0, 1, 2, -2, -1, 0, 1, 2, -2, -1, 0, 1, 2, -2, -1, 0, 1, 2]
    ]).T

    slices = scarlet.transformation.get_filter_slices(coords)
    true_slices = np.array([
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2],
        [2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 2, 0, 0, 0, 1, 2, 0, 0, 0, 1, 2, 0, 0, 0, 1, 2, 0, 0, 0, 1, 2],
        [2, 1, 0, 0, 0, 2, 1, 0, 0, 0, 2, 1, 0, 0, 0, 2, 1, 0, 0, 0, 2, 1, 0, 0, 0]
    ])
    np.testing.assert_almost_equal(slices, true_slices)


class TestConvolutions(object):
    def test_init(self):
        kernel, true_coords, true_slices = get_kernel_info()
        convolution = scarlet.transformation.Convolution(kernel)
        np.testing.assert_almost_equal(convolution._flat_values, kernel.flatten())
        np.testing.assert_array_equal(convolution._flat_coords, true_coords.reshape(-1, 2))
        np.testing.assert_almost_equal(convolution._slices, true_slices)

        kernel, true_coords, true_slices = get_kernel_info(coords=true_coords+10)
        convolution = scarlet.transformation.Convolution(kernel, coords=true_coords)
        np.testing.assert_almost_equal(convolution._flat_values, kernel.flatten())
        np.testing.assert_array_equal(convolution._flat_coords, true_coords.reshape(-1, 2))
        np.testing.assert_almost_equal(convolution._slices, true_slices)

        center = (1, 1)
        kernel, true_coords, true_slices = get_kernel_info(center=center)
        convolution = scarlet.transformation.Convolution(kernel, center=center)
        np.testing.assert_almost_equal(convolution._flat_values, kernel.flatten())
        np.testing.assert_array_equal(convolution._flat_coords, true_coords.reshape(-1, 2))
        np.testing.assert_almost_equal(convolution._slices, true_slices)

        with pytest.raises(ValueError):
            convolution = scarlet.transformation.Convolution(kernel.flatten())
        with pytest.raises(ValueError):
            convolution = scarlet.transformation.Convolution(kernel[:-1])
        with pytest.raises(ValueError):
            convolution = scarlet.transformation.Convolution(kernel[:, :-1])

    def test_T(self):
        Cache._cache = {}
        shape = (7, 7)
        kernel, true_coords, true_slices = get_kernel_info()
        _convolution = scarlet.transformation.Convolution(kernel)
        image = np.zeros(shape)
        image[(2, 2)] = 1
        convolution = _convolution.T.dot(image)
        true_convolution = np.zeros(shape)
        true_convolution[:kernel.shape[0], 1:kernel.shape[1]+1] = np.flipud(np.fliplr(kernel))
        np.testing.assert_almost_equal(convolution, true_convolution)
        np.testing.assert_array_equal(_convolution._flat_values, _convolution.T._flat_values)
        np.testing.assert_array_equal(_convolution._flat_coords, -_convolution.T._flat_coords)

    def test_dot(self):
        Cache._cache = {}
        shape = (7, 7)
        kernel, true_coords, true_slices = get_kernel_info()
        _convolution = scarlet.transformation.Convolution(kernel)
        # Test basic shift up and left
        image = np.zeros(shape)
        image[(0, 1)] = 1
        convolution = _convolution.dot(image)
        true_convolution = np.zeros(shape)
        true_convolution[:-kernel.shape[0]+1, :-kernel.shape[1]-1] = kernel[2:, :]
        np.testing.assert_almost_equal(convolution, true_convolution)
        # Test basic shift down and right
        image = np.zeros(shape)
        image[(-2, -1)] = 1
        convolution = _convolution.dot(image)
        true_convolution = np.zeros(shape)
        true_convolution[-kernel.shape[0]+1:, -kernel.shape[1]+1:] = kernel[:-1, :-1]
        np.testing.assert_almost_equal(convolution, true_convolution)

        # Test (slightly) more complex image
        image = np.zeros(shape)
        image[(1, 1)] = 1
        image[(1, 2)] = .5
        image[(2, 1)] = .2
        convolution = _convolution.dot(image)
        true_convolution = [
            [0.280000000, 0.493333333, 0.606666667, 0.200000000, 0.000000000, 0.000000000, 0.000000000],
            [0.520000000, 0.833333333, 0.946666667, 0.300000000, 0.000000000, 0.000000000, 0.000000000],
            [0.760000000, 1.173333333, 1.286666667, 0.400000000, 0.000000000, 0.000000000, 0.000000000],
            [1.000000000, 1.513333333, 1.626666667, 0.500000000, 0.000000000, 0.000000000, 0.000000000],
            [0.173333333, 0.186666667, 0.200000000, 0.000000000, 0.000000000, 0.000000000, 0.000000000],
            [0.000000000, 0.000000000, 0.000000000, 0.000000000, 0.000000000, 0.000000000, 0.000000000],
            [0.000000000, 0.000000000, 0.000000000, 0.000000000, 0.000000000, 0.000000000, 0.000000000]
        ]
        np.testing.assert_almost_equal(convolution, true_convolution)


def test_FFTKernel():
    Cache._cache = {}
    kernel, _, _ = get_kernel_info()
    _kernel = scarlet.transformation.FFTKernel(kernel, "test")
    assert _kernel.shape == kernel.shape
    assert _kernel.key == "test"
    np.testing.assert_array_equal(_kernel.kernel, kernel)
    np.testing.assert_array_equal(_kernel.T.kernel, kernel[::-1, ::-1])
    np.testing.assert_array_equal(_kernel.T.key, "test")
    np.testing.assert_array_equal(_kernel.T.window, None)
    np.testing.assert_array_equal(_kernel.T.transposed, True)

    _kernel = scarlet.transformation.FFTKernel(kernel, "test", transposed=False)
    window = (np.arange(kernel.shape[0]), np.arange(kernel.shape[1]))
    _kernel = scarlet.transformation.FFTKernel(kernel, "test", window)
    np.testing.assert_array_equal(np.hstack(_kernel.window), np.hstack(window))
    windowT = (-window[0][::-1], (-window[1][::-1]))
    np.testing.assert_array_equal(np.hstack(_kernel.T.window), np.hstack(windowT))

    _kernel = scarlet.transformation.FFTKernel(kernel, "test", transposed=True)
    np.testing.assert_array_equal(_kernel.transposed, True)
    np.testing.assert_array_equal(_kernel.T.transposed, False)

    shape = (11, 11)
    projected_kernel = scarlet.resample.project_image(kernel, shape)
    Kernel = np.fft.fft2(np.fft.ifftshift(projected_kernel))
    np.testing.assert_almost_equal(_kernel.Kernel(shape), Kernel)
    # Check caching
    cache_key = (shape, _kernel.transposed)
    _Kernel = Cache.check("test", cache_key)
    np.testing.assert_almost_equal(_Kernel, Kernel)

    cache_key = (shape, not _kernel.transposed)
    with pytest.raises(KeyError):
        _Kernel = Cache.check("test", cache_key)
    projected_kernel = scarlet.resample.project_image(kernel, shape)[::-1, ::-1]
    Kernel = np.fft.fft2(np.fft.ifftshift(projected_kernel))
    np.testing.assert_almost_equal(_kernel.T.Kernel(shape), Kernel)
    cache_key = (shape, not _kernel.transposed)
    _Kernel = Cache.check("test", cache_key)
    np.testing.assert_almost_equal(_Kernel, Kernel)

    # Check that using `center` gives the same result as window
    _kernel, ywin, xwin = scarlet.resample.get_separable_kernel(.25, .1)
    kernel0 = scarlet.transformation.FFTKernel(_kernel, "test", (ywin, xwin))
    kernel1 = scarlet.transformation.FFTKernel(_kernel, "test", center=(2, 2))
    assert_array_equal(kernel0.window, kernel1.window)


class TestFFTConvolution(object):
    def test_init(self):
        kernel0, _, _ = get_kernel_info(shape=(5, 9))
        kernel1, _, _ = get_kernel_info(shape=(7, 3))
        _kernel0 = scarlet.transformation.FFTKernel(kernel0, "test0")
        _kernel1 = scarlet.transformation.FFTKernel(kernel1, "test1")
        _convolution = scarlet.transformation.FFTConvolution(_kernel0)
        np.testing.assert_array_equal(_convolution.shape, (5, 9))
        _convolution = scarlet.transformation.FFTConvolution(_kernel0, _kernel1)
        np.testing.assert_array_equal(_convolution.shape, (7, 9))

        _convolution = scarlet.transformation.FFTConvolution.fromInterpolation(.25, .33)
        true_kernel = [
            [9.3384959E-04, -4.3701781E-03, 2.4431883E-02, 1.1293776E-02, -2.7718484E-03, 3.7321775E-04],
            [-4.1331457E-03, 1.9342069E-02, -1.0813361E-01, -4.9985375E-02, 1.2267986E-02, -1.6518328E-03],
            [2.7686826E-02, -1.2956729E-01, 7.2435786E-01, 3.3483852E-01, -8.2179922E-02, 1.1065181E-02],
            [8.4046463E-03, -3.9331603E-02, 2.1988694E-01, 1.0164398E-01, -2.4946636E-02, 3.3589597E-03],
            [-2.1087478E-03, 9.8684024E-03, -5.5170211E-02, -2.5502742E-02, 6.2591764E-03, -8.4277182E-04],
            [2.2881675E-04, -1.0708041E-03, 5.9864286E-03, 2.7672605E-03, -6.7917291E-04, 9.1447780E-05]
        ]
        true_window = [[-2, -1, 0, 1, 2, 3], [-2, -1, 0, 1, 2, 3]]
        np.testing.assert_almost_equal(_convolution.kernels[0].kernel, true_kernel)
        np.testing.assert_almost_equal(_convolution.kernels[0].window, true_window)

    def test_T(self):
        kernel0, _, _ = get_kernel_info(shape=(5, 9))
        kernel1, _, _ = get_kernel_info(shape=(7, 3))
        _kernel0 = scarlet.transformation.FFTKernel(kernel0, "test0")
        _kernel1 = scarlet.transformation.FFTKernel(kernel1, "test1")
        _convolution = scarlet.transformation.FFTConvolution(_kernel0, _kernel1)
        np.testing.assert_array_equal(_convolution.T.kernels[0].kernel, _kernel1.kernel[::-1, ::-1])
        np.testing.assert_array_equal(_convolution.T.kernels[1].kernel, _kernel0.kernel[::-1, ::-1])

    def test_dot(self):
        kernel0, _, _ = get_kernel_info(shape=(5, 9))
        _kernel0 = scarlet.transformation.FFTKernel(kernel0, "test0")
        _convolution0 = scarlet.transformation.FFTConvolution(_kernel0)
        _convolution1 = scarlet.transformation.FFTConvolution.fromInterpolation(.25, .33)
        _convolution01 = _convolution0.dot(_convolution1)

        np.testing.assert_array_equal(type(_convolution01), scarlet.transformation.FFTConvolution)
        np.testing.assert_almost_equal(_convolution01.kernels[0].kernel, _convolution0.kernels[0].kernel)
        np.testing.assert_almost_equal(_convolution01.kernels[1].kernel, _convolution1.kernels[0].kernel)
        np.testing.assert_almost_equal(_convolution01.shape, (6, 9))

        # Test basic shift up and left
        shape = (7, 7)
        kernel, _, _ = get_kernel_info()
        _kernel = scarlet.transformation.FFTKernel(kernel, "test")
        _convolution = scarlet.transformation.FFTConvolution(_kernel)
        image = np.zeros(shape)
        image[(0, 1)] = 1
        convolution = _convolution.dot(image)
        true_convolution = np.zeros(shape)
        true_convolution[:-kernel.shape[0]+1, :-kernel.shape[1]-1] = kernel[2:, :]
        np.testing.assert_almost_equal(convolution, true_convolution)


class TestLinearTranslation(object):
    def test_init(self):
        _convolution = scarlet.transformation.LinearTranslation(.33, .25)
        true_values = [0.5025, 0.1675, 0.2475, 0.0825]
        true_coords = np.array([[0, 0, 1, 1], [0, 1, 0, 1]]).T
        true_slices = [[0, 0, 1, 1],
                       [0, 0, 0, 0],
                       [0, 1, 0, 1],
                       [0, 0, 0, 0]]
        np.testing.assert_almost_equal(_convolution._flat_values, true_values)
        np.testing.assert_array_equal(_convolution._flat_coords, true_coords)
        np.testing.assert_almost_equal(_convolution._slices, true_slices)

        _convolution = scarlet.transformation.LinearTranslation(-.33, .25)
        true_values = [0.5025, 0.1675, 0.2475, 0.0825]
        true_coords = np.array([[0, 0, -1, -1],
                                [0, 1, 0, 1]]).T
        true_slices = [[0, 0, 0, 0],
                       [0, 0, 1, 1],
                       [0, 1, 0, 1],
                       [0, 0, 0, 0]]
        np.testing.assert_almost_equal(_convolution._flat_values, true_values)
        np.testing.assert_array_equal(_convolution._flat_coords, true_coords)
        np.testing.assert_almost_equal(_convolution._slices, true_slices)

        _convolution = scarlet.transformation.LinearTranslation(-.33, -.25)
        true_values = [0.5025, 0.1675, 0.2475, 0.0825]
        true_coords = np.array([[0, 0, -1, -1],
                                [0, -1, 0, -1]]).T
        true_slices = [[0, 0, 0, 0],
                       [0, 0, 1, 1],
                       [0, 0, 0, 0],
                       [0, 1, 0, 1]]
        np.testing.assert_almost_equal(_convolution._flat_values, true_values)
        np.testing.assert_array_equal(_convolution._flat_coords, true_coords)
        np.testing.assert_almost_equal(_convolution._slices, true_slices)

    def test_T(self):
        shape = (7, 7)
        image = np.zeros(shape)
        image[(2, 2)] = 1
        _convolution = scarlet.transformation.LinearTranslation(-.33, .25)
        kernel = _convolution.T._flat_values.reshape(2, 2)
        convolution = _convolution.T.dot(image)
        true_convolution = np.zeros(shape)
        true_convolution[2:4, 1:3] = kernel[:, ::-1]

        np.testing.assert_almost_equal(convolution, true_convolution)
        np.testing.assert_array_equal(_convolution._flat_values, _convolution.T._flat_values)
        np.testing.assert_array_equal(_convolution._flat_coords, -_convolution.T._flat_coords)

    def test_dot(self):
        shape = (7, 7)
        image = np.zeros(shape)
        image[(2, 2)] = 1
        _convolution = scarlet.transformation.LinearTranslation(-.33, .25)
        kernel = _convolution._flat_values.reshape(2, 2)
        convolution = _convolution.dot(image)
        true_convolution = np.zeros(shape)
        true_convolution[1:3, 2:4] = kernel[::-1, :]

        np.testing.assert_almost_equal(convolution, true_convolution)


class TestFilterChain(object):
    def test_all(self):
        Cache._cache = {}
        _kernel, _, _ = get_kernel_info(shape=(5, 9))
        _kernel += 1
        dyx = (.234, -.726)
        kernel = scarlet.transformation.FFTKernel(_kernel, "test0")
        translation = scarlet.transformation.FFTConvolution.fromInterpolation(
            *dyx, scarlet.resample.bilinear
        )
        kernel_convolution = scarlet.transformation.FFTConvolution(kernel)
        convolution = scarlet.transformation.LinearFilterChain([translation, kernel_convolution])
        true_translation = scarlet.transformation.LinearTranslation(*dyx)
        true_kernel_convolution = scarlet.transformation.Convolution(_kernel)
        true_convolution = scarlet.transformation.LinearFilterChain([
            true_translation, true_kernel_convolution
        ])

        np.testing.assert_almost_equal(convolution.filters[0].kernels[0].kernel,
                                       translation.kernels[0].kernel)
        np.testing.assert_almost_equal(convolution.filters[1].kernels[0].kernel,
                                       kernel_convolution.kernels[0].kernel)

        np.testing.assert_almost_equal(convolution.T.filters[0].kernels[0].kernel,
                                       kernel_convolution.T.kernels[0].kernel)
        np.testing.assert_almost_equal(convolution.T.filters[1].kernels[0].kernel,
                                       translation.T.kernels[0].kernel)

        shape = (11, 11)
        image = np.zeros(shape)
        image[5, 5] = 1
        separate = convolution.dot(image)
        combined = translation.dot(kernel_convolution).dot(image)
        truth = true_convolution.dot(image)
        assert_almost_equal(separate, truth)
        assert_almost_equal(combined, truth)


class TestGamma(object):
    def test_init(self):
        # Test initializing with true convolutions
        dx = .4
        dy = .1
        true_center = np.array([2, 1])
        center = np.array([0, 0])
        config = Config(use_fft=False)
        kernel, true_coords, _ = get_kernel_info()
        gamma = scarlet.transformation.Gamma([kernel], center, dy, dx, config)

        values = gamma.psfFilters[0]._flat_values
        coords = gamma.psfFilters[0]._flat_coords
        slices = np.array(scarlet.transformation.get_filter_slices(coords))
        true_values = kernel
        true_coords = true_coords.reshape(-1, 2)
        true_coords[:, 0] += (true_center-center)[0]
        true_coords[:, 1] += (true_center-center)[1]
        true_slices = scarlet.transformation.get_filter_slices(true_coords)
        assert_almost_equal(values, true_values.flatten())
        assert_array_equal(coords, true_coords)
        assert_almost_equal(slices, true_slices)

        # Test initialization with FFT convolutions
        gamma = scarlet.transformation.Gamma([kernel], center, dy, dx)
        _kernel = gamma.psfFilters[0].kernels[0].kernel
        _window = gamma.psfFilters[0].kernels[0].window
        assert_array_equal(kernel, _kernel)
        true_window = (np.arange(kernel.shape[0])-center[0], np.arange(kernel.shape[1]-center[1]))
        assert_array_equal(np.hstack(_window), np.hstack(true_window))

    def test_usage(self):
        # Multiband PSF
        Cache._cache = {}
        dy = .103
        dx = .562
        psfs = np.arange(70).reshape(2, 5, 7)
        image = np.zeros((21, 21))
        image[10, 10] = 1
        shifted = np.zeros((2, 21, 21))
        shifted[:, 8:13, 7:14] += (1-dy)*(1-dx)*psfs
        shifted[:, 9:14, 7:14] += dy*(1-dx)*psfs
        shifted[:, 8:13, 8:15] += (1-dy)*dx*psfs
        shifted[:, 9:14, 8:15] += dy*dx*psfs

        # Convolution
        config = Config(use_fft=False)
        gamma = scarlet.transformation.Gamma(psfs, config=config)
        convolved = np.array([gamma()[n].dot(image) for n in range(len(psfs))])
        truth = np.zeros((2, 21, 21))
        truth[:, 8:13, 7:14] = psfs
        assert_array_equal(convolved, truth)

        convolved = np.array([gamma((dy, dx))[n].dot(image) for n in range(len(psfs))])
        assert_almost_equal(convolved, shifted)

        # FFT Convolution
        config = Config(interpolation=scarlet.resample.bilinear)
        gamma = scarlet.transformation.Gamma(psfs, config=config)
        convolved = np.array([gamma()[n].dot(image) for n in range(len(psfs))])
        assert_almost_equal(convolved, truth)
        convolved = np.array([gamma((dy, dx))[n].dot(image) for n in range(len(psfs))])
        assert_almost_equal(convolved, shifted)


def test_operators():
    # PSF Op
    psf = np.arange(35).reshape(7, 5)
    image = np.zeros((21, 21))
    image[10, 10] = 1

    psf_op = scarlet.transformation.getPSFOp(psf, image.shape)
    psf_convolution = psf_op.dot(image)
    _kernel = scarlet.transformation.FFTKernel(psf, "test0")
    _convolution = scarlet.transformation.FFTConvolution(_kernel)
    truth = _convolution.dot(image)
    assert_almost_equal(psf_convolution, truth)

    # Zero Operator
    zero_op = scarlet.transformation.getZeroOp(image.shape)
    result = zero_op.dot(image)
    truth = np.zeros(image.shape)
    assert_array_equal(result, truth)

    # Identity Operator
    identity = scarlet.transformation.getIdentityOp(image.shape)
    result = identity.dot(image.reshape(-1)).reshape(image.shape)
    assert_array_equal(result, image)

    # Symmetry Operator
    symmetry = scarlet.transformation.getSymmetryOp(psf.shape)
    result = symmetry.dot(psf.reshape(-1)).reshape(psf.shape)
    truth = np.arange(35)
    truth -= truth[::-1]
    truth = truth.reshape(7, 5)
    assert_array_equal(result, truth)

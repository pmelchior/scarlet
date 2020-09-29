import pytest
import numpy as np
from numpy.testing import assert_array_equal, assert_almost_equal

import scarlet


class TestProjections(object):
    """Test project_image

    Because the behavior of projections is dependent on
    whether the input image and the output image have an
    even or odd number of pixels, we have tests for all
    four different cases (odd-odd, even-even, odd-even, even-odd).
    """

    def test_odd2odd(self):
        project_image = scarlet.interpolation.project_image
        img = np.arange(35).reshape(5, 7)

        # samller to bigger
        shape = (11, 9)
        result = project_image(img, shape)
        truth = np.zeros(shape)
        truth[3:-3, 1:-1] = img
        assert_array_equal(result, truth)

        # bigger to smaller
        shape = (3, 3)
        result = project_image(img, shape)
        truth = img[1:-1, 2:-2]
        assert_array_equal(result, truth)

        # lower left smaller to bigger
        shape = (11, 9)
        result = project_image(img, shape, (-6, -6))
        truth = np.zeros(shape)
        truth[:4, :5] = img[-4:, -5:]
        assert_array_equal(result, truth)

        # lower left bigger to smaller
        shape = (3, 3)
        result = project_image(img, shape, (-4, -6))
        truth = np.zeros(shape)
        truth[:2, :2] = img[-2:, -2:]
        assert_array_equal(result, truth)

        # upper right smaller to bigger
        shape = (11, 9)
        result = project_image(img, shape, (4, 0))
        truth = np.zeros(shape)
        truth[-2:, -5:] = img[:2, :5]
        assert_array_equal(result, truth)

        # upper right bigger to smaller
        shape = (3, 3)
        result = project_image(img, shape, (0, 1))
        truth = np.zeros(shape)
        truth[-2:, -1:] = img[:2, :1]
        assert_array_equal(result, truth)

    def test_even2even(self):
        project_image = scarlet.interpolation.project_image
        img = np.arange(48).reshape(8, 6)

        # samller to bigger
        shape = (12, 8)
        result = project_image(img, shape)
        truth = np.zeros(shape)
        truth[2:-2, 1:-1] = img
        assert_array_equal(result, truth)

        # bigger to smaller
        shape = (6, 4)
        result = project_image(img, shape)
        truth = img[1:-1, 1:-1]
        assert_array_equal(result, truth)

        # lower left smaller to bigger
        shape = (14, 18)
        result = project_image(img, shape, (-10, -11))
        truth = np.zeros(shape)
        truth[:5, :4] = img[-5:, -4:]
        assert_array_equal(result, truth)

        # lower left bigger to smaller
        shape = (4, 4)
        result = project_image(img, shape, (-1, -1))
        truth = np.zeros(shape)
        truth[-3:, -3:] = img[:3, :3]
        assert_array_equal(result, truth)

        # upper right smaller to bigger
        shape = (12, 10)
        result = project_image(img, shape, (3, 1))
        truth = np.zeros(shape)
        truth[-3:, -4:] = img[:3, :4]
        assert_array_equal(result, truth)

        # upper right bigger to smaller
        shape = (4, 4)
        result = project_image(img, shape, (0, -1))
        truth = np.zeros(shape)
        truth[-2:, -3:] = img[:2, :3]
        assert_array_equal(result, truth)

    def test_odd2even(self):
        project_image = scarlet.interpolation.project_image
        img = np.arange(35).reshape(5, 7)

        # samller to bigger
        shape = (10, 8)
        result = project_image(img, shape)
        truth = np.zeros(shape)
        truth[3:8, 1:] = img
        assert_array_equal(result, truth)

        # bigger to smaller
        shape = (4, 4)
        result = project_image(img, shape)
        truth = img[:4, 1:-2]
        assert_array_equal(result, truth)

        # lower left smaller to bigger
        shape = (14, 18)
        result = project_image(img, shape, (-9, -11))
        truth = np.zeros(shape)
        truth[:3, :5] = img[-3:, -5:]
        assert_array_equal(result, truth)

        # lower left bigger to smaller
        shape = (4, 4)
        result = project_image(img, shape, (-4, -5))
        truth = np.zeros(shape)
        truth[:3, :4] = img[-3:, -4:]
        assert_array_equal(result, truth)

        # upper right smaller to bigger
        shape = (12, 10)
        result = project_image(img, shape, (3, 1))
        truth = np.zeros(shape)
        truth[-3:, -4:] = img[:3, :4]

        # upper right bigger to smaller
        shape = (4, 4)
        result = project_image(img, shape, (1, 0))
        truth = np.zeros(shape)
        truth[-1:, -2:] = img[:1, :2]
        assert_array_equal(result, truth)

    def test_even2odd(self):
        project_image = scarlet.interpolation.project_image
        img = np.arange(48).reshape(8, 6)

        # samller to bigger
        shape = (11, 9)
        result = project_image(img, shape)
        truth = np.zeros(shape)
        truth[1:-2, 1:-2] = img
        assert_array_equal(result, truth)

        # bigger to smaller
        shape = (3, 3)
        result = project_image(img, shape)
        truth = img[3:-2, 2:-1]
        assert_array_equal(result, truth)

        # lower left smaller to bigger
        shape = (11, 9)
        result = project_image(img, shape, (-9, -5))
        truth = np.zeros(shape)
        truth[:4, :5] = img[-4:, -5:]
        assert_array_equal(result, truth)

        # lower left bigger to smaller
        shape = (3, 3)
        result = project_image(img, shape, (-7, -5))
        truth = np.zeros(shape)
        truth[:2, :2] = img[-2:, -2:]
        assert_array_equal(result, truth)

        # upper right smaller to bigger
        shape = (11, 9)
        result = project_image(img, shape, (4, 0))
        truth = np.zeros(shape)
        truth[-2:, -5:] = img[:2, :5]
        assert_array_equal(result, truth)

        # upper right bigger to smaller
        shape = (3, 3)
        result = project_image(img, shape, (0, 1))
        truth = np.zeros(shape)
        truth[-2:, -1:] = img[:2, :1]
        assert_array_equal(result, truth)

    def test_zoom(self):
        # Test that zomming out and in keeps a consistent center
        kernel = np.arange(4).reshape(2, 2) + 1
        p3 = scarlet.interpolation.project_image(kernel, (3, 3))
        p6 = scarlet.interpolation.project_image(p3, (6, 6))
        p5 = scarlet.interpolation.project_image(p6, (5, 5))
        p2 = scarlet.interpolation.project_image(p3, (2, 2))
        assert_array_equal(p2, kernel)
        truth = [[1.0, 2.0, 0.0], [3.0, 4.0, 0.0], [0.0, 0.0, 0.0]]
        assert_array_equal(p3, truth)
        truth = [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 2.0, 0.0, 0.0],
            [0.0, 0.0, 3.0, 4.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ]
        assert_array_equal(p6, truth)
        truth = [
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 2.0, 0.0, 0.0],
            [0.0, 3.0, 4.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
        ]
        assert_array_equal(p5, truth)


def interpolate_comparison(func, zero_truth, positive_truth, **kwargs):
    # zero shift
    result = func(0, **kwargs)
    truth = zero_truth
    assert_almost_equal(result[0], truth[0])
    assert_array_equal(result[1], truth[1])

    # positive shift
    result = func(0.103, **kwargs)
    truth = positive_truth
    assert_almost_equal(result[0], truth[0])
    assert_array_equal(result[1], truth[1])

    # negative shift
    result = func(-0.103, **kwargs)
    truth = (truth[0][::-1], -truth[1][::-1])
    assert_almost_equal(result[0], truth[0])
    assert_array_equal(result[1], truth[1])

    with pytest.raises(ValueError):
        scarlet.interpolation.lanczos(1.1)
    with pytest.raises(ValueError):
        scarlet.interpolation.lanczos(-1.1)


class TestConvolutions:
    """Test FFT convolutions and interpolation algorithms
    """

    def test_bilinear(self):
        zero_truth = (np.array([1, 0]), np.array([0, 1]))
        positive_truth = (np.array([1 - 0.103, 0.103]), np.array([0, 1]))
        interpolate_comparison(
            scarlet.interpolation.bilinear, zero_truth, positive_truth
        )

    def test_cubic_spline(self):
        zero_truth = (np.array([0.0, 1.0, 0.0, 0.0]), np.array([-1, 0, 1, 2]))
        positive_truth = (
            np.array([-0.08287473, 0.97987473, 0.11251627, -0.00951627]),
            np.array([-1, 0, 1, 2]),
        )
        interpolate_comparison(
            scarlet.interpolation.cubic_spline, zero_truth, positive_truth
        )

    def test_catmull_rom(self):
        # Catmull Rom should be the same as the cubic spline
        # with a=0.5 and b=0
        zero_truth = scarlet.interpolation.cubic_spline(0, a=0.5)
        positive_truth = scarlet.interpolation.cubic_spline(0.103, a=0.5)
        interpolate_comparison(
            scarlet.interpolation.catmull_rom, zero_truth, positive_truth
        )

    def test_mitchel_netravali(self):
        # Mitchel Netravali should be the same as the cubic spline
        # with a=1/3 and b=1/3
        zero_truth = scarlet.interpolation.cubic_spline(0, a=1 / 3, b=1 / 3)
        positive_truth = scarlet.interpolation.cubic_spline(0.103, a=1 / 3, b=1 / 3)
        interpolate_comparison(
            scarlet.interpolation.mitchel_netravali, zero_truth, positive_truth
        )

    def test_lanczos(self):
        # test Lanczos 3
        zero_truth = (np.array([0, 0, 1, 0, 0, 0]), np.arange(6) - 2)
        positive_truth = (
            np.array(
                [
                    0.01763955,
                    -0.07267534,
                    0.98073579,
                    0.09695747,
                    -0.0245699,
                    0.00123974,
                ]
            ),
            np.array([-2, -1, 0, 1, 2, 3]),
        )
        interpolate_comparison(
            scarlet.interpolation.lanczos, zero_truth, positive_truth
        )

        # test Lanczos 5
        _truth = np.zeros((10,))
        _truth[4] = 1
        zero_truth = (_truth, np.arange(10) - 4)
        positive_truth = (
            np.array(
                [
                    5.11187895e-03,
                    -1.55432491e-02,
                    3.52955166e-02,
                    -8.45895745e-02,
                    9.81954247e-01,
                    1.06954413e-01,
                    -4.15882547e-02,
                    1.85994926e-02,
                    -6.77652513e-03,
                    4.34415682e-04,
                ]
            ),
            np.array([-4, -3, -2, -1, 0, 1, 2, 3, 4, 5]),
        )
        interpolate_comparison(
            scarlet.interpolation.lanczos, zero_truth, positive_truth, a=5
        )

    def test_separable(self):
        result = scarlet.interpolation.get_separable_kernel(0.103, 0.42)
        truth = [
            [
                0.000506097,
                -0.002566513,
                0.012535221,
                0.008810656,
                -0.002073032,
                0.000332194,
            ],
            [
                -0.002085129,
                0.010574090,
                -0.051645379,
                -0.036300092,
                0.008540937,
                -0.001368644,
            ],
            [
                0.028138304,
                -0.142694735,
                0.696941621,
                0.489860766,
                -0.115257837,
                0.018469518,
            ],
            [
                0.002781808,
                -0.014107082,
                0.068901018,
                0.048428598,
                -0.011394616,
                0.001825933,
            ],
            [
                -0.000704935,
                0.003574863,
                -0.017460144,
                -0.012272247,
                0.002887499,
                -0.000462708,
            ],
            [
                0.000035569,
                -0.000180379,
                0.000880996,
                0.000619227,
                -0.000145696,
                0.000023347,
            ],
        ]
        assert_almost_equal(result[0], truth)
        assert_array_equal(result[1], [-2, -1, 0, 1, 2, 3])
        assert_array_equal(result[2], [-2, -1, 0, 1, 2, 3])

        result = scarlet.interpolation.get_separable_kernel(
            0.103, -0.42, kernel=scarlet.interpolation.bilinear
        )
        truth = [[0.376740000, 0.520260000], [0.043260000, 0.059740000]]
        assert_almost_equal(result[0], truth)
        assert_array_equal(result[1], [0, 1])
        assert_array_equal(result[2], [-1, 0])

        result = scarlet.interpolation.get_separable_kernel(0.103, 0.42, a=5)
        truth = [
            [
                0.0000458,
                -0.0001796,
                0.0004278,
                -0.0009684,
                0.0037091,
                0.0026576,
                -0.0008415,
                0.0003764,
                -0.0001524,
                0.0000312,
            ],
            [
                -0.0001391,
                0.0005461,
                -0.0013007,
                0.0029445,
                -0.0112779,
                -0.0080806,
                0.0025588,
                -0.0011444,
                0.0004633,
                -0.0000948,
            ],
            [
                0.0003160,
                -0.0012401,
                0.0029536,
                -0.0066863,
                0.0256097,
                0.0183494,
                -0.0058105,
                0.0025986,
                -0.0010520,
                0.0002154,
            ],
            [
                -0.0007572,
                0.0029722,
                -0.0070786,
                0.0160245,
                -0.0613765,
                -0.0439765,
                0.0139254,
                -0.0062278,
                0.0025211,
                -0.0005161,
            ],
            [
                0.0087903,
                -0.0345021,
                0.0821710,
                -0.1860199,
                0.7124863,
                0.5104987,
                -0.1616529,
                0.0722953,
                -0.0292664,
                0.0059916,
            ],
            [
                0.0009574,
                -0.0037580,
                0.0089501,
                -0.0202613,
                0.0776040,
                0.0556035,
                -0.0176072,
                0.0078744,
                -0.0031877,
                0.0006526,
            ],
            [
                -0.0003723,
                0.0014613,
                -0.0034802,
                0.0078784,
                -0.0301756,
                -0.0216209,
                0.0068464,
                -0.0030619,
                0.0012395,
                -0.0002538,
            ],
            [
                0.0001665,
                -0.0006535,
                0.0015564,
                -0.0035235,
                0.0134954,
                0.0096695,
                -0.0030619,
                0.0013694,
                -0.0005543,
                0.0001135,
            ],
            [
                -0.0000607,
                0.0002381,
                -0.0005671,
                0.0012837,
                -0.0049169,
                -0.0035230,
                0.0011156,
                -0.0004989,
                0.0002020,
                -0.0000413,
            ],
            [
                0.0000039,
                -0.0000153,
                0.0000364,
                -0.0000823,
                0.0003152,
                0.0002258,
                -0.0000715,
                0.0000320,
                -0.0000129,
                0.0000027,
            ],
        ]
        assert_almost_equal(result[0], truth)
        assert_array_equal(result[1], np.arange(10) - 4)
        assert_array_equal(result[2], np.arange(10) - 4)

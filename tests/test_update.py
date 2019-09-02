import pytest
import numpy as np

import scarlet
import scarlet.update as update
import scarlet.measurement as measurement
from scarlet.cache import Cache


class TestUpdate(object):
    def test_pixel_center(self):
        morph = np.zeros((15, 15))
        morph[4, 7] = 1
        morph[11, 9] = 2
        # Use the default window, which misses the higher point
        center = measurement.max_pixel(morph, (5, 5))
        np.testing.assert_array_equal([4, 7], center)

        # Make window wider and test again
        center = measurement.max_pixel(morph, (5, 5), window=tuple([slice(0, 15)] * 2))
        np.testing.assert_array_equal([11, 9], center)

    def test_non_negativity(self):
        shape = (6, 3, 3)
        frame = scarlet.Frame(shape)
        sed = np.array([-.1, .1, 4, -.2, .2, 0], dtype=frame.dtype)
        morph = np.array([[-1, -.5, -1], [.1, 2, .3], [-.5, .3, 0]], dtype=frame.dtype)

        # Test SED only
        src = scarlet.Component(frame, sed.copy(), morph.copy())
        update.positive_sed(src)
        np.testing.assert_array_almost_equal(src.sed, [0, .1, 4, 0, .2, 0])
        np.testing.assert_array_equal(src.morph, morph)
        # Test morph only
        src = scarlet.Component(frame, sed.copy(), morph.copy())
        update.positive_morph(src)
        np.testing.assert_array_equal(src.sed, sed)
        np.testing.assert_array_almost_equal(src.morph, [[0, 0, 0], [.1, 2, .3], [0, .3, 0]])

        # Test SED and morph
        src = scarlet.Component(frame, sed.copy(), morph.copy())
        update.positive(src)
        np.testing.assert_array_almost_equal(src.sed, [0, .1, 4, 0, .2, 0])
        np.testing.assert_array_almost_equal(src.morph, [[0, 0, 0], [.1, 2, .3], [0, .3, 0]])

    def test_normalized(self):
        shape = (6, 5, 5)
        frame = scarlet.Frame(shape)
        sed = np.arange(shape[0], dtype=frame.dtype)
        morph = np.arange(shape[1]*shape[2], dtype=frame.dtype).reshape(shape[1], shape[2])

        # Test SED normalization
        src = scarlet.Component(frame, sed.copy(), morph.copy())
        update.normalized(src, type='sed')
        np.testing.assert_array_equal(src.sed, sed/15)
        np.testing.assert_array_equal(src.morph, morph*15)

        # Test morph unity normalization
        src = scarlet.Component(frame, sed.copy(), morph.copy())
        update.normalized(src, type='morph')
        norm = np.sum(morph)
        np.testing.assert_array_equal(src.sed, sed*norm)
        np.testing.assert_array_equal(src.morph, morph/norm)

        # Test morph max normalization
        src = scarlet.Component(frame, sed.copy(), morph.copy())
        update.normalized(src)
        np.testing.assert_array_equal(src.sed, sed*24)
        np.testing.assert_array_equal(src.morph, morph/24)

        with pytest.raises(ValueError):
            update.normalized(src, type='fubar')

    def test_sparsity(self):
        shape = (6, 5, 5)
        frame = scarlet.Frame(shape)
        sed = np.arange(shape[0])
        morph = np.arange(shape[1]*shape[2], dtype=float).reshape(shape[1], shape[2])

        # Test l0 sparsity
        src = scarlet.Component(frame, sed.copy(), morph.copy())
        src.L_morph = 1
        update.sparse_l0(src, thresh=4)
        true_morph = morph.copy()
        true_morph[0, :-1] = 0
        np.testing.assert_array_equal(src.sed, sed)
        np.testing.assert_array_equal(src.morph, true_morph)

        # Test l1 sparsity
        src = scarlet.Component(frame, sed.copy(), morph.copy())
        src.L_morph = 0.5
        update.sparse_l1(src, thresh=2)
        true_morph = np.zeros((morph.size))
        true_morph[5:] = np.arange(20) + 1
        true_morph = true_morph.reshape(5, 5)
        np.testing.assert_array_equal(src.sed, sed)
        np.testing.assert_array_equal(src.morph, true_morph)

    def test_thresh(self):
        # Use a random seed in the test to prevent race conditions
        np.random.seed(0)
        noise = np.random.rand(21, 21)*2  # noise background to eliminate
        signal = np.zeros(noise.shape)
        func = scarlet.psf.gaussian
        signal[7:14, 7:14] = scarlet.psf.generate_psf_image(func, (21, 21), normalize=False,
                                                            amplitude=10, sigma=3)[7:14, 7:14]
        morph = signal + noise
        sed = np.arange(5)
        shape = (len(sed), morph.shape[0], morph.shape[1])
        frame = scarlet.Frame(shape)
        src = scarlet.Component(frame, sed.copy(), morph.copy())
        thresh, _ = measurement.threshold(src.morph)
        true_morph = np.zeros(morph.shape)
        true_morph[7:14, 7:14] = morph[7:14, 7:14]
        update.threshold(src)
        np.testing.assert_array_equal(src.sed, sed)
        np.testing.assert_array_almost_equal(src.morph, true_morph)
        assert src.bboxes["thresh"] == scarlet.bbox.Box((7, 7), 7, 7)

    def test_monotonic(self):
        shape = (6, 5, 5)
        frame = scarlet.Frame(shape, dtype=np.float64)
        sed = np.arange(shape[0])
        morph = np.arange(shape[1]*shape[2], dtype=float).reshape(shape[1], shape[2])

        src = scarlet.Component(frame, sed.copy(), morph.copy())
        src.L_morph = 1
        update.monotonic(src, (2, 2), use_nearest=True, exact=False, thresh=0)
        new_X = [[0.0, 1.0, 2.0, 3.0, 4.0],
                 [5.0, 6.0, 7.0, 8.0, 9.0],
                 [10.0, 11.0, 12.0, 12.0, 12.0],
                 [11.0, 12.0, 12.0, 12.0, 12.0],
                 [12.0, 12.0, 12.0, 12.0, 12.0]]
        np.testing.assert_array_equal(src.sed, sed)
        np.testing.assert_array_equal(src.morph, new_X)

        # Weighted
        # We need to clear the cache, since this has already been created
        Cache._cache = {}
        src = scarlet.Component(frame, sed.copy(), morph.copy())
        src.L_morph = 1
        update.monotonic(src, (2, 2))
        new_X = [[0.000000000, 1.000000000, 2.000000000, 3.000000000, 4.000000000],
                 [5.000000000, 6.000000000, 7.000000000, 8.000000000, 9.000000000],
                 [9.742640687, 11.000000000, 12.000000000, 12.000000000, 10.828427125],
                 [11.030627697, 11.707106781, 12.000000000, 12.000000000, 11.771236166],
                 [11.556349186, 11.868867239, 11.914213562, 11.983249156, 11.928090416]]
        np.testing.assert_array_equal(src.sed, sed)
        np.testing.assert_almost_equal(src.morph, new_X)

        # Test that use_nearest=True and thresh !=0 are incompatible
        Cache._cache = {}
        src = scarlet.Component(frame, sed.copy(), morph.copy())
        src.L_morph = 1
        with pytest.raises(ValueError):
            update.monotonic(src, (2, 2), use_nearest=True, thresh=.25)

        # Use a threshold to force a gradient of 75% or steeper
        Cache._cache = {}
        src = scarlet.Component(frame, sed.copy(), morph.copy())
        src.L_morph = 1
        update.monotonic(src, (2, 2), thresh=.25)
        new_X = [[0.000000000, 1.000000000, 2.000000000, 3.000000000, 4.000000000],
                 [5.000000000, 6.000000000, 7.000000000, 7.242640687, 5.806841831],
                 [5.801461031, 9.000000000, 12.000000000, 9.000000000, 6.074431804],
                 [5.895545844, 7.681980515, 9.000000000, 7.681980515, 5.935521488],
                 [4.988519641, 5.949655012, 6.170941546, 5.949655012, 4.997301087]]
        np.testing.assert_array_equal(src.sed, sed)
        np.testing.assert_almost_equal(src.morph, new_X)

        # Test that exact=True is not implemented
        Cache._cache = {}
        src = scarlet.Component(frame, sed.copy(), morph.copy())
        src.L_morph = 1
        with pytest.raises(NotImplementedError):
            update.monotonic(src, (2, 2), exact=True)

    def test_symmetry(self):
        shape = (6, 5, 5)
        frame = scarlet.Frame(shape)
        sed = np.arange(shape[0])
        morph = np.arange(shape[1]*shape[2], dtype=float).reshape(shape[1], shape[2])

        # Centered symmetry
        src = scarlet.Component(frame, sed.copy(), morph.copy())
        src.L_morph = 1
        src.pixel_center = (2, 2)
        update.symmetric(src, src.pixel_center)
        result = np.ones_like(morph) * 12
        np.testing.assert_array_equal(src.morph, result)

        # Centered symmetry at half strength
        src = scarlet.Component(frame, sed.copy(), morph.copy())
        src.L_morph = 1
        src.pixel_center = (2, 2)
        update.symmetric(src, src.pixel_center, strength=.5, algorithm="soft")
        result = [[6.0, 6.5, 7.0, 7.5, 8.0],
                  [8.5, 9.0, 9.5, 10.0, 10.5],
                  [11.0, 11.5, 12.0, 12.5, 13.0],
                  [13.5, 14.0, 14.5, 15.0, 15.5],
                  [16.0, 16.5, 17.0, 17.5, 18.0]]
        np.testing.assert_array_equal(src.morph, result)

        # Uncentered symmetry
        src = scarlet.Component(frame, sed.copy(), morph.copy())
        src.L_morph = 1
        src.pixel_center = (1, 1)
        update.symmetric(src, src.pixel_center)
        result = morph.copy()
        result[:3, :3] = 6
        np.testing.assert_array_equal(src.morph, result)

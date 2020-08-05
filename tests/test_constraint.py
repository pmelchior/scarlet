import pytest
import numpy as np
from numpy.testing import assert_array_equal, assert_almost_equal
import scarlet


class TestUpdate(object):
    def test_positivity(self):
        X = np.random.rand(100) - 0.5
        step = 0

        constraint = scarlet.PositivityConstraint()
        X_ = constraint(X, step)
        assert all(X_ >= 0)

        threshold = 0.1
        constraint = scarlet.PositivityConstraint(zero=threshold)
        X_ = constraint(X, step)
        assert all(X_ >= threshold)

    def test_normalization(self):
        X = np.random.rand(100)
        step = 0

        X_ = X.copy()
        constraint = scarlet.NormalizationConstraint(type="sum")
        X_ = constraint(X_, step)
        assert_almost_equal(X_, X / X.sum())

        X_ = X.copy()
        constraint = scarlet.NormalizationConstraint(type="max")
        X_ = constraint(X_, step)
        assert_almost_equal(X_, X / X.max())

    def test_l0(self):
        X = np.random.rand(100) - 0.5
        step = 0.5
        thresh = 0.25

        X_ = X.copy()
        constraint = scarlet.L0Constraint(thresh=thresh, type="relative")
        X_ = constraint(X_, step)
        mask = np.abs(X) < thresh * step
        assert all(np.abs(X_[mask]) == 0)
        assert_array_equal(X_[~mask], X[~mask])

        X_ = X.copy()
        constraint = scarlet.L0Constraint(thresh=thresh, type="absolute")
        X_ = constraint(X_, step)
        mask = np.abs(X) < thresh
        assert all(np.abs(X_[mask]) == 0)
        assert_array_equal(X_[~mask], X[~mask])

    def test_l1(self):
        X = np.random.rand(100) - 0.5
        step = 0.5
        thresh = 0.25

        X_ = X.copy()
        constraint = scarlet.L1Constraint(thresh=thresh, type="relative")
        X_ = constraint(X_, step)
        thresh_ = thresh * step
        mask = np.abs(X) < thresh_
        assert all(np.abs(X_[mask]) == 0)
        assert_array_equal(np.abs(X_[~mask]), np.abs(np.abs(X[~mask]) - thresh_))

        X_ = X.copy()
        constraint = scarlet.L1Constraint(thresh=thresh, type="absolute")
        X_ = constraint(X_, step)
        mask = np.abs(X) < thresh
        assert all(np.abs(X_[mask]) == 0)
        assert_array_equal(np.abs(X_[~mask]), np.abs(np.abs(X[~mask]) - thresh))

    def test_threshold(self):
        # Use a random seed in the test to prevent race conditions
        np.random.seed(0)
        noise = np.random.rand(21, 21) * 2  # noise background to eliminate
        signal = np.zeros(noise.shape)
        func = scarlet.psf.gaussian
        signal[7:14, 7:14] = (
            10 * scarlet.PSF(func, shape=(None, 21, 21)).image[0, 7:14, 7:14]
        )
        X = signal + noise

        step = 0
        constraint = scarlet.ThresholdConstraint()
        X_ = constraint(X, step)

        # regression test with thresh from reference version
        thresh = 0.05704869232578929
        mask = X < thresh
        assert all(X_[mask] == 0)
        assert_array_equal(X_[~mask], X[~mask])

    def test_monotonic(self):
        shape = (5, 5)
        X = np.arange(shape[0] * shape[1], dtype=float).reshape(*shape)

        step = 0
        X_ = X.copy()
        constraint = scarlet.MonotonicityConstraint(
            neighbor_weight="nearest", min_gradient=0
        )
        X_ = constraint(X_, step)
        new_X = [
            [0.0, 1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0, 9.0],
            [10.0, 11.0, 12.0, 12.0, 12.0],
            [11.0, 12.0, 12.0, 12.0, 12.0],
            [12.0, 12.0, 12.0, 12.0, 12.0],
        ]
        assert_array_equal(X_, new_X)

        X_ = X.copy()
        constraint = scarlet.MonotonicityConstraint(
            neighbor_weight="angle", min_gradient=0
        )
        X_ = constraint(X_, step)
        new_X = [
            [0.000000000, 1.000000000, 2.000000000, 3.000000000, 4.000000000],
            [5.000000000, 6.000000000, 7.000000000, 8.000000000, 9.000000000],
            [9.742640687, 11.000000000, 12.000000000, 12.000000000, 10.828427125],
            [11.030627697, 11.707106781, 12.000000000, 12.000000000, 11.771236166],
            [11.556349186, 11.868867239, 11.914213562, 11.983249156, 11.928090416],
        ]
        assert_almost_equal(X_, new_X)

        X_ = X.copy()
        constraint = scarlet.MonotonicityConstraint(
            neighbor_weight="angle", min_gradient=0.25
        )
        X_ = constraint(X_, step)
        new_X = [
            [0.000000000, 1.000000000, 2.000000000, 3.000000000, 4.000000000],
            [5.000000000, 6.000000000, 7.000000000, 7.242640687, 5.806841831],
            [5.801461031, 9.000000000, 12.000000000, 9.000000000, 6.074431804],
            [5.895545844, 7.681980515, 9.000000000, 7.681980515, 5.935521488],
            [4.988519641, 5.949655012, 6.170941546, 5.949655012, 4.997301087],
        ]
        assert_almost_equal(X_, new_X)

    def test_symmetry(self):
        shape = (5, 5)
        X = np.arange(shape[0] * shape[1], dtype=float).reshape(*shape)

        # symmetry
        step = 0
        X_ = X.copy()
        constraint = scarlet.SymmetryConstraint()
        X_ = constraint(X_, step)
        new_X = np.ones_like(X) * 12
        assert_almost_equal(X_, new_X)

        # symmetry at half strength
        X_ = X.copy()
        constraint = scarlet.SymmetryConstraint(strength=0.5)
        X_ = constraint(X_, step)
        new_X = [
            [6.0, 6.5, 7.0, 7.5, 8.0],
            [8.5, 9.0, 9.5, 10.0, 10.5],
            [11.0, 11.5, 12.0, 12.5, 13.0],
            [13.5, 14.0, 14.5, 15.0, 15.5],
            [16.0, 16.5, 17.0, 17.5, 18.0],
        ]
        assert_almost_equal(X_, new_X)

    def test_center_on(self):
        shape = (5, 5)
        X = np.zeros(shape)

        constraint = scarlet.CenterOnConstraint()
        step = 0
        X = constraint(X, step)
        assert X[2, 2] > 0

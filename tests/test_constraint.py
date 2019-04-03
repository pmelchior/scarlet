import sys

import pytest
import numpy as np
import proxmin

import scarlet
from scarlet.constraint import Normalization
from scarlet.cache import Cache


try:
    import torch
except ImportError:
    pass


class TestConstraint(object):
    def test_minimal(self):
        sed = np.array([-1, 6, 8, 6, -2])
        morph = np.arange(25, dtype=float).reshape(5, 5) - 3

        # A normalization
        constraint = scarlet.constraint.MinimalConstraint()
        _sed = sed.copy()
        _morph = morph.copy()
        new_sed = constraint.prox_sed((5, 5))(_sed, 0)
        new_morph = constraint.prox_morph((5, 5,))(_morph, 0)
        true_morph = [[0.0, 0.0, 0.0, 0.0, 1.0],
                      [2.0, 3.0, 4.0, 5.0, 6.0],
                      [7.0, 8.0, 9.0, 10.0, 11.0],
                      [12.0, 13.0, 14.0, 15.0, 16.0],
                      [17.0, 18.0, 19.0, 20.0, 21.0]]
        np.testing.assert_array_equal(new_sed, [0, .3, .4, .3, 0])
        np.testing.assert_array_equal(new_morph, true_morph)

        # S normalization
        constraint = scarlet.constraint.MinimalConstraint(normalization=Normalization.S)
        _sed = sed.copy()
        _morph = morph.copy()
        new_sed = constraint.prox_sed((5, 5))(_sed, 0)
        new_morph = constraint.prox_morph((5, 5,))(_morph, 0)
        true_morph = [[0.000000, 0.000000, 0.000000, 0.000000, 0.004329],
                      [0.008658, 0.012987, 0.017316, 0.021645, 0.025974],
                      [0.030303, 0.034632, 0.038961, 0.043290, 0.047619],
                      [0.051948, 0.056277, 0.060606, 0.064935, 0.069264],
                      [0.073593, 0.077922, 0.082251, 0.086580, 0.090909]]
        np.testing.assert_array_equal(new_sed, [0, 6, 8, 6, 0])
        np.testing.assert_almost_equal(new_morph, true_morph)

        # Smax Normalization
        constraint = scarlet.constraint.MinimalConstraint(normalization=Normalization.Smax)
        _sed = sed.copy()
        _morph = morph.copy()
        new_sed = constraint.prox_sed((5, 5))(_sed, 0)
        new_morph = constraint.prox_morph((5, 5,))(_morph, 0)
        true_morph = [[0.000000000, 0.000000000, 0.000000000, 0.000000000, 0.047619048],
                      [0.095238095, 0.142857143, 0.190476190, 0.238095238, 0.285714286],
                      [0.333333333, 0.380952381, 0.428571429, 0.476190476, 0.523809524],
                      [0.571428571, 0.619047619, 0.666666667, 0.714285714, 0.761904762],
                      [0.809523810, 0.857142857, 0.904761905, 0.952380952, 1.000000000]]
        np.testing.assert_array_equal(new_sed, [0, 6, 8, 6, 0])
        np.testing.assert_almost_equal(new_morph, true_morph)

    def test_norms(self):
        x = np.hstack([np.arange(4), np.arange(3)[::-1]])
        y = np.hstack([np.arange(4), np.arange(3)[::-1]])
        X, Y = np.meshgrid(x, y)
        r = np.sqrt(X**2 + Y**2)

        l0 = r.copy()
        constraint = scarlet.constraint.L0Constraint(thresh=2.1)
        constraint.prox_morph(l0.shape)(l0, 1)

        l1 = r.copy()
        constraint = scarlet.constraint.L1Constraint(thresh=1)
        l1 = constraint.prox_morph(l1.shape)(l1, 1)

        true_l0 = [[0.0000000, 0.0000000, 0.0000000, 3.0000000, 0.0000000, 0.0000000, 0.0000000],
                   [0.0000000, 0.0000000, 2.2360680, 3.1622777, 2.2360680, 0.0000000, 0.0000000],
                   [0.0000000, 2.2360680, 2.8284271, 3.6055513, 2.8284271, 2.2360680, 0.0000000],
                   [3.0000000, 3.1622777, 3.6055513, 4.2426407, 3.6055513, 3.1622777, 3.0000000],
                   [0.0000000, 2.2360680, 2.8284271, 3.6055513, 2.8284271, 2.2360680, 0.0000000],
                   [0.0000000, 0.0000000, 2.2360680, 3.1622777, 2.2360680, 0.0000000, 0.0000000],
                   [0.0000000, 0.0000000, 0.0000000, 3.0000000, 0.0000000, 0.0000000, 0.0000000]]
        true_l1 = [[0.0000000, 0.0000000, 1.0000000, 2.0000000, 1.0000000, 0.0000000, 0.0000000],
                   [0.0000000, 0.4142136, 1.2360680, 2.1622777, 1.2360680, 0.4142136, 0.0000000],
                   [1.0000000, 1.2360680, 1.8284271, 2.6055513, 1.8284271, 1.2360680, 1.0000000],
                   [2.0000000, 2.1622777, 2.6055513, 3.2426407, 2.6055513, 2.1622777, 2.0000000],
                   [1.0000000, 1.2360680, 1.8284271, 2.6055513, 1.8284271, 1.2360680, 1.0000000],
                   [0.0000000, 0.4142136, 1.2360680, 2.1622777, 1.2360680, 0.4142136, 0.0000000],
                   [0.0000000, 0.0000000, 1.0000000, 2.0000000, 1.0000000, 0.0000000, 0.0000000]]
        np.testing.assert_almost_equal(l0, true_l0)
        np.testing.assert_almost_equal(l1, true_l1)

    def test_monotonic(self):
        X = np.arange(25, dtype=float).reshape(5, 5)
        # Nearest neighbor
        _X = X.copy()
        constraint = scarlet.constraint.DirectMonotonicityConstraint(use_nearest=True, exact=False, thresh=0)
        constraint.prox_morph(_X.shape)(_X, 0)
        new_X = [[0.0, 1.0, 2.0, 3.0, 4.0],
                 [5.0, 6.0, 7.0, 8.0, 9.0],
                 [10.0, 11.0, 12.0, 12.0, 12.0],
                 [11.0, 12.0, 12.0, 12.0, 12.0],
                 [12.0, 12.0, 12.0, 12.0, 12.0]]
        np.testing.assert_array_equal(_X, new_X)

        # Weighted
        # We need to clear the cache, since this has already been created
        Cache._cache = {}
        _X = X.copy()
        constraint = scarlet.constraint.DirectMonotonicityConstraint(use_nearest=False, exact=False, thresh=0)
        constraint.prox_morph(_X.shape)(_X, 0)
        new_X = [[0.000000000, 1.000000000, 2.000000000, 3.000000000, 4.000000000],
                 [5.000000000, 6.000000000, 7.000000000, 8.000000000, 9.000000000],
                 [9.742640687, 11.000000000, 12.000000000, 12.000000000, 10.828427125],
                 [11.030627697, 11.707106781, 12.000000000, 12.000000000, 11.771236166],
                 [11.556349186, 11.868867239, 11.914213562, 11.983249156, 11.928090416]]
        np.testing.assert_almost_equal(_X, new_X)

        Cache._cache = {}
        # Test that use_nearest=True and thresh !=0 are incompatible
        constraint = scarlet.constraint.DirectMonotonicityConstraint(use_nearest=True, thresh=.25)
        with pytest.raises(ValueError):
            constraint.prox_morph((5, 5))

        # Use a threshold to force a gradient of 75% or steeper
        Cache._cache = {}
        _X = X.copy()
        constraint = scarlet.constraint.DirectMonotonicityConstraint(use_nearest=False, thresh=.25)
        constraint.prox_morph(_X.shape)(_X, 0)
        new_X = [[0.000000000, 1.000000000, 2.000000000, 3.000000000, 4.000000000],
                 [5.000000000, 6.000000000, 7.000000000, 7.242640687, 5.806841831],
                 [5.801461031, 9.000000000, 12.000000000, 9.000000000, 6.074431804],
                 [5.895545844, 7.681980515, 9.000000000, 7.681980515, 5.935521488],
                 [4.988519641, 5.949655012, 6.170941546, 5.949655012, 4.997301087]]
        np.testing.assert_almost_equal(_X, new_X)

    @pytest.mark.skipif('torch' not in sys.modules, reason="pytorch is not installed")
    def test_symmetry(self):
        Cache._cache = {}
        X = torch.arange(25, dtype=torch.float32).reshape(5, 5)
        _X = X.clone()
        constraint = scarlet.constraint.DirectSymmetryConstraint()
        constraint.prox_morph(X.shape)(_X, 0)
        result = torch.ones_like(X) * 12
        np.testing.assert_array_equal(_X, result)

        _X = X.clone()
        constraint = scarlet.constraint.DirectSymmetryConstraint(sigma=0)
        constraint.prox_morph(X.shape)(_X, 0)
        np.testing.assert_array_equal(_X, X)

        _X = X.clone()
        constraint = scarlet.constraint.DirectSymmetryConstraint(sigma=0.5)
        constraint.prox_morph(X.shape)(_X, 0)
        result = [[6.0, 6.5, 7.0, 7.5, 8.0],
                  [8.5, 9.0, 9.5, 10.0, 10.5],
                  [11.0, 11.5, 12.0, 12.5, 13.0],
                  [13.5, 14.0, 14.5, 15.0, 15.5],
                  [16.0, 16.5, 17.0, 17.5, 18.0]]
        np.testing.assert_array_equal(_X, result)

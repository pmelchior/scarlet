import pytest
import numpy as np

import scarlet


class TestProx(object):
    def test_prox_monotonic(self):
        X = np.arange(25).reshape(5, 5).astype(np.float64)
        # First test the nearest neighbor implementation
        prox = scarlet.operator.prox_strict_monotonic(X.shape, use_nearest=True, thresh=0)
        didx = [12, 17, 13, 7, 11, 18, 16, 6, 8, 14, 10, 22, 2, 9, 23, 5, 15, 3, 1, 19, 21, 0, 4, 20, 24]
        assert prox.func == scarlet.operator._prox_strict_monotonic
        assert prox.keywords["thresh"] == 0
        assert prox.keywords['ref_idx'] == [6, 7, 7, 7, 8, 11, 12, 12, 12, 13, 11, 12, 12,
                                            12, 13, 11, 12, 12, 12, 13, 16, 17, 17, 17, 18]
        np.testing.assert_array_equal(prox.keywords["dist_idx"], didx)
        _X = X.copy()
        prox(_X, 0.0)
        nearest_X = [[0.0, 1.0, 2.0, 3.0, 4.0],
                     [5.0, 6.0, 7.0, 8.0, 9.0],
                     [10.0, 11.0, 12.0, 12.0, 12.0],
                     [11.0, 12.0, 12.0, 12.0, 12.0],
                     [12.0, 12.0, 12.0, 12.0, 12.0]]
        np.testing.assert_array_equal(_X, nearest_X)

        # Test that use_nearest=True and thresh !=0 are incompatible
        with pytest.raises(ValueError):
            prox = scarlet.operator.prox_strict_monotonic(X.shape, use_nearest=True, thresh=.25)

        # Now test weighted monotonicity
        prox = scarlet.operator.prox_strict_monotonic(X.shape, use_nearest=False, thresh=0)
        assert prox.func == scarlet.operator._prox_weighted_monotonic
        assert prox.keywords["thresh"] == 0
        np.testing.assert_array_equal(prox.keywords["didx"], didx[1:])
        np.testing.assert_array_equal(prox.keywords["offsets"], [-6, -5, -4, -1, 1, 4, 5, 6])
        _X = X.copy()
        prox(_X, 0.0)
        weighted_X = [[0., 1., 2., 3., 4.],
                      [5., 6., 7., 8., 9.],
                      [9.74264069, 11., 12., 12., 10.82842712],
                      [11.0306277, 11.70710678, 12., 12., 11.77123617],
                      [11.55634919, 11.86886724, 11.91421356, 11.98324916, 11.92809042]]
        np.testing.assert_almost_equal(_X, weighted_X)
        # Use a threshold to force a gradient of 75% or steeper
        prox = scarlet.operator.prox_strict_monotonic(X.shape, use_nearest=False, thresh=.25)
        threshold_X = [[0.000000000, 1.000000000, 2.000000000, 3.000000000, 4.000000000],
                       [5.000000000, 6.000000000, 7.000000000, 7.242640687, 5.806841831],
                       [5.801461031, 9.000000000, 12.000000000, 9.000000000, 6.074431804],
                       [5.895545844, 7.681980515, 9.000000000, 7.681980515, 5.935521488],
                       [4.988519641, 5.949655012, 6.170941546, 5.949655012, 4.997301087]]
        _X = X.copy()
        prox(_X, 0.0)
        np.testing.assert_almost_equal(_X, threshold_X)

    # Skipped test for unused prox_cone
    def test_prox_center_on(self):
        X = np.zeros((5, 5))
        _X = X.copy()
        scarlet.operator.prox_center_on(_X, 0)
        result = X.copy()
        result[2, 2] = 1e-10
        np.testing.assert_array_equal(_X, result)

        _X = X.copy()
        scarlet.operator.prox_center_on(_X, 0, tiny=.1)
        result = X.copy()
        result[2, 2] = .1
        np.testing.assert_array_equal(_X, result)

    def test_prox_max_unity(self):
        X = np.arange(11, dtype=float)
        _X = X.copy()
        print(scarlet.operator.prox_max_unity(_X, 0))
        result = X/10
        np.testing.assert_array_equal(_X, result)

    def test_prox_sed_on(self):
        X = np.zeros((5, 5))
        _X = X.copy()
        scarlet.operator.prox_sed_on(_X, 0)
        result = np.ones_like(X) * 1e-10
        np.testing.assert_array_equal(_X, result)

        _X = X.copy()
        scarlet.operator.prox_sed_on(_X, 0, .1)
        result = np.ones_like(X) * .1
        np.testing.assert_array_equal(_X, result)

    def test_soft_symmetry(self):
        X = np.arange(25, dtype=float).reshape(5, 5)
        _X = X.copy()
        scarlet.operator.prox_soft_symmetry(_X, 0)
        result = np.ones_like(X) * 12
        np.testing.assert_array_equal(_X, result)

        _X = X.copy()
        scarlet.operator.prox_soft_symmetry(_X, 0, 0)
        np.testing.assert_array_equal(_X, X)

        _X = X.copy()
        scarlet.operator.prox_soft_symmetry(_X, 0, .5)
        result = [[6.0, 6.5, 7.0, 7.5, 8.0],
                  [8.5, 9.0, 9.5, 10.0, 10.5],
                  [11.0, 11.5, 12.0, 12.5, 13.0],
                  [13.5, 14.0, 14.5, 15.0, 15.5],
                  [16.0, 16.5, 17.0, 17.5, 18.0]]
        np.testing.assert_array_equal(_X, result)

    def test_bulge_disk(self):
        disk_sed = np.arange(5)
        bulge_sed = np.arange(5)[::-1]
        new_sed = scarlet.operator.project_disk_sed(bulge_sed, disk_sed)
        np.testing.assert_array_equal(new_sed, [0, 5, 6, 7, 4])

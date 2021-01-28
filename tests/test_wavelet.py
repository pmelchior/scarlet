from functools import partial

import numpy as np
import scarlet
from numpy.testing import assert_almost_equal, assert_equal


class TestWavelet(object):
    def get_psfs(self, sigmas, boxsize):
        psf = scarlet.GaussianPSF(sigmas, boxsize=boxsize)
        return psf.get_model()

    """Test the wavelet object"""

    def test_transform_inverse(self):
        psf = self.get_psfs(1, 128)
        starlet_transform = scarlet.Starlet(psf, lvl=4)

        # Test number of levels
        assert_equal(starlet_transform.coefficients.shape[1], 4)

        # Test inverse
        inverse = starlet_transform.image
        assert_almost_equal(inverse, psf)

    def test_setter(self):
        psf = self.get_psfs(1, 128)
        starlet_transform = scarlet.Starlet(psf, lvl=4)

        starlet = scarlet.Starlet(psf, lvl=4)
        star_coeff = starlet.coefficients
        star_coeff[0, :, 10:20, :] = 0

        new_starlet = scarlet.Starlet(coefficients=star_coeff)
        assert_almost_equal(new_starlet.image, starlet.image)
        # Test inverse

        star_coeff[0, :, :, :] = 0
        assert_almost_equal(starlet.image[0], psf[0] * 0)

from functools import partial

import numpy as np
import scarlet
import scarlet.wavelet as wavelet
from numpy.testing import assert_almost_equal, assert_equal


class TestWavelet(object):

    def get_psfs(self, shape, sigmas):

        shape_ = (None, *shape)
        psfs = np.array([
            scarlet.PSF(partial(scarlet.psf.gaussian, sigma=s), shape=shape_).image[0]
            for s in sigmas
        ])

        psfs /= psfs.sum(axis=(1, 2))[:, None, None]
        return psfs


    """Test the wavelet object"""
    def test_transform_inverse(self):
        """Test matching two 2D psfs
        """
        # Narrow PSF
        shape = (128,128)
        psf = self.get_psfs(shape, [1])
        # Wide PSF
        starlet_transform = wavelet.Starlet(psf, lvl = 4)

        # Test number of levels
        assert_equal(starlet_transform.coefficients.shape[1], 4)

        # Test inverse
        inverse = starlet_transform.image
        assert_almost_equal(inverse, psf)

    def test_setter(self):
        """Test matching two 2D psfs
        """
        # Narrow PSF
        shape = (128,128)
        psf = self.get_psfs(shape, [1])
        # Wide PSF
        starlet = wavelet.Starlet(psf, lvl = 4)
        star_coeff = starlet.coefficients
        star_coeff[0,:, 10:20, :] = 0

        new_starlet = wavelet.Starlet(coefficients=star_coeff)
        assert_almost_equal(new_starlet.image, starlet.image)
        # Test inverse

        star_coeff[0, :, :, :] = 0
        assert_almost_equal(starlet.image[0], psf[0]*0)
from functools import partial

import numpy as np
import scarlet
import scarlet.wavelet as wavelet
from numpy.testing import assert_array_equal, assert_almost_equal, assert_equal


class TestWavelet(object):

    def get_psfs(self, shape, sigmas):

        shape_ = (None, *shape)
        psfs = np.array([
            scarlet.PSF(partial(scarlet.psf.gaussian, sigma=s), shape=shape_).image[0]
            for s in sigmas
        ])

        psfs /= psfs.sum(axis=(1, 2))[:, None, None]
        return psfs


    """Test the Fourier object"""
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
        inverse = wavelet.Starlet.from_starlet(starlet_transform.coefficients).image
        assert_almost_equal(inverse, psf)


    def test_getitem(self):
        """Test matching two PSFs with a spectral dimension
        """
        # PSF
        shape = (128, 128)
        psf = self.get_psfs(shape, [1])
        # Slice
        starlet_transform = wavelet.Starlet(psf)
        sliced = starlet_transform[0, 20:, :-10]
        assert_array_equal(sliced.image.shape, [1, 108, 118])
        assert_array_equal(sliced.coefficients.shape, [1, 7, 108, 118])

import numpy as np
from numpy.testing import assert_array_equal, assert_almost_equal
import astropy.wcs

import scarlet


def get_airy_wcs():
    wcs = astropy.wcs.WCS(naxis=2)
    # Set up an "Airy's zenithal" projection
    wcs.wcs.crpix = [-234.75, 8.3393]
    wcs.wcs.cdelt = np.array([-0.066667, 0.066667])
    wcs.wcs.crval = [0, -90]
    wcs.wcs.ctype = ["RA---AIR", "DEC--AIR"]
    wcs.wcs.set_pv([(2, 1, 45.0)])
    return wcs


class TestObservation(object):
    def get_psfs(self, shape, sigmas):
        psfs = np.array([
            scarlet.psf.generate_psf_image(scarlet.psf.gaussian, shape, amplitude=1, sigma=s)
            for s in sigmas
        ])

        normalized = psfs.copy()
        normalized /= psfs.sum(axis=(1, 2))[:, None, None]

        return psfs, normalized

    def test_frame(self):
        # Minimal initialization
        shape = (5, 11, 13)
        frame = scarlet.Frame(shape)
        wcs = get_airy_wcs()
        psfs = np.arange(1, 716).reshape(5, 11, 13)
        norm_psfs = psfs / psfs.sum(axis=(1,2))[:, None, None]

        assert frame.C == 5
        assert frame.Ny == 11
        assert frame.Nx == 13
        assert frame.shape == shape
        assert frame.psfs is None
        assert_array_equal(frame.get_pixel((5.1, 1.3)), (5, 1))

        # Full initialization
        frame = scarlet.Frame(shape, wcs=wcs, psfs=norm_psfs)
        assert frame.C == 5
        assert frame.Ny == 11
        assert frame.Nx == 13
        assert frame.shape == shape
        assert_almost_equal(frame.psfs.sum(axis=(1, 2)), [1]*5)

        skycoord = [210.945, -73.1]
        assert_array_equal(frame.get_pixel(skycoord), [-110, -202])

    def test_init(self):
        images = np.arange(1, 430, dtype=np.float32).reshape(3, 11, 13)
        weights = np.ones_like(images)
        psfs = np.arange(1, 76).reshape(3, 5, 5)
        norm_psfs = psfs / psfs.sum(axis=(1,2))[:, None, None]
        wcs = get_airy_wcs()
        channels = np.arange(len(images))

        # Minimal init
        obs = scarlet.Observation(images)
        assert obs.frame.C == 3
        assert obs.frame.Ny == 11
        assert obs.frame.Nx == 13
        assert obs.frame.shape == images.shape
        assert obs.frame.psfs is None
        assert_array_equal(obs.frame.get_pixel((5.1, 1.3)), (5, 1))
        assert obs.weights == 1
        assert obs.frame.channels is None

        # Full init
        obs = scarlet.Observation(images, psfs=norm_psfs, weights=weights, wcs=wcs, channels=channels)
        assert obs.frame.C == 3
        assert obs.frame.Ny == 11
        assert obs.frame.Nx == 13
        assert obs.frame.shape == images.shape
        assert_almost_equal(obs.frame.psfs.image, norm_psfs)
        assert_almost_equal(obs.frame.psfs.sum(axis=(1, 2)), [1]*3)
        assert_array_equal(obs.weights, weights)
        assert_array_equal(obs.frame.channels, channels)

        skycoord = [210.945, -73.1]
        assert_array_equal(obs.frame.get_pixel(skycoord), [-110, -202])

    def test_psf_match(self):
        shape = (43, 43)
        target_psf = self.get_psfs(shape, [.9])[1]
        psfs, truth = self.get_psfs(shape, [2.1, 1.1, 3.5])
        psfs /= psfs.sum(axis=(1,2))[:,None,None]

        frame = scarlet.Frame(psfs.shape, psfs=target_psf)
        observation = scarlet.Observation(psfs, psfs)
        observation.match(frame)
        result = observation.render(np.array([target_psf[0]]*len(psfs)))

        assert_almost_equal(result, truth)

    def test_render(self):
        shape = (43, 43)
        target_psf = self.get_psfs(shape, [.9])[1][0]
        target_psf = target_psf[None]
        psfs, normalized = self.get_psfs(shape, [2.1, 1.1, 3.5])
        psfs /= psfs.sum(axis=(1,2))[:,None,None]

        ry = rx = 21
        coords = [[33, 31], [43, 33], [54, 26], [68, 72]]
        images = np.zeros((3, 101, 101))
        for coord in coords:
            py, px = coord
            images[:, py-ry:py+ry+1, px-rx:px+rx+1] += normalized

        model = np.zeros_like(images)
        for coord in coords:
            py, px = coord
            model[:, py-ry:py+ry+1, px-rx:px+rx+1] += target_psf

        frame = scarlet.Frame(images.shape, psfs=target_psf)
        observation = scarlet.Observation(images, psfs=psfs)
        observation.match(frame)
        result = observation.render(model)
        assert_almost_equal(result, images)

    def test_get_loss(self):
        shape = (3,4,5)
        frame = scarlet.Frame(shape)
        images = np.arange(60).reshape(shape)
        weights = np.ones_like(images) * 2
        observation = scarlet.Observation(images, weights=weights).match(frame)
        model = 4 * np.ones_like(images)
        true_loss = 0.5 * np.sum((weights * (model-images))**2)
        assert_almost_equal(true_loss, observation.get_loss(model))

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

        norm_psf = psfs.copy()
        norm_psf /= psfs.sum(axis=(1, 2))[:, None, None]
        normalized = psfs.copy()
        normalized /= psfs.sum(axis=(1, 2))[:, None, None]

        return psfs, normalized

    def test_scene(self):
        # Minimal initialization
        shape = (5, 11, 13)
        scene = scarlet.Scene(shape)
        wcs = get_airy_wcs()
        psfs = np.arange(1, 716).reshape(5, 11, 13)
        norm_psfs = psfs / np.array([10296, 30745, 51194, 71643, 92092])[:, None, None]

        assert scene.B == 5
        assert scene.Ny == 11
        assert scene.Nx == 13
        assert scene.shape == shape
        assert scene.psfs is None
        assert_array_equal(scene.get_pixel((5.1, 1.3)), (5, 1))

        # Full initialization
        scene = scarlet.Scene(shape, wcs, psfs)
        assert scene.B == 5
        assert scene.Ny == 11
        assert scene.Nx == 13
        assert scene.shape == shape
        assert_almost_equal(scene.psfs, norm_psfs)
        assert_almost_equal(scene.psfs.sum(axis=(1, 2)), [1]*5)

        skycoord = [210.945, -73.1]

        assert_array_equal(scene.get_pixel(skycoord), [-110, -202])


    def test_init(self):
        images = np.arange(1, 430).reshape(3, 11, 13)
        weights = np.ones_like(images)
        psfs = np.arange(1, 76).reshape(3, 5, 5)
        norm_psfs = psfs / np.array([325, 950, 1575])[:, None, None]
        wcs = get_airy_wcs()
        structure = (0, 1, 1, 1, 0)

        # Minimal init
        obs = scarlet.Observation(images)
        assert obs.B == 3
        assert obs.Ny == 11
        assert obs.Nx == 13
        assert obs.shape == images.shape
        assert obs.psfs is None
        assert_array_equal(obs.get_pixel((5.1, 1.3)), (5, 1))
        assert obs.weights == 1
        assert obs.structure is None

        # Full init
        obs = scarlet.Observation(images, psfs, weights, wcs, structure=structure)
        assert obs.B == 3
        assert obs.Ny == 11
        assert obs.Nx == 13
        assert obs.shape == images.shape
        assert_almost_equal(obs.psfs, norm_psfs)
        assert_almost_equal(obs.psfs.sum(axis=(1, 2)), [1]*3)
        assert_array_equal(obs.weights, weights)
        assert_array_equal(obs.structure, structure)

        skycoord = [210.945, -73.1]

        assert_array_equal(obs.get_pixel(skycoord), [-110, -202])


    def test_psf_match(self):
        shape = (43, 43)
        target_psf = self.get_psfs(shape, [.9])[1][0]
        psfs, truth = self.get_psfs(shape, [2.1, 1.1, 3.5])

        scene = scarlet.Scene(psfs.shape, psfs=target_psf)
        observation = scarlet.Observation(psfs, psfs)
        observation.match(scene)
        result = observation.get_model(np.array([target_psf]*len(psfs)))

        assert_almost_equal(result, truth)

    def test_get_model(self):
        shape = (43, 43)
        target_psf = self.get_psfs(shape, [.9])[1][0]
        psfs, normalized = self.get_psfs(shape, [2.1, 1.1, 3.5])

        ry = rx = 21
        coords = [[33, 31], [43, 33], [54, 26], [68, 72]]
        images = np.zeros((3, 101, 101))
        for coord in coords:
            py, px = coord
            images[:, py-ry:py+ry+1, px-rx:px+rx+1] += normalized

        model = np.zeros_like(images)
        for coord in coords:
            py, px = coord
            model[:, py-ry:py+ry+1, px-rx:px+rx+1] += target_psf[None]

        scene = scarlet.Scene(images.shape, psfs=target_psf)
        observation = scarlet.Observation(images, psfs)
        observation.match(scene)
        result = observation.get_model(model)
        assert_almost_equal(result, images)

    def test_get_loss(self):
        images = np.arange(60).reshape(3, 4, 5)
        weights = np.ones_like(images) * 0.5
        observation = scarlet.Observation(images, weights=weights)
        model = 0.5 * images
        true_loss = 0.5 * np.sum((weights * (model-images))**2)
        assert_almost_equal(true_loss, observation.get_loss(model))

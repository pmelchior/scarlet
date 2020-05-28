import numpy as np
from numpy.testing import assert_array_equal, assert_almost_equal
from functools import partial
import scarlet


class TestObservation(object):
    def get_psfs(self, shape, sigmas):

        shape_ = (None, *shape)
        psfs = np.array([
            scarlet.PSF(partial(scarlet.psf.gaussian, sigma=s), shape=shape_).image[0]
            for s in sigmas
        ])

        psfs /= psfs.sum(axis=(1, 2))[:, None, None]
        return psfs

    def test_render_loss(self):
        # model frame with minimal PSF
        shape0 = (3, 13, 13)
        s0 = 0.9
        model_psf = scarlet.PSF(partial(scarlet.psf.gaussian, sigma=s0), shape=shape0)
        shape = (3, 43, 43)
        channels = np.arange(shape[0])
        model_frame = scarlet.Frame(shape, psfs=model_psf, channels=channels)

        # insert point source manually into center for model
        origin = (0, shape[1]//2 - shape0[1]//2, shape[2]//2 - shape0[2]//2)
        bbox = scarlet.Box(shape0, origin=origin)
        model = np.zeros(shape)
        box = np.stack([model_psf.image[0] for c in range(shape[0])], axis=0)
        bbox.insert_into(model, box)

        # generate observation with wider PSFs
        psf = scarlet.PSF(self.get_psfs(shape[1:], [2.1, 1.1, 3.5]))
        images = np.ones(shape)
        observation = scarlet.Observation(images, psfs=psf, channels=channels)
        observation.match(model_frame)
        model_ = observation.render(model)
        assert_almost_equal(model_, psf.image)

        # compute the expected loss
        weights = 1
        log_norm = np.prod(images.shape) / 2 * np.log(2*np.pi) + np.sum(np.log(1 / weights)) / 2
        true_loss = log_norm + np.sum(weights * (model_ - images)** 2) / 2
        assert_almost_equal(observation.get_loss(model), true_loss)

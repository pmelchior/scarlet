import numpy as np
from numpy.testing import assert_almost_equal, assert_array_equal

import scarlet


def init_data(shape, coords, amplitudes=None, convolve=True, dtype=np.float32):
    import scipy.signal

    B, Ny, Nx = shape
    K = len(coords)

    if amplitudes is None:
        amplitudes = np.ones((K,))
    assert K == len(amplitudes)

    _seds = [
        np.arange(B, dtype=dtype),
        np.arange(B, dtype=dtype)[::-1],
        np.ones((B,), dtype=dtype)
    ]
    seds = np.array([_seds[n % 3]*amplitudes[n] for n in range(K)])

    morphs = np.zeros((K, Ny, Nx))
    for k, coord in enumerate(coords):
        morphs[k, coord[0], coord[1]] = 1
    images = seds.T.dot(morphs.reshape(K, -1)).reshape(shape)

    if convolve:
        psf_radius = 20
        psf_shape = (2*psf_radius+1, 2*psf_radius+1)
        target_psf = scarlet.psf.generate_psf_image(scarlet.psf.gaussian, psf_shape, sigma=.9).image
        target_psf /= target_psf.sum()

        psfs = np.array([
            scarlet.psf.generate_psf_image(scarlet.psf.gaussian, psf_shape, sigma=1+.2*b).image
            for b in range(B)
        ], dtype=dtype)
        psfs /= psfs.max(axis=(1, 2))[:, None, None]
        # Convolve the image with the psf in each channel
        # Use scipy.signal.convolve without using FFTs as a sanity check
        images = np.array([scipy.signal.convolve(img, psf, method="direct", mode="same")
                           for img, psf in zip(images, psfs)], dtype=dtype)
        # Convolve the true morphology with the target PSF,
        # also using scipy.signal.convolve as a sanity check
        morphs = np.array([scipy.signal.convolve(m, target_psf, method="direct", mode="same")
                           for m in morphs], dtype=dtype)
        morphs /= morphs.max()
        psfs /= psfs.sum(axis=(1, 2))[:, None, None]

    channels = range(len(images))
    return target_psf, psfs, images, channels, seds, morphs


class TestBlend(object):
    def test_model_render(self):
        shape = (6, 31, 55)
        coords = [(20, 10), (10, 30), (17, 42)]
        result = init_data(shape, coords, [3, 2, 1], dtype=np.float64)
        target_psf, psfs, images, channels, seds, morphs = result

        # Test init with psfs
        frame = scarlet.Frame(images.shape, psfs=target_psf[None], dtype=np.float64)
        observation = scarlet.Observation(images, psfs=psfs).match(frame)

        sources = [scarlet.PointSource(frame, coord, observation) for coord in coords]
        blend = scarlet.Blend(sources, observation)
        model = observation.render(blend.get_model())

        assert_almost_equal(images, model, decimal=5)

        for s0, s in zip(sources, blend.sources):
            assert_array_equal(s.get_model(), s0.get_model())

    def test_fit_point_source(self):
        shape = (6, 31, 55)
        coords = [(20, 10), (10, 30), (17, 42)]
        amplitudes = [3, 2, 1]
        result = init_data(shape, coords, amplitudes, dtype=np.float64)
        target_psf, psfs, images, channels, seds, morphs = result
        B, Ny, Nx = shape

        frame = scarlet.Frame(images.shape, psfs=target_psf[None], dtype=np.float64)
        observation = scarlet.Observation(images, psfs=psfs).match(frame)
        sources = [scarlet.PointSource(frame, coord, observation) for coord in coords]
        blend = scarlet.Blend(sources, observation)
        # Try to run for 10 iterations
        # Since the model is already near exact, it should converge
        # on the 2nd iteration (since it doesn't calculate the initial loss)
        blend.fit(10)

        assert blend.it == 2
        assert_almost_equal(blend.mse, [3.875628098330452e-15, 3.875598349723412e-15], decimal=10)
        assert blend.mse[0] > blend.mse[1]

    def test_fit_extended_source(self):
        shape = (6, 31, 55)
        coords = [(20, 10), (10, 30), (17, 42)]
        amplitudes = [3, 2, 1]
        result = init_data(shape, coords, amplitudes, dtype=np.float64)
        target_psf, psfs, images, channels, seds, morphs = result
        B, Ny, Nx = shape

        frame = scarlet.Frame(images.shape, psfs=target_psf[None], dtype=np.float64)
        observation = scarlet.Observation(images, psfs=psfs).match(frame)
        bg_rms = np.ones((B,))
        sources = [scarlet.ExtendedSource(frame, coord, observation, bg_rms) for coord in coords]
        blend = scarlet.Blend(sources, observation)

        # Scale the input psfs by the observation and model psfs to ensure
        # the sources were initialized correctly
        psf_scale = observation.frame.psfs.max(axis=(1, 2)) / frame.psfs[0].max()
        scaled_seds = np.array([c.sed*psf_scale for c in blend.components])

        assert_almost_equal(scaled_seds, seds)

        # Fit the model
        blend.fit(100)
        assert blend.it < 20
        mse = np.array(blend.mse[:-1])
        _mse = np.array(blend.mse[1:])
        assert np.all(mse-_mse >= 0)

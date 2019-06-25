import pytest
import numpy as np
from numpy.testing import assert_almost_equal, assert_array_equal

import scarlet


def init_data(shape, coords, amplitudes=None, convolve=True):
    import scipy.signal

    B, Ny, Nx = shape
    K = len(coords)

    if amplitudes is None:
        amplitudes = np.ones((K,))
    assert K == len(amplitudes)

    _seds = [
        np.arange(B, dtype=float),
        np.arange(B, dtype=float)[::-1],
        np.ones((B,), dtype=float)
    ]
    seds = np.array([_seds[n % 3]*amplitudes[n] for n in range(K)])

    morphs = np.zeros((K, Ny, Nx))
    for k, coord in enumerate(coords):
        morphs[k, coord[0], coord[1]] = 1
    images = seds.T.dot(morphs.reshape(K, -1)).reshape(shape)

    if convolve:
        psf_radius = 20
        psf_shape = (2*psf_radius+1, 2*psf_radius+1)
        psf_center = (psf_radius, psf_radius)
        target_psf = scarlet.psf.generate_psf_image(scarlet.psf.gaussian, psf_shape, psf_center,
                                                    amplitude=1, sigma=.9)
        target_psf /= target_psf.sum()

        psfs = np.array([scarlet.psf.generate_psf_image(scarlet.psf.gaussian, psf_shape, psf_center,
                                                        amplitude=1, sigma=1+.2*b) for b in range(B)])
        # Convolve the image with the psf in each band
        # Use scipy.signal.convolve without using FFTs as a sanity check
        images = np.array([scipy.signal.convolve(img, psf, method="direct", mode="same")
                           for img, psf in zip(images, psfs)])
        # Convolve the true morphology with the target PSF,
        # also using scipy.signal.convolve as a sanity check
        morphs = np.array([scipy.signal.convolve(m, target_psf, method="direct", mode="same")
                           for m in morphs])
        morphs /= morphs.max()
    return target_psf, psfs, images, seds, morphs


class TestBlend(object):
    def test_init(self):
        shape = (6, 31, 55)
        coords = [(20, 10), (10, 30), (17, 42)]
        target_psf, psfs, images, seds, morphs = init_data(shape, coords, [3, 2, 1])

        images1 = images[:4]
        images2 = images[4:]
        psfs1 = psfs[:4]
        psfs2 = psfs[4:]
        # Test vanilla init
        observations = scarlet.Observation(images1)
        scene = scarlet.Scene(images.shape)
        sources = [scarlet.PointSource(coord, scene, observations) for coord in coords]
        blend = scarlet.Blend(scene, sources, observations)

        assert len(blend.observations) == 1
        assert blend.observations[0] == observations
        assert blend.mse == []
        assert blend.shape == scene.shape
        assert blend.psfs == scene.psfs

        # Test init with psfs
        obs1 = scarlet.Observation(images1, psfs=psfs1, structure=(1, 1, 1, 1, 0, 0))
        obs2 = scarlet.Observation(images2, psfs=psfs2, structure=(0, 0, 0, 0, 1, 1))
        observations = [obs1, obs2]
        scene = scarlet.Scene(images.shape, psfs=target_psf[None])
        sources = [scarlet.PointSource(coord, scene, observations) for coord in coords]

        with pytest.raises(AttributeError):
            obs1.psfs_fft
        with pytest.raises(AttributeError):
            obs2.psfs_fft
        blend = scarlet.Blend(scene, sources, observations)

        assert blend.observations == [obs1, obs2]
        assert blend.mse == []
        assert blend.shape == scene.shape
        assert_array_equal(blend.psfs, scene.psfs)
        assert_array_equal(obs1.psfs_fft.shape, (4, 81, 55))
        assert_array_equal(obs2.psfs_fft.shape, (2, 81, 55))

        model = np.vstack([obs1.get_model(blend.get_model()), obs2.get_model(blend.get_model())])
        assert_almost_equal(images, model)
        assert blend.converged is False

        for k in range(len(sources)):
            assert_array_equal(blend.sources[k].get_model(), sources[k].get_model())

    def test_fit_point_source(self):
        shape = (6, 31, 55)
        coords = [(20, 10), (10, 30), (17, 42)]
        amplitudes = [3, 2, 1]
        target_psf, psfs, images, seds, morphs = init_data(shape, coords, amplitudes)
        B, Ny, Nx = shape
        K = len(coords)

        observations = scarlet.Observation(images, psfs=psfs)
        scene = scarlet.Scene(images.shape, psfs=target_psf[None])
        sources = [scarlet.PointSource(coord, scene, observations) for coord in coords]
        blend = scarlet.Blend(scene, sources, observations)
        # Try to run for 10 iterations
        # Since the model is already near exact, it should converge
        # on the 2nd iteration (since it doesn't calculate the initial loss)
        blend.fit(10)
        assert blend.it == 2
        assert_almost_equal(blend.L_sed, 2.5481250470053265)
        assert_almost_equal(blend.L_morph, 9024.538938935855)
        assert_almost_equal(blend.mse, [3.875628098330452e-15, 3.875598349723412e-15], decimal=10)
        assert blend.mse[0] > blend.mse[1]

    def test_fit_extended_source(self):
        shape = (6, 31, 55)
        coords = [(20, 10), (10, 30), (17, 42)]
        amplitudes = [3, 2, 1]
        target_psf, psfs, images, seds, morphs = init_data(shape, coords, amplitudes)
        B, Ny, Nx = shape

        observations = scarlet.Observation(images, psfs=psfs)
        scene = scarlet.Scene(images.shape, psfs=target_psf[None])
        bg_rms = np.ones((B,))
        sources = [scarlet.ExtendedSource(coord, scene, observations, bg_rms) for coord in coords]
        blend = scarlet.Blend(scene, sources, observations)

        # Scale the input psfs by the observation and model psfs to ensure
        # the sources were initialized correctly
        psf_scale = observations.psfs.max(axis=(1, 2)) / scene.psfs[0].max()
        scaled_seds = np.array([src.sed*psf_scale for src in blend.sources])

        assert_almost_equal(scaled_seds, seds)

        # Fit the model
        blend.fit(100)
        assert blend.it == 13
        mse = np.array(blend.mse[:-1])
        _mse = np.array(blend.mse[1:])
        assert np.all(mse-_mse >= 0)

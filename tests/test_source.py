import pytest
import numpy as np
from numpy.testing import assert_almost_equal, assert_array_equal

import scarlet
from scarlet.source import SourceInitError


def create_sources(shape, coords, amplitudes=None):
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

    src = np.array([[.25, .5, .25], [.5, 1, .5], [.25, .5, .25]])
    morphs = np.zeros((K, Ny, Nx))
    for k, (cy, cx) in enumerate(coords):
        morphs[k, cy-1:cy+2, cx-1: cx+2] = src

    images = seds.T.dot(morphs.reshape(K, -1)).reshape(shape)

    return seds, morphs, images


class TestPointSource(object):
    def test_get_pixel_sed(self):
        images1 = np.arange(210, dtype=float).reshape(5, 6, 7)
        images2 = images1 / 4
        obs1 = scarlet.Observation(images1)
        obs2 = scarlet.Observation(images2)
        observations = [obs1, obs2]
        skycoord = (3, 2)
        true_sed1 = images1[:, skycoord[0], skycoord[1]]
        true_sed2 = images2[:, skycoord[0], skycoord[1]]
        true_sed = np.concatenate((true_sed1, true_sed2))
        sed = np.concatenate([scarlet.source.get_pixel_sed(skycoord, obs) for obs in observations])
        assert_array_equal(sed, true_sed)

    def test_point_source(self):
        shape = (5, 11, 21)
        coords = [(4, 8), (8, 11), (5, 16)]

        B, Ny, Nx = shape
        seds, morphs, images = create_sources(shape, coords, [2, 3, .1])
        psfs = np.array([[[.25, .5, .25], [.5, 1, .5], [.25, .5, .25]]])
        psfs /= psfs.sum(axis=(1,2))[:,None,None]

        frame = scarlet.Frame(images.shape)
        obs = scarlet.Observation(images).match(frame)

        src = scarlet.PointSource(frame, coords[0], obs)
        truth = np.zeros_like(src.morph)
        truth[coords[0]] = 1

        assert_array_equal(src.sed, seds[0])
        assert_array_equal(src.morph, truth)
        assert src.pixel_center == coords[0]
        assert src.symmetric is True
        assert src.monotonic is True
        assert src.center_step == 5
        assert src.delay_thresh == 10

        # frame PSF same as source
        frame = scarlet.Frame(images.shape, psfs=psfs)
        src = scarlet.PointSource(frame, coords[0], obs)

        # We need to multiply by 4 because of psf normalization
        assert_almost_equal(src.sed*4, seds[0])
        assert_almost_equal(morphs[0], src.morph)
        assert src.pixel_center == coords[0]


class TestExtendedSource(object):
    def test_get_best_fit_seds(self):
        shape = (7, 11, 21)
        coords = [(4, 8), (8, 11), (5, 16)]
        seds, morphs, images = create_sources(shape, coords)

        frame = scarlet.Frame(images.shape)
        obs = scarlet.Observation(images).match(frame)

        _seds = scarlet.source.get_best_fit_seds(morphs, frame, obs)

        assert_array_equal(_seds, seds)

    def test_build_detection_coadd(self):
        truth = np.array([[[0.05235454, 0.02073789, 0.04880617, 0.03637619, 0.02399899],
                           [0.03744485, 0.29331713, 0.52876383, 0.28429441, 0.04679640],
                           [0.02611349, 0.52057058, 1.02958156, 0.51620345, 0.02391584],
                           [0.03627858, 0.28669982, 0.54027293, 0.26347546, 0.05124271],
                           [0.03635369, 0.05010319, 0.04445647, 0.04545365, 0.02991638]],
                          [[0.00704491, 0.00566508, 0.00848275, 0.00673316, 0.00564367],
                           [0.00686349, 0.25730076, 0.50470101, 0.25151582, 0.00715177],
                           [0.00951771, 0.50618275, 1.00161330, 0.50362382, 0.00521353],
                           [0.00189936, 0.25494626, 0.50437369, 0.25620321, 0.00515993],
                           [0.00751398, 0.00719382, 0.00812517, 0.00260853, 0.00908961]],
                          [[0.66289178, 0.31370533, 0.62558879, 0.43856888, 0.72209347],
                           [0.96661099, 0.57698197, 1.58224512, 0.93506400, 1.08122335],
                           [0.99298264, 1.26054670, 1.58495813, 1.61148374, 0.82737327],
                           [1.05820433, 0.92412937, 1.24225533, 1.33838207, 0.79615945],
                           [0.82488505, 1.13293652, 0.93197919, 1.37564087, 0.96079598]]])
        true_cutoff = np.array([0.03630302, 0.00769658, 0.82658430])

        np.random.seed(0)
        shape = (5, 11, 21)
        coords = [(4, 8), (8, 11), (5, 16)]

        B, Ny, Nx = shape
        K = len(coords)
        seds, morphs, images = create_sources(shape, coords, [2, 3, .1])
        bg_rms = np.arange(1, B+1) / 10

        # Add noise to the image
        noise = np.random.rand(*shape) * bg_rms[:, None, None]
        images += noise

        frame = scarlet.Frame(shape)
        for k in range(K):
            observation = scarlet.Observation(images).match(frame)
            coadd, cutoff = scarlet.source.build_detection_coadd(seds[k], bg_rms, observation)
            cy, cx = coords[k]
            window = slice(cy-2, cy+3), slice(cx-2, cx+3)
            assert_almost_equal(coadd[window], truth[k])
            assert_almost_equal(cutoff, true_cutoff[k])

        with pytest.raises(ValueError):
            scarlet.source.build_detection_coadd(seds[0], np.zeros_like(bg_rms), observation, frame)

    def test_init_extended(self):
        shape = (5, 11, 15)
        B, Ny, Nx = shape

        x = np.linspace(-2, 2, 5)
        y = np.linspace(-2, 2, 5)
        x, y = np.meshgrid(x, y)
        r = np.sqrt(x**2 + y**2)

        true_sed = np.arange(B)
        true_morph = np.zeros(shape[1:])

        skycoord = (np.array(true_morph.shape) - 1) // 2
        cy, cx = skycoord
        true_morph[cy-2:cy+3, cx-2:cx+3] = 3-r

        morph = true_morph.copy()
        morph[5, 3] = 10

        # Test function
        images = true_sed[:, None, None] * morph[None, :, :]
        frame = scarlet.Frame(shape)
        observation = scarlet.Observation(images).match(frame)
        bg_rms = np.ones_like(true_sed) * 1e-3
        sed, morph = scarlet.source.init_extended_source(skycoord, frame, observation, bg_rms)

        assert_array_equal(sed/3, true_sed)
        assert_almost_equal(morph*3, true_morph)

        # Test ExtendedSource.__init__
        src = scarlet.ExtendedSource(frame, skycoord, observation, bg_rms)
        assert_array_equal(src.pixel_center, skycoord)
        assert src.symmetric is True
        assert src.monotonic is True
        assert src.center_step == 5
        assert src.delay_thresh == 10

        assert_array_equal(src.sed/3, true_sed)
        assert_almost_equal(src.morph*3, true_morph)

        # Test monotonicity
        morph = true_morph.copy()
        morph[5, 5] = 2

        images = true_sed[:, None, None] * morph[None, :, :]
        frame = scarlet.Frame(shape)
        observation = scarlet.Observation(images).match(frame)
        bg_rms = np.ones_like(true_sed) * 1e-3
        sed, morph = scarlet.source.init_extended_source(skycoord, frame, observation, bg_rms,
                                                         symmetric=False)

        _morph = true_morph.copy()
        _morph[5, 5] = 1.5816233815926433
        assert_array_equal(sed/3, true_sed)
        assert_almost_equal(morph*3, _morph)

        # Test symmetry
        morph = true_morph.copy()
        morph[5, 5] = 2

        images = true_sed[:, None, None] * morph[None, :, :]
        frame = scarlet.Frame(shape)
        observation = scarlet.Observation(images).match(frame)
        bg_rms = np.ones_like(true_sed) * 1e-3
        sed, morph = scarlet.source.init_extended_source(skycoord, frame, observation, bg_rms,
                                                         monotonic=False)

        assert_array_equal(sed/3, true_sed)
        assert_almost_equal(morph*3, true_morph)

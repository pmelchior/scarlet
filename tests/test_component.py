import pytest
import numpy as np
from numpy.testing import assert_almost_equal, assert_array_equal

import scarlet


class TestCubeComponent:
    def test_model(self):
        frame_shape = (10, 20, 30)
        frame = scarlet.Frame(frame_shape, channels=np.arange(10))

        shape = (5, 4, 6)
        cube = np.zeros(shape)
        on_location = (1, 2, 3)
        cube[on_location] = 1
        cube = scarlet.Parameter(cube, name="cube")
        origin = (2, 3, 4)
        bbox = scarlet.Box(shape, origin=origin)

        component = scarlet.CubeComponent(frame, cube, bbox=bbox)
        model = component.get_model(frame=frame)

        # everything zero except at one location?
        test_loc = tuple(np.array(on_location) + np.array(origin))
        mask = np.zeros(model.shape, dtype="bool")
        mask[test_loc] = True
        assert_array_equal(model[~mask], 0)
        assert model[test_loc] == 1


class TestFactorizedComponent:
    def test_model(self):
        frame_shape = (10, 20, 30)
        frame = scarlet.Frame(frame_shape, channels=np.arange(10))

        shape = (5, 4, 6)
        on_location = (1, 2, 3)
        sed = np.zeros(shape[0])
        sed[on_location[0]] = 1
        morph = np.zeros(shape[1:])
        morph[on_location[1:]] = 1

        origin = (2, 3, 4)
        box = scarlet.Box(shape, origin=origin)
        spectrum = scarlet.TabulatedSpectrum(frame, sed, bbox=box[0])
        morphology = scarlet.ImageMorphology(frame, morph, bbox=box[1:])

        component = scarlet.FactorizedComponent(frame, spectrum, morphology)
        model = component.get_model(frame=frame)

        # everything zero except at one location?
        test_loc = tuple(np.array(on_location) + np.array(origin))
        mask = np.zeros(model.shape, dtype="bool")
        mask[test_loc] = True
        assert_array_equal(model[~mask], 0)
        assert model[test_loc] == 1

        # now with shift
        shift_loc = (0, 1, 0)
        shift = scarlet.Parameter(np.array(shift_loc[1:]), step=0.1, name="shift")
        morphology = scarlet.ImageMorphology(
            frame, morph, bbox=box[1:], shifting=True, shift=shift
        )

        component = scarlet.FactorizedComponent(frame, spectrum, morphology)
        model = component.get_model(frame=frame)

        # everything zero except at one location?
        test_loc = tuple(np.array(on_location) + np.array(origin) + np.array(shift_loc))
        mask = np.zeros(model.shape, dtype="bool")
        mask[test_loc] = True
        assert_almost_equal(model[~mask], 0)
        assert_almost_equal(model[test_loc], 1)


class TestFunctionComponent:
    def test_model(self):
        frame_shape = (10, 20, 30)
        frame = scarlet.Frame(frame_shape, channels=np.arange(10))

        shape = (5, 4, 6)
        origin = (2, 3, 4)
        box = scarlet.Box(shape, origin=origin)

        on_location = (1, 2, 3)
        sed = np.zeros(shape[0])
        sed[on_location[0]] = 1
        spectrum = scarlet.TabulatedSpectrum(frame, sed, bbox=box[0])

        # construct functional morphology where the parameter sets
        # the location of single pixel that is on
        class OnePixelMorphology(scarlet.Morphology):
            def __init__(self, model_frame, on_pixel, bbox=None):
                self._bbox = bbox
                self._on_pixel = scarlet.Parameter(on_pixel, step=1, name="on_pixel")
                super().__init__(model_frame, self._on_pixel, bbox=bbox)

            def get_model(self, *params):
                on_pixel = self._on_pixel
                for p in params:
                    if p._value.name == "on_pixel":
                        on_pixel = p

                morph = np.zeros(self._bbox.shape)
                morph[tuple(np.round(on_pixel).astype("int"))] = 1
                return morph

        morphology = OnePixelMorphology(
            frame, np.array(on_location[1:], dtype="float"), bbox=box[1:]
        )
        component = scarlet.FactorizedComponent(frame, spectrum, morphology)
        model = component.get_model(frame=frame)

        # everything zero except at one location?
        test_loc = tuple(np.array(on_location) + np.array(origin))
        mask = np.zeros(model.shape, dtype="bool")
        mask[test_loc] = True
        assert_array_equal(model[~mask], 0)
        assert model[test_loc] == 1


class TestCombinedComponent:
    def test_model(self):
        frame_shape = (10, 20, 30)
        frame = scarlet.Frame(frame_shape, channels=np.arange(10))

        shape = (5, 4, 6)
        origin = (2, 3, 4)
        box = scarlet.Box(shape, origin=origin)
        on_location1 = (1, 2, 3)
        cube = np.zeros(shape)
        cube[on_location1] = 1
        cube = scarlet.Parameter(cube, name="cube")
        component1 = scarlet.CubeComponent(frame, cube, bbox=box)

        # make factorized component with a different origin
        on_location2 = (1, 1, 1)
        sed = np.zeros(shape[0])
        sed[on_location2[0]] = 1
        morph = np.zeros(shape[1:])
        morph[on_location2[1:]] = 1

        spectrum = scarlet.TabulatedSpectrum(frame, sed, bbox=box[0])
        morphology = scarlet.ImageMorphology(frame, morph, bbox=box[1:])
        component2 = scarlet.FactorizedComponent(frame, spectrum, morphology)

        combined = scarlet.CombinedComponent([component1, component2])
        model = combined.get_model(frame=frame)

        # everything zero except at one location?
        test_locs = [
            tuple(np.array(on_location1) + np.array(origin)),
            tuple(np.array(on_location2) + np.array(origin)),
        ]
        mask = np.zeros(model.shape, dtype="bool")
        for test_loc in test_locs:
            mask[test_loc] = True
        assert_array_equal(model[~mask], 0)
        assert_array_equal(model[mask], 1)

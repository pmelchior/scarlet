import pytest
import numpy as np
from numpy.testing import assert_almost_equal, assert_array_equal

import scarlet

class TestCubeComponent:

    def test_model(self):
        frame_shape = (10, 20, 30)
        frame = scarlet.Frame(frame_shape, channels=np.arange(10))

        shape = (5,4,6)
        cube = np.zeros(shape)
        on_location = (1,2,3)
        cube[on_location] = 1
        cube = scarlet.Parameter(cube)
        origin = (2,3,4)
        bbox = scarlet.Box(shape, origin=origin)

        component = scarlet.CubeComponent(frame, bbox, cube)
        model = component.get_model(frame=frame)

        # everything zero except at one location?
        test_loc = tuple(np.array(on_location) + np.array(origin))
        mask = np.zeros(model.shape, dtype='bool')
        mask[test_loc] = True
        assert_array_equal(model[~mask], 0)
        assert model[test_loc] == 1


class TestFactorizedComponent:

    def test_model(self):
        frame_shape = (10, 20, 30)
        frame = scarlet.Frame(frame_shape, channels=np.arange(10))

        shape = (5,4,6)
        on_location = (1,2,3)
        sed = np.zeros(shape[0])
        sed[on_location[0]] = 1
        morph = np.zeros(shape[1:])
        morph[on_location[1:]] = 1

        sed = scarlet.Parameter(sed)
        morph = scarlet.Parameter(morph)
        origin = (2,3,4)
        bbox = scarlet.Box(shape, origin=origin)

        component = scarlet.FactorizedComponent(frame, bbox, sed, morph)
        model = component.get_model(frame=frame)

        # everything zero except at one location?
        test_loc = tuple(np.array(on_location) + np.array(origin))
        mask = np.zeros(model.shape, dtype='bool')
        mask[test_loc] = True
        assert_array_equal(model[~mask], 0)
        assert model[test_loc] == 1

        # now with shift
        shift_loc = (0,1,0)
        shift = scarlet.Parameter(np.array(shift_loc[1:]))
        component = scarlet.FactorizedComponent(frame, bbox, sed, morph, shift=shift)
        model = component.get_model(frame=frame)

        # everything zero except at one location?
        test_loc = tuple(np.array(on_location) + np.array(origin) + np.array(shift_loc))
        mask = np.zeros(model.shape, dtype='bool')
        mask[test_loc] = True
        assert_almost_equal(model[~mask], 0)
        assert_almost_equal(model[test_loc],1)

class TestFunctionComponent:

    def test_model(self):
        frame_shape = (10, 20, 30)
        frame = scarlet.Frame(frame_shape, channels=np.arange(10))

        shape = (5,4,6)
        on_location = (1,2,3)
        sed = np.zeros(shape[0])
        sed[on_location[0]] = 1

        fparams = np.array(on_location[1:])
        def f(*params):
            morph = np.zeros(shape[1:])
            morph[tuple(params)] = 1
            return morph

        sed = scarlet.Parameter(sed)
        fparams = scarlet.Parameter(fparams)
        origin = (2,3,4)
        bbox = scarlet.Box(shape, origin=origin)

        component = scarlet.FunctionComponent(frame, bbox, sed, fparams, f)
        model = component.get_model(frame=frame)

        # everything zero except at one location?
        test_loc = tuple(np.array(on_location) + np.array(origin))
        mask = np.zeros(model.shape, dtype='bool')
        mask[test_loc] = True
        assert_array_equal(model[~mask], 0)
        assert model[test_loc] == 1


class TestComponentTree:

    def test_model(self):
        frame_shape = (10, 20, 30)
        frame = scarlet.Frame(frame_shape, channels=np.arange(10))

        shape = (5,4,6)
        on_location = (1,2,3)
        cube = np.zeros(shape)
        cube[on_location] = 1
        cube = scarlet.Parameter(cube)
        origin1 = (2,3,4)
        bbox1 = scarlet.Box(shape, origin=origin1)
        component1 = scarlet.CubeComponent(frame, bbox1, cube)

        sed = np.zeros(shape[0])
        sed[on_location[0]] = 1
        morph = np.zeros(shape[1:])
        morph[on_location[1:]] = 1

        sed = scarlet.Parameter(sed)
        morph = scarlet.Parameter(morph)

        origin2 = (5,6,7)
        bbox2 = scarlet.Box(shape, origin=origin2)
        component2 = scarlet.FactorizedComponent(frame, bbox2, sed, morph)

        tree = scarlet.ComponentTree([component1, component2])
        model = tree.get_model(frame=frame)

        # everything zero except at one location?
        test_locs = [tuple(np.array(on_location) + np.array(origin1)), tuple(np.array(on_location) + np.array(origin2))]
        mask = np.zeros(model.shape, dtype='bool')
        for test_loc in test_locs:
            mask[test_loc] = True
        assert_array_equal(model[~mask], 0)
        assert_array_equal(model[mask], 1)

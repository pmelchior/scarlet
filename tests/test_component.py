import pytest
import numpy as np
from numpy.testing import assert_almost_equal, assert_array_equal

import scarlet

class UpdateComponent(scarlet.Component):
    def __init__(self, norm="sed", *args, **kwargs):
        self.norm = norm
        super().__init__(*args, **kwargs)

    def update(self):
        scarlet.update.normalized(self, self.norm)


class TestComponent(object):
    def test_init(self):
        # Minimal init
        sed = np.arange(5)
        morph = np.arange(20).reshape(4, 5)
        shape = (len(sed), morph.shape[0], morph.shape[1])
        frame = scarlet.Frame(shape)
        component = scarlet.Component(frame, sed, morph)

        assert_array_equal(component.sed, sed)
        assert_array_equal(component.morph, morph)
        assert component._index is None
        assert component._parent is None
        assert isinstance(component._morph, scarlet.Parameter)
        assert isinstance(component._sed, scarlet.Parameter)

        # explicitly set sed and morph fixed
        sed = scarlet.Parameter(sed, name="sed", fixed=True)
        morph = scarlet.Parameter(morph, name="morph", fixed=False)
        component = scarlet.Component(frame, sed, morph)
        assert component._sed.fixed is True
        assert component._morph.fixed is False

    def test_properties(self):
        sed = np.arange(5)
        morph = np.arange(20).reshape(4, 5)
        shape = (len(sed), morph.shape[0], morph.shape[1])
        frame = scarlet.Frame(shape)
        component = scarlet.Component(frame, sed, morph)

        assert component.frame.C == 5
        assert component.frame.Ny == 4
        assert component.frame.Nx == 5
        assert component.coord is None

    def test_methods(self):
        # get_model
        sed = np.arange(5)
        morph = np.arange(20).reshape(4, 5)
        shape = (len(sed), morph.shape[0], morph.shape[1])
        frame = scarlet.Frame(shape)
        component = scarlet.Component(frame, sed, morph)
        model = component.get_model()
        truth = sed[:, None, None] * morph[None, :, :]
        assert_array_equal(model, truth)

        other_sed = np.ones_like(sed)
        other_morph = morph + 10
        other_truth = other_sed[:, None, None] * other_morph[None, :, :]
        model = component.get_model()
        other_model = component.get_model(other_sed, other_morph)
        assert_array_equal(model, truth)
        assert_array_equal(other_model, other_truth)

        # get_flux
        flux = component.get_flux()
        true_flux = model.sum(axis=(1, 2))
        assert_array_equal(flux, true_flux)

        # empty update
        test = component.update()
        assert test is component
        # Nothing should be changed
        assert component.frame.C == 5
        assert component.frame.Ny == 4
        assert component.frame.Nx == 5
        assert component.coord is None

    def test_prior(self):
        pass

class TestComponentTree(object):
    def test_init(self):
        shape = (5, 5, 5)
        frame = scarlet.Frame(shape)
        sed1 = np.arange(shape[0], dtype=float)
        sed2 = np.ones_like(sed1)
        sed3 = np.arange(shape[0], dtype=float)[::-1]
        morph1 = np.arange(shape[1]*shape[2], dtype=float).reshape(shape[1], shape[2])
        morph2 = np.zeros_like(morph1)
        morph2[1:-1, 1:-1] = [[1, 2, 1], [2, 4, 2], [1, 2, 1]]
        morph3 = np.ones_like(morph1)
        c1 = scarlet.Component(frame, sed1, morph1)
        c2 = scarlet.Component(frame, sed2, morph2)
        c3 = scarlet.Component(frame, sed3, morph3)
        tree1 = scarlet.ComponentTree([c1, c2])
        tree2 = scarlet.ComponentTree([tree1, c3])

        # properties and tree coordinates
        assert c1._parent == tree1
        assert c2._parent == tree1
        assert tree1._parent == tree2
        assert c1._index == 0
        assert c2._index == 1
        assert tree1._index == 0
        assert c3._index == 1
        assert tree2._index is None
        assert c1.coord == (0, 0)
        assert c2.coord == (0, 1)
        assert c3.coord == (1,)
        assert tree1.coord == (0,)
        assert tree2.coord is None
        assert tree1.n_components == 2
        assert tree2.n_components == 3
        assert tree2.coord is None
        assert_array_equal(tree1.components, (c1, c2))
        assert_array_equal(tree2.components, (c1, c2, c3))
        assert_array_equal(tree1.sources, (c1, c2))
        assert_array_equal(tree2.sources, (tree1, c3))
        assert tree1.n_nodes == 2
        assert tree2.n_nodes == 2

        # shapes
        assert tree1.K == 2
        assert tree2.K == 3
        assert tree1.frame.C == 5
        assert tree2.frame.C == 5
        assert tree1.frame.Ny == 5
        assert tree1.frame.Nx == 5
        assert tree2.frame.Ny == 5
        assert tree2.frame.Nx == 5

        # get_model
        model1 = c1.get_model()
        model2 = c2.get_model()
        model3 = c3.get_model()
        tmodel1 = tree1.get_model()
        tmodel2 = tree2.get_model()
        assert_array_equal(model1, sed1[:, None, None] * morph1[None, :, :])
        assert_array_equal(model2, sed2[:, None, None] * morph2[None, :, :])
        assert_array_equal(model3, sed3[:, None, None] * morph3[None, :, :])
        assert_array_equal(tmodel1, model1 + model2)
        assert_array_equal(tmodel2, tmodel1 + model3)
        assert_array_equal(tmodel2, model1 + model2 + model3)

        # get_flux
        flux1 = c1.get_flux()
        flux2 = c2.get_flux()
        flux3 = c3.get_flux()
        tflux1 = tree1.get_flux()
        tflux2 = tree2.get_flux()
        assert_array_equal(tflux1, flux1 + flux2)
        assert_array_equal(tflux2, flux1 + flux2 + flux3)

    def test_items(self):
        sed = np.arange(3, dtype=float)
        morph = np.arange(25, dtype=float).reshape(5, 5)
        shape = (len(sed), morph.shape[0], morph.shape[1])
        frame = scarlet.Frame(shape)
        c1 = UpdateComponent(frame=frame, sed=sed, morph=morph)
        c2 = UpdateComponent(frame=frame, sed=sed, morph=morph)
        c3 = UpdateComponent(frame=frame, sed=sed, morph=morph)
        c4 = UpdateComponent(frame=frame, sed=sed, morph=morph)
        c5 = UpdateComponent("morph", frame, sed, morph)
        tree1 = scarlet.ComponentTree([c1, c2])
        tree2 = scarlet.ComponentTree([c3, c4])

        # Test iadd
        tree1 += tree2
        assert tree1.n_components == 4
        assert tree1.n_nodes == 4
        assert tree1.components == (c1, c2, c3, c4)
        assert tree1.sources == (c1, c2, c3, c4)

        tree1 += c5
        assert tree1.n_components == 5
        assert tree1.n_nodes == 5
        assert tree1.components == (c1, c2, c3, c4, c5)
        assert tree1.sources == (c1, c2, c3, c4, c5)
        assert tree2.n_components == 2
        assert tree2.n_nodes == 2
        assert tree2.components == (c3, c4)
        assert tree2.sources == (c3, c4)

        # Test getitem
        tree1[0] == c1
        tree1[-1] == c5

        # Test update
        tree2.update()
        sed_norm = sed.sum()
        morph_norm = morph.sum()
        assert_array_equal(c1.sed, sed)
        assert_array_equal(c2.sed, sed)
        assert_array_equal(c3.sed, sed / sed_norm)
        assert_array_equal(c4.sed, sed / sed_norm)
        assert_array_equal(c5.sed, sed)
        assert_array_equal(c1.morph, morph)
        assert_array_equal(c2.morph, morph)
        assert_array_equal(c3.morph, morph * sed_norm)
        assert_array_equal(c4.morph, morph * sed_norm)
        assert_array_equal(c5.morph, morph)

        tree1.update()
        assert_array_equal(c1.sed, sed / sed_norm)
        assert_array_equal(c2.sed, sed / sed_norm)
        assert_array_equal(c3.sed, sed / sed_norm)
        assert_array_equal(c4.sed, sed / sed_norm)
        assert_array_equal(c5.sed, sed * morph_norm)
        assert_array_equal(c1.morph, morph * sed_norm)
        assert_array_equal(c2.morph, morph * sed_norm)
        assert_array_equal(c3.morph, morph * sed_norm)
        assert_array_equal(c4.morph, morph * sed_norm)
        assert_array_equal(c5.morph, morph / morph_norm)

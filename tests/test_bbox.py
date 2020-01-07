import numpy as np
import scarlet


class TestBox(object):
    def test_from_data(self):
        x = np.arange(25).reshape(5, 5)
        x[0] = 0
        x[:, -2:] = 0
        bbox = scarlet.Box.from_data(x)
        assert bbox == scarlet.Box((4, 3), origin=(1, 0))

        x += 10
        bbox = scarlet.Box.from_data(x)
        assert bbox == scarlet.Box((5, 5), origin=(0, 0))

        bbox = scarlet.Box.from_data(x, min_value=10)
        assert bbox == scarlet.Box((4, 3), origin=(1, 0))

    def test_contains(self):
        bbox = scarlet.Box((6, 4, 3), origin=(0, 1, 0))
        p = (2, 2, 2)
        assert bbox.contains(p)

        p = (3, 0, 3)
        assert not bbox.contains(p)

        p = (7, 3, 3)
        assert not bbox.contains(p)

        p = (3, 3, -1)
        assert not bbox.contains(p)

    def test_extract_from(self):
        image = np.zeros((3, 5, 5))
        image[1, 1, 1] = 1

        # simple one pixel box extraction
        bbox = scarlet.Box.from_data(image)
        extracted = bbox.extract_from(image)
        assert extracted.shape == (1, 1, 1) and extracted[0, 0, 0] == 1

        # offset box extraction past boundary of image
        bbox = scarlet.Box.from_bounds((0, 3), (-2, 3), (-3, 2))
        extracted = bbox.extract_from(image)
        assert extracted.shape == (3, 5, 5) and extracted[1, 3, 4] == 1

    def test_insert_into(self):
        image = np.zeros((3, 5, 5))
        sub = np.zeros((3, 5, 5))
        sub[1, 3, 4] = 1
        bbox = scarlet.Box.from_bounds((0, 3), (-2, 3), (-3, 2))
        image = bbox.insert_into(image, sub)
        assert image.shape == (3, 5, 5) and image[1, 1, 1] == 1

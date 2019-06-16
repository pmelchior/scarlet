import numpy as np
import scarlet


class TestBox(object):
    def test_init(self):
        yx0 = (4, 1)
        height = 7
        width = 6
        bottom = yx0[0]
        left = yx0[1]
        top = yx0[0] + height - 1
        right = yx0[1] + width - 1
        # default initialization
        bbox1 = scarlet.bbox.Box(yx0, height, width)

        assert bbox1.width == width
        assert bbox1.height == height
        assert bbox1.bottom == bottom
        assert bbox1.top == top
        assert bbox1.left == left
        assert bbox1.right == right
        assert bbox1.slices == (slice(bottom, top+1), slice(left, right+1))
        assert bbox1.shape == (height, width)
        assert bbox1.is_empty is False

        # from_bounds
        bbox2 = scarlet.bbox.Box.from_bounds(bottom, top, left, right)
        assert bbox2.width == bbox1.width
        assert bbox2.height == bbox1.height
        assert bbox2.bottom == bbox1.bottom
        assert bbox2.top == bbox1.top
        assert bbox2.left == bbox1.left
        assert bbox2.right == bbox1.right
        assert bbox2.slices == bbox1.slices
        assert bbox2.shape == bbox1.shape
        assert bbox1.is_empty is False

        # Test empty bounding box
        bbox = scarlet.bbox.Box.from_bounds(10, 9, 5, 10)
        assert bbox.is_empty is True
        assert bbox.width == 0
        assert bbox.height == 0
        assert bbox.bottom is None
        assert bbox.top is None
        assert bbox.left is None
        assert bbox.right is None
        assert bbox.yx0 is None

    def test_operations(self):
        yx0 = (5, 10)
        height, width = 9, 7
        bbox1 = scarlet.bbox.Box(yx0, height, width)

        # Copy
        bbox2 = bbox1.copy()
        assert bbox1.bottom == bbox2.bottom
        assert bbox1.top == bbox2.top
        assert bbox1.left == bbox2.left
        assert bbox1.right == bbox2.right
        bbox2.yx0 = (0, 2)
        bbox2.height = 1
        bbox2.width = 3
        assert bbox1.bottom == bbox2.bottom + 5
        assert bbox1.top == bbox2.top + 13
        assert bbox1.left == bbox2.left + 8
        assert bbox1.right == bbox2.right + 12

        # Equal
        bbox2 = bbox1.copy()
        assert bbox1 == bbox2
        assert bbox1 & bbox2 == bbox1
        assert bbox1 | bbox2 == bbox1

        # Lower left
        bbox2 = scarlet.bbox.Box((3, 7), 9, 8)
        and12 = scarlet.bbox.Box.from_bounds(bbox1.bottom, bbox2.top, bbox1.left, bbox2.right)
        or12 = scarlet.bbox.Box.from_bounds(bbox2.bottom, bbox1.top, bbox2.left, bbox1.right)
        assert bbox1 != bbox2
        assert bbox1 & bbox2 == and12
        assert bbox1 | bbox2 == or12

        # Upper right
        bbox2 = scarlet.bbox.Box((7, 12), 11, 13)
        and12 = scarlet.bbox.Box.from_bounds(bbox2.bottom, bbox1.top, bbox2.left, bbox1.right)
        or12 = scarlet.bbox.Box.from_bounds(bbox1.bottom, bbox2.top, bbox1.left, bbox2.right)
        assert bbox1 != bbox2
        assert bbox1 & bbox2 == and12
        assert bbox1 | bbox2 == or12

        # No overlap
        bbox2 = scarlet.bbox.Box((30, 50), 10, 11)
        and12 = scarlet.bbox.Box((0, 0), width=0, height=0)
        or12 = scarlet.bbox.Box.from_bounds(bbox1.bottom, bbox2.top, bbox1.left, bbox2.right)
        assert bbox1 != bbox2
        assert bbox1 & bbox2 == and12
        assert bbox1 | bbox2 == or12
        assert and12.is_empty is True


class TestMethods(object):
    def test_trim(self):
        x = np.arange(25).reshape(5, 5)
        x[0] = 0
        x[:, -2:] = 0
        bbox = scarlet.bbox.trim(x)
        assert bbox == scarlet.bbox.Box((1, 0), 4, 3)

        x += 10
        bbox = scarlet.bbox.trim(x)
        assert bbox == scarlet.bbox.Box((0, 0), 5, 5)

        bbox = scarlet.bbox.trim(x, 10)
        assert bbox == scarlet.bbox.Box((1, 0), 4, 3)

    def test_flux_at_edge(self):
        x = np.arange(25).reshape(5, 5)
        edge = scarlet.bbox.flux_at_edge(x)
        assert edge is True

        x[0] = 0
        edge = scarlet.bbox.flux_at_edge(x)
        assert edge is True

        x[-1] = 0
        edge = scarlet.bbox.flux_at_edge(x)
        assert edge is True

        x[:, 0] = 0
        edge = scarlet.bbox.flux_at_edge(x)
        assert edge is True

        x[:, -1] = 0
        edge = scarlet.bbox.flux_at_edge(x)
        assert edge is False

        x = np.arange(25).reshape(5, 5)
        edge = scarlet.bbox.flux_at_edge(x, 5)
        assert edge is True

        x[0] = 5
        edge = scarlet.bbox.flux_at_edge(x, 5)
        assert edge is True

        x[-1] = 5
        edge = scarlet.bbox.flux_at_edge(x, 5)
        assert edge is True

        x[:, 0] = 5
        edge = scarlet.bbox.flux_at_edge(x, 5)
        assert edge is True

        x[:, -1] = 5
        edge = scarlet.bbox.flux_at_edge(x, 5)
        assert edge is False

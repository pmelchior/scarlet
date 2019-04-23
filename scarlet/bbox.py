import numpy as np


class BoundingBox(object):
    """Bounding Box for an object

    This is just a container with convenience methods to generate
    slices and properties for the portion of an array/tensor
    contained in the bounding box.

    Parameters
    ----------
    bounds: tuple of int
        Tuple of integers for the `(bottom, top, left, right)` sides
        of the box. If `bounds` is none then the `BoundingBox` is unbounded.
    """
    def __init__(self, bounds=None):
        if bounds is not None:
            self.bounded = True
            bottom, top, left, right = bounds
            self.bottom = int(bottom)
            self.top = int(top)
            self.left = int(left)
            self.right = int(right)
        else:
            self.bounded = False
            self.bottom = self.top = self.left = self.right = None

    @property
    def slices(self):
        """Slices for this bounding box
        """
        if not self.bounded:
            return slice(None), slice(None)
        return slice(self.bottom, self.top+1), slice(self.left, self.right+1)

    @property
    def width(self):
        """Width of the bounding box
        """
        return self.right-self.left

    @property
    def height(self):
        """Height of the bounding box
        """
        return self.top-self.bottom

    @property
    def shape(self):
        """Shape of the image contained in this bounding box

        This is really just (height, width).
        """
        return (self.height, self.width)

    def __or__(self, other):
        """Union of two bounding boxes

        Parameters
        ----------
        other: `BoundingBox`
            The other bounding box in the union

        Returns
        -------
        result: `BoundingBox`
            The smallest rectangular box that contains *both* boxes.
        """
        bottom = min(self.bottom, other.bottom)
        top = max(self.top, other.top)
        left = min(self.left, other.left)
        right = max(self.right, other.right)
        return BoundingBox((bottom, top, left, right))

    def __and__(self, other):
        """Intersection of two bounding boxes

        Parameters
        ----------
        other: `BoundingBox`
            The other bounding box in the intersection

        Returns
        -------
        result: `BoundingBox`
            The rectangular box that is in the overlap region
            of both boxes.
        """
        bottom = max(self.bottom, other.bottom)
        top = min(self.top, other.top)
        left = max(self.left, other.left)
        right = min(self.right, other.right)
        return BoundingBox((bottom, top, left, right))

    def __str__(self):
        return "(({0}, {1}), ({2}, {3}))".format(self.bottom, self.top, self.left, self.right)

    def __repr__(self):
        result = "<BoundingBox yx0={0}, width={1}, height={2}>"
        return result.format((self.bottom, self.left), self.width, self.height)

    def copy(self):
        return BoundingBox((self.bottom, self.top, self.left, self.right))


def trim(X, min_value=0):
    """Trim a tensor or array

    Parameters
    ----------
    X: tensor or array
        2D Matrix to trim
    min_value: float
        Minimum value of the result.
        `X` will be trimmed so that all values
        over `min_value` are contained in the result.

    Returns
    -------
    bbox: `BoundingBox`
        Bounding box for the trimmed `result` (bottom, top, left, right)
    """
    nonzero = np.where(X > 0)
    left = nonzero[1].min()
    right = nonzero[1].max()
    bottom = nonzero[0].min()
    top = nonzero[0].max()
    return BoundingBox((bottom, top, left, right))


def flux_at_edge(X, min_value):
    """Determine if an edge of the input has flux above min_value

    Parameters
    ----------
    X: tensor or array
        2D matrix to evaluate
    min_value: float
        Minimum value to trigger a positive result.

    Returns
    -------
    result: bool
        Whether or not any edge pixels are above the minimum_value
    """
    return bool(max(X[:, 0].max(), X[:, -1].max(), X[0].max(), X[-1].max()) > min_value)


def resize(X, bbox, min_value):
    """Check if enough flux at the bounding box warrants a resize

    Parameters
    ----------
    X: tensor or array
        2D matrix to evaluate
    bbox: `BoundingBox`
        The current bounding box of `X`.
    min_value: float
        Minimum value to trigger a resize.

    Returns
    -------
    result: `BoundingBox`
        The bounding box for the trimmed version of `X`.
        If `result` is `None` then the source does not need to
        be trimmed.
    """
    if flux_at_edge(X[bbox.slices], min_value):
        return trim(X, min_value)
    return bbox

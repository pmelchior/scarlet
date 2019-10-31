import numpy as np


class Box(object):
    """Bounding Box for an object

    This is just a container with convenience methods to generate
    slices and properties for the portion of an array/tensor
    contained in the bounding box.

    Parameters
    ----------
    yx0: tuple
        Minimum (y,x) value of the box (lower left corner).
    height: int
        Height of the box.
    width: int
        Width of the box
    """
    def __init__(self, yx0, height, width):
        if width <= 0 or height <= 0:
            self.yx0 = None
            self.width = self.height = 0
        else:
            self.yx0 = yx0
            self.height = height
            self.width = width

    @staticmethod
    def from_shape(shape):
        """Initialize a box to cover `image`

        Parameters
        ----------
        shape: tuple
            2D (Height, Width)

        Returns
        -------
        bbox: `Box`
            A new box bounded by the shape.
        """
        return Box((0,0), *shape)

    @staticmethod
    def from_image(image):
        """Initialize a box to cover `image`

        Parameters
        ----------
        image: array-like
            2D image

        Returns
        -------
        bbox: `Box`
            A new box bounded by the image.
        """
        return Box.from_shape(image.shape)

    @staticmethod
    def from_bounds(bottom, top, left, right):
        """Initialize a box from its bounds

        Parameters
        ----------
        bottom: int
            Minimum in the y direction.
        top: int
            Maximum in the y direction.
        left: int
            Minimum in the x direction.
        right: int
            Maximum in the x direction.

        Returns
        -------
        bbox: `Box`
            A new box bounded by the input bounds.
        """
        return Box((bottom, left), top-bottom, right-left)

    @staticmethod
    def from_data(X, min_value=0):
        """Define range of `X` above `min_value`

        Parameters
        ----------
        X: array-like
            Data to threshold
        min_value: float
            Minimum value of the result.

        Returns
        -------
        bbox: `Box`
            Bounding box for the thresholded `X` (bottom, top, left, right)
        """
        nonzero = np.where(X > min_value)
        left = nonzero[1].min()
        right = nonzero[1].max()
        bottom = nonzero[0].min()
        top = nonzero[0].max()
        return Box.from_bounds(bottom, top, left, right)

    @property
    def is_empty(self):
        """Whether the box has a width and height
        """
        return self.width == 0 or self.height == 0

    def slices_for(self, im_or_shape):
        """Slices for this bounding box
        """
        if hasattr(im_or_shape, 'shape'):
            shape = im_or_shape.shape
        else:
            shape = im_or_shape

        im_box = Box.from_shape(shape)
        overlap = self & im_box
        if overlap.is_empty:
            return slice(0, 0), slice(0, 0)
        return slice(overlap.bottom, overlap.top), slice(overlap.left, overlap.right)

    def image_to_box(self, image, box=None):
        imbox = Box.from_image(image)

        if box is None:
            box = np.zeros(self.shape)
        assert box.shape == self.shape
        boxbox = Box.from_image(box)

        # imbox now in the frame of this bbox (i.e. of box)
        imbox -= self.yx0
        overlap = imbox & boxbox

        box[overlap.slices_for(box)] = image[self.slices_for(image)]
        return box

    def box_to_image(self, box, image):
        imbox = Box.from_image(image)
        boxbox = Box.from_image(box)

        # imbox now in the frame of this bbox (i.e. of box)
        imbox -= self.yx0
        overlap = imbox & boxbox

        image[self.slices_for(image)] = box[overlap.slices_for(box)]
        return image

    @property
    def bottom(self):
        """Minimum y value
        """
        if self.is_empty:
            return None
        return self.yx0[0]

    @property
    def left(self):
        """Minimum x value
        """
        if self.is_empty:
            return None
        return self.yx0[1]

    @property
    def top(self):
        """Maximum y value
        """
        if self.is_empty:
            return None
        return self.yx0[0] + self.height

    @property
    def right(self):
        """Maximum x value
        """
        if self.is_empty:
            return None
        return self.yx0[1] + self.width

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
        other: `Box`
            The other bounding box in the union

        Returns
        -------
        result: `Box`
            The smallest rectangular box that contains *both* boxes.
        """
        bottom = min(self.bottom, other.bottom)
        top = max(self.top, other.top)
        left = min(self.left, other.left)
        right = max(self.right, other.right)
        return Box.from_bounds(bottom, top, left, right)

    def __and__(self, other):
        """Intersection of two bounding boxes

        If there is no intersection between the two bounding
        boxes then an empty bounding box is returned.

        Parameters
        ----------
        other: `Box`
            The other bounding box in the intersection

        Returns
        -------
        result: `Box`
            The rectangular box that is in the overlap region
            of both boxes.
        """
        bottom = max(self.bottom, other.bottom)
        top = min(self.top, other.top)
        left = max(self.left, other.left)
        right = min(self.right, other.right)
        if top < bottom or right < left:
            return Box((0, 0), width=0, height=0)
        return Box.from_bounds(bottom, top, left, right)

    def __str__(self):
        return "(({0}, {1}), ({2}, {3}))".format(self.bottom, self.top, self.left, self.right)

    def __repr__(self):
        result = "<Box yx0={0}, height={1}, width={2}>"
        return result.format(self.yx0, self.height, self.width)

    def __iadd__(self, yx):
        self.yx0 = (self.yx0[0] + yx[0], self.yx0[1] + yx[1])
        return self

    def __isub__(self, yx):
        self.yx0 = (self.yx0[0] - yx[0], self.yx0[1] - yx[1])
        return self

    def copy(self):
        return Box((self.yx0[0], self.yx0[1]), self.height, self.width)

    def __eq__(self, other):
        return all([
            self.left == other.left,
            self.right == other.right,
            self.top == other.top,
            self.bottom == other.bottom,
        ])


def flux_at_edge(X, min_value=0):
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

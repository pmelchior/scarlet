import numpy as np


class Box:
    """Bounding Box for an object

    A Bounding box describes the location of a data unit in the global/model coordinate
    system. It is used to identify spatial and channel overlap and to map from model
    to observed frames and back.

    Parameters
    ----------
    origin: tuple
        Minimum (y,x) value of the box (lower left corner).
    height: int
        Height of the box.
    width: int
        Width of the box
    """
    def __init__(self, shape, origin=(0,0,0)):
        # bbox always in 3D
        if len(shape) == 2:
            shape = (0, *shape)
        assert len(shape) == 3
        self.shape = shape

        if len(origin) == 2:
            origin = (0, *origin)
        assert len(origin) == 3
        self.origin = origin

    @staticmethod
    def from_image(image):
        """Initialize a box to cover `image`

        Parameters
        ----------
        image: array-like
            2D image

        Returns
        -------
        bbox: `:class:`scarlet.bbox.Box`
            A new box bounded by the image.
        """
        return Box(image.shape)

    @staticmethod
    def from_bounds(front, back, bottom, top, left, right):
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
        bbox: :class:`scarlet.bbox.Box`
            A new box bounded by the input bounds.
        """
        if back < front:
            back, front = front, back
        if top < bottom:
            top, bottom = bottom, top
        if right < left:
            right, left = left, right
        return Box((back-front, top-bottom, right-left), origin=(front, bottom, left))

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
        bbox: :class:`scarlet.bbox.Box`
            Bounding box for the thresholded `X` (bottom, top, left, right)
        """
        sel = X > min_value
        if sel.any():
            nonzero = np.where(sel)
            bounds = []
            for dim in range(len(X.shape)):
                bounds.append(nonzero[dim].min())
                bounds.append(nonzero[dim].max())
            if len(X.shape) == 2:
                bounds.insert(0,0)
                bounds.insert(1,0)
        else:
            bounds = [0,] * 6
        return Box.from_bounds(*bounds)

    @property
    def is_empty(self):
        """Whether the box has non-zero volume
        """
        return any(np.array(self.shape) == 0)

    def contains(self, p):
        """Whether the box cotains a given point
        """
        if len(p) == 2:
            p = (0,*p)

        for d in range(len(self.shape)):
            if p[d] < self.origin[d] or p[d] > self.origin[d] + self.shape[d]:
                return False
        return True

    def slices_for(self, im_or_shape):
        """Slices for `im_or_shape` to be limited to this bounding box.

        Parameters
        ----------
        im_or_shape: array or tuple
            Array or shpae of the array to be sliced

        Returns
        -------
        If shape is 2D: `slice_y`, `slice_x`
        If shape is 3: `slice(None)`, `slice_y`, `slice_x`
        """
        if hasattr(im_or_shape, 'shape'):
            shape = im_or_shape.shape
        else:
            shape = im_or_shape
        assert len(shape) in [2,3]

        im_box = Box(shape)
        overlap = self & im_box
        zslice, yslice, xslice = slice(overlap.front, overlap.back), slice(overlap.bottom, overlap.top), slice(overlap.left, overlap.right)

        if len(shape) == 2:
            return yslice, xslice
        else:
            return zslice, yslice, xslice

    def image_to_box(self, image, box=None):
        """Excize box described by this bbox from image

        Parameters
        ----------
        image: array
            Full origin image
        box: array
            Excized destination image

        Returns
        -------
        box: array
        """
        imbox = Box.from_image(image)

        if box is None:
            if len(image.shape) == 3:
                box = np.zeros(self.shape)
            else:
                box = np.zeros(self.shape[1:])
        boxbox = Box.from_image(box)

        # imbox now in the frame of this bbox (i.e. of box)
        imbox -= self.origin
        overlap = imbox & boxbox
        box[overlap.slices_for(box)] = image[self.slices_for(image)]
        return box

    def box_to_image(self, box, image):
        """Insert `box` into `image` according to this bbox

        Inverse operation to :func:`~scarlet.bbox.Box.image_to_box`.

        Parameters
        ----------
        box: array
            Excized box
        image: array
            Full image

        Returns
        -------
        image: array
        """
        imbox = Box.from_image(image)
        boxbox = Box.from_image(box)

        # imbox now in the frame of this bbox (i.e. of box)
        imbox -= self.origin
        overlap = imbox & boxbox
        image[self.slices_for(image)] = box[overlap.slices_for(box)]
        return image

    @property
    def C(self):
        """Number of channels in the model
        """
        return self.shape[0]

    @property
    def Ny(self):
        """Number of pixel in the y-direction
        """
        return self.shape[1]

    @property
    def Nx(self):
        """Number of pixels in the x-direction
        """
        return self.shape[2]

    @property
    def front(self):
        """Minimum z value
        """
        return self.origin[0]

    @property
    def bottom(self):
        """Minimum y value
        """
        return self.origin[1]

    @property
    def left(self):
        """Minimum x value
        """
        return self.origin[2]

    @property
    def back(self):
        """Maximum y value
        """
        return self.origin[0] + self.shape[0]

    @property
    def top(self):
        """Maximum y value
        """
        return self.origin[1] + self.shape[1]

    @property
    def right(self):
        """Maximum x value
        """
        return self.origin[2] + self.shape[2]

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
        front = min(self.front, other.front)
        back = max(self.back, other.back)
        bottom = min(self.bottom, other.bottom)
        top = max(self.top, other.top)
        left = min(self.left, other.left)
        right = max(self.right, other.right)
        return Box.from_bounds(front, back, bottom, top, left, right)

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
        front = max(self.front, other.front)
        back = min(self.back, other.back)
        bottom = max(self.bottom, other.bottom)
        top = min(self.top, other.top)
        left = max(self.left, other.left)
        right = min(self.right, other.right)
        return Box.from_bounds(front, back, bottom, top, left, right)

    def __str__(self):
        return "Box({0}..{1}, {2}..{3}, {4}..{5})".format(self.front, self.back, self.bottom, self.top, self.left, self.right)

    def __repr__(self):
        result = "<Box shape={0}, origin={1}>"
        return result.format(self.shape, self.origin)

    def __iadd__(self, offset):
        self.origin = tuple([a + o for a,o in zip(self.origin, offset)])
        return self

    def __isub__(self, offset):
        self.origin = tuple([a - o for a,o in zip(self.origin, offset)])
        return self

    def __copy__(self):
        return Box(self.shape, offset=self.offset)

    def __eq__(self, other):
        return self.shape == other.shape and self.origin == other.origin


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

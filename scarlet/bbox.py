import numpy as np


class Box:
    """Bounding Box for an object

    A Bounding box describes the location of a data unit in the global/model coordinate
    system. It is used to identify spatial and channel overlap and to map from model
    to observed frames and back.

    Parameters
    ----------
    shape: tuple
        Size of the box
    origin: tuple
        Front/low/left corner coordinate of the box
    """

    def __init__(self, shape, origin=None):
        self.shape = tuple(shape)
        if origin is None:
            origin = (0,) * len(shape)
        assert len(origin) == len(shape)
        self.origin = tuple(origin)

    @staticmethod
    def from_image(image):
        """Initialize a box to cover `image`

        Parameters
        ----------
        image: array-like

        Returns
        -------
        bbox: `:class:`scarlet.bbox.Box`
            A new box bounded by the image.
        """
        return Box(image.shape)

    @staticmethod
    def from_bounds(*args):
        """Initialize a box from its bounds

        Parameters
        ----------
        args: ints
            Min/Max coordinate for every dimension

        Returns
        -------
        bbox: :class:`scarlet.bbox.Box`
            A new box bounded by the input bounds.
        """
        assert len(args) % 2 == 0
        dims = len(args) // 2
        shape = [args[dim * 2 + 1] - args[dim * 2] for dim in range(dims)]
        origin = [args[dim * 2] for dim in range(dims)]
        return Box(shape, origin=origin)

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
            Bounding box for the thresholded `X`
        """
        sel = X > min_value
        if sel.any():
            nonzero = np.where(sel)
            bounds = []
            for dim in range(len(X.shape)):
                bounds.append(nonzero[dim].min())
                bounds.append(nonzero[dim].max() + 1)
        else:
            bounds = [0,] * len(X.shape) * 2
        return Box.from_bounds(*bounds)

    def contains(self, p):
        """Whether the box cotains a given coordinate `p`
        """
        assert len(p) == self.D

        for d in range(self.D):
            if p[d] < self.origin[d] or p[d] > self.origin[d] + self.shape[d]:
                return False
        return True

    def slices_for(self, im_or_shape):
        """Slices for `im_or_shape` to be limited to this bounding box.

        Parameters
        ----------
        im_or_shape: array or tuple
            Array or shape of the array to be sliced

        Returns
        -------
        slices for every dimension
        """
        if hasattr(im_or_shape, "shape"):
            shape = im_or_shape.shape
        else:
            shape = im_or_shape
        assert len(shape) == self.D

        im_box = Box(shape)
        overlap = self & im_box
        return tuple(slice(overlap.start[d], overlap.stop[d]) for d in range(self.D))

    def extract_from(self, image, sub=None):
        """Extract sub-image described by this bbox from image

        Parameters
        ----------
        image: array
            Full image
        sub: array
            Extracted image

        Returns
        -------
        sub: array
        """
        imbox = Box.from_image(image)

        if sub is None:
            sub = np.zeros(self.shape)
        subbox = Box.from_image(sub)

        # imbox now in the frame of this bbox (i.e. of box)
        imbox -= self.origin
        overlap = imbox & subbox
        sub[overlap.slices_for(sub)] = image[self.slices_for(image)]
        return sub

    def insert_into(self, image, sub):
        """Insert `sub` into `image` according to this bbox

        Inverse operation to :func:`~scarlet.bbox.Box.extract_from`.

        Parameters
        ----------
        image: array
            Full image
        sub: array
            Extracted sub-image

        Returns
        -------
        image: array
        """
        imbox = Box.from_image(image)
        subbox = Box.from_image(sub)

        # imbox now in the frame of this bbox (i.e. of box)
        imbox -= self.origin
        overlap = imbox & subbox
        image[self.slices_for(image)] = sub[overlap.slices_for(sub)]
        return image

    @property
    def D(self):
        """Dimensionality of this BBox
        """
        return len(self.shape)

    @property
    def start(self):
        """Tuple of start coordinates
        """
        return self.origin

    @property
    def stop(self):
        """Tuple of stop coordinates (last is included)
        """
        return tuple(o + s for o, s in zip(self.origin, self.shape))

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
        assert other.D == self.D
        bounds = []
        for d in range(self.D):
            bounds.append(min(self.start[d], other.start[d]))
            bounds.append(max(self.stop[d], other.stop[d]))
        return Box.from_bounds(*bounds)

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
        assert other.D == self.D
        bounds = []
        for d in range(self.D):
            bounds.append(max(self.start[d], other.start[d]))
            bounds.append(min(self.stop[d], other.stop[d]))
        return Box.from_bounds(*bounds)

    def __repr__(self):
        result = "<Box shape={0}, origin={1}>"
        return result.format(self.shape, self.origin)

    def __iadd__(self, offset):
        self.origin = tuple([a + o for a, o in zip(self.origin, offset)])
        return self

    def __isub__(self, offset):
        self.origin = tuple([a - o for a, o in zip(self.origin, offset)])
        return self

    def __copy__(self):
        return Box(self.shape, offset=self.offset)

    def __eq__(self, other):
        return self.shape == other.shape and self.origin == other.origin

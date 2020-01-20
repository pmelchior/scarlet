import numpy as np
from .measure import Moments


class Box:
    """Bounding Box for an object

    A Bounding box describes the location of a data unit in the global/model coordinate
    system. It is used to identify spatial and channel overlap and to map from model
    to observed frames and back.

    The `BBox` code is agnostic about the meaning of the dimensions.
    We generally use this convention:

    - 2D shapes denote (Height, Width)
    - 3D shapes denote (Channels, Height, Width)

    Parameters
    ----------
    shape: tuple
        Size of the box
    origin: tuple
        Minimum corner coordinate of the box
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
    def from_bounds(*bounds):
        """Initialize a box from its bounds

        Parameters
        ----------
        bounds: tuple of (min,max) pairs
            Min/Max coordinate for every dimension

        Returns
        -------
        bbox: :class:`scarlet.bbox.Box`
            A new box bounded by the input bounds.
        """
        shape = [max(0, cmax - cmin) for cmin, cmax in bounds]
        origin = [cmin for cmin, cmax in bounds]
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
                bounds.append((nonzero[dim].min(), nonzero[dim].max() + 1))
        else:
            bounds = [[0, 0]] * len(X.shape)
        return Box.from_bounds(*bounds)

    @staticmethod
    def from_moments(image, fwhm_frac=1, fwhm_func=None):
        """Generate a bounding box from a set of moments

        Parameters
        ----------
        image: array
            Full image to use for moment calculation.
        fwhm_frac: float
            Fraction of the FWHM to use for the bounding box
        fwhm_func: Function
            For multiband images, this function is used to
            choose how the fwhm in x and y is calculated.
            By default (`fwhm_func=None`) the minimum 2nd moment
            is used from all of the bands.

        Returns
        -------
        box: `~scarlet.bbox.Box`
            The box to use for the moment calculation.
        """
        if fwhm_func is None:
            func = np.min
        else:
            func = fwhm_func
        moments = Moments(image)
        sigma_y, sigma_x = np.sqrt(moments[2, 0]), np.sqrt(moments[0, 2])
        fwhm_y, fwhm_x = (np.ceil(2.355*sigma_y)).astype(int), (np.ceil(2.355*sigma_x)).astype(int)
        # Collapse the FWHM into a single value for each direction
        fwhm_y = func(fwhm_y)
        fwhm_x = func(fwhm_x)
        shape = (len(image), 2*fwhm_y+1, 2*fwhm_x+1)
        cy, cx = (np.array(image.shape[-2:])-1) // 2
        return Box(shape, origin=(0, cy-fwhm_y, cx-fwhm_x))

    @staticmethod
    def from_center(center, shape):
        """Generate a box given a shape and central position
        """
        if len(center) == 3:
            center = center[1:]
        if len(shape) == 2:
            shape = (0,) + tuple(shape)
        center = np.asarray(center)
        shape = np.asarray(shape)
        # The center must be an integer coordinate
        assert np.all([int(c) == c for c in center])
        # The shape must be odd in each dimnsion, otherwise
        # the center is not well defined
        assert np.all([s % 2 == 1 for s in shape[1:]])
        origin = (0,) + tuple((center - np.array(0.5*(shape[1:]-1))).astype(int))
        return Box(tuple(shape), origin=origin)

    def contains(self, p):
        """Whether the box contains a given coordinate `p`
        """
        assert len(p) == self.D

        for d in range(self.D):
            if p[d] < self.origin[d] or p[d] >= self.origin[d] + self.shape[d]:
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
        """Tuple of stop coordinates
        """
        return tuple(o + s for o, s in zip(self.origin, self.shape))

    @property
    def center(self):
        """Get the center of the Box
        """
        cmin = np.array(self.start)
        cmax = np.array(self.stop) - 1
        center = 0.5*(cmax-cmin)
        return tuple(center)

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
            bounds.append(
                (min(self.start[d], other.start[d]), max(self.stop[d], other.stop[d]))
            )
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
            bounds.append(
                (max(self.start[d], other.start[d]), min(self.stop[d], other.stop[d]))
            )
        return Box.from_bounds(*bounds)

    def overlaps(self, other):
        """Whether or not this `Box` overlaps with `other`
        """
        return np.all([self.start[k] < other.stop[k] for k in range(len(self.shape))])

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

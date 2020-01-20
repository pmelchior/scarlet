import numpy as np


def get_model(component):
    frame_ = component.frame
    component.set_frame(component.bbox)
    model = component.get_model()
    component.set_frame(frame_)
    return model


def max_pixel(component):
    """Determine pixel with maximum value

    Parameters
    ----------
    component: `scarlet.Component` or `scarlet.ComponentTree`
        Component to analyze
    """
    model = get_model(component)
    return tuple(
        np.unravel_index(np.argmax(model_), model.shape) + component.bbox.origin
    )


def flux(component):
    """Determine flux in every channel

    Parameters
    ----------
    component: `scarlet.Component` or `scarlet.ComponentTree`
        Component to analyze
    """
    model = get_model(component)
    return model.sum(axis=(1, 2))


def centroid(component):
    """Determine centroid of model

    Parameters
    ----------
    component: `scarlet.Component` or `scarlet.ComponentTree`
        Component to analyze
    """
    model = get_model(component)
    indices = np.indices(model.shape)
    centroid = np.array([np.sum(ind * model) for ind in indices]) / model.sum()
    return centroid + component.bbox.origin


class Moments:
    """Keep track of the various moments of an image

    Since deriving higher order moments often requires using lower
    order moments, this class caches moments as they are calculated
    and allows users to ask for any order moment that they wish.

    Attributes
    ----------
    n: int
        The highest order moment calculated for a given instance.
        Initially `n=-1`, since no moments have been calculated.
    """
    def __init__(self, image, origin=None):
        """Initialize the class

        Parameters
        ----------
        image: `~numpy.array`
            The 2D (y,x) or 3D (channels, y, x) image to calculate
            the moments of. If the image has a channel dimension,
            a given moment will be calculated for each channel.
        origin: tuple
            (y,x) position of the origin.
            If `origin` has 3 entries then the first dimension
            (the channel) is ignored.
        """
        dims = len(image.shape)
        self.dims = dims
        assert dims in [2, 3]
        self._moments = {}
        # Current highest moment calculated
        self.n = -1
        self.image = image
        if dims == 2:
            self.inv_image = 1/image.sum()
            self.C = None
        else:
            self.inv_image = 1/image.sum(axis=(1, 2))
            self.C = image.shape[0]

        # Set the spatial shape of the moments
        if dims == 2:
            shape = image.shape
        else:
            shape = image.shape[1:]

        # Make sure that an origin is defined
        if origin is None:
            origin = (np.array(shape)-1) // 2
        # Ignore the channels dimension if given
        if len(origin) == 3:
            origin = origin[1:]
        origin = np.asarray(origin)
        # Set the coordinates for each pixel location
        y = np.arange(shape[0], dtype=float)
        x = np.arange(shape[1], dtype=float)
        X, Y = np.array(np.meshgrid(x, y))
        Y -= origin[0]
        X -= origin[1]
        self.coords = Y, X
        self.moments = {}
        self.n = []

    def generate_moments(self, n):
        """Generate the `n`th order moment
        """
        Y, X = self.coords
        if self.dims == 3:
            # Use the same spaitial coordinate for each channel
            Y = np.repeat(Y[None, :, :], self.C, axis=0)
            X = np.repeat(X[None, :, :], self.C, axis=0)

        # Generate each moment for the given dimension
        for i in range(n+1):
            j = n - i
            if n == 1:
                if self.dims == 2:
                    self.moments[(i, j)] = (X**i * Y**j * self.image).sum() * self.inv_image
                else:
                    self.moments[(i, j)] = (X**i * Y**j * self.image).sum(axis=(1, 2)) * self.inv_image
            else:
                if self.dims == 2:
                    self.moments[(i, j)] = (
                        (X-self[(1, 0)])**i *
                        (Y-self[(0, 1)])**j * self.image).sum() * self.inv_image
                else:
                    self.moments[(i, j)] = (
                        (X-self[(1, 0)][:, None, None])**i *
                        (Y-self[(0, 1)][:, None, None])**j *
                        self.image).sum(axis=(1, 2)) * self.inv_image
        self.n.append(n)

    def __getitem__(self, index):
        """Get the (x, y) moment
        """
        assert len(index) == 2
        try:
            moment = self.moments[index]
        except KeyError:
            n = index[0] + index[1]
            self.generate_moments(n)
            moment = self.moments[index]
        return moment


def get_ellipse_params(moments):
    """Calculate the elliptical parameters of a given set of moments

    Parameters
    ----------
    moments: `Moment`
        The moments to calculate the elliptical parameters from.

    Returns
    -------
    a: float
        Semi-major axis
    b: float
        Semi-minor axis
    theta: float
        Angle (in radians) that the ellipse makes with the x-axis.
        This will always range from `-np.pi/2` to `np.pi/2`.
    """
    x2 = moments[2, 0]
    y2 = moments[0, 2]
    xy = moments[1, 1]

    mu_sum = x2 + y2
    mu_diff = x2 - y2
    mu_diff2 = mu_diff**2

    a = np.sqrt(0.5*mu_sum + np.sqrt(xy**2 + 0.25*mu_diff2))
    b = np.sqrt(0.5*mu_sum - np.sqrt(xy**2 + 0.25*mu_diff2))

    if moments.dims == 2:
        # Insert theta=pi/4 by hand
        if mu_diff == 0:
            theta = np.pi/4
        else:
            theta = 0.5 * np.arctan2(2*xy, mu_diff)
        # Fix the sign (if necessary)
        if np.sign(theta) == np.sign(xy):
            theta = -theta
    else:
        # Insert theta=pi/4 by hand
        theta = np.zeros(moments.C)
        theta[mu_diff == 0] = np.pi/4
        cuts = mu_diff != 0
        theta[cuts] = 0.5*np.arctan2(2*xy[cuts], mu_diff[cuts])
        # Fix the sign (if necessary)
        cuts = np.sign(theta) == np.sign(xy)
        theta[cuts] = -theta
    return a, b, theta

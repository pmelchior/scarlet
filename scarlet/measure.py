import numpy as np


def max_pixel(component):
    """Determine pixel with maximum value

    Parameters
    ----------
    component: `scarlet.Component` or `scarlet.ComponentTree`
        Component to analyze
    """
    model = component.get_model()
    return tuple(
        np.unravel_index(np.argmax(model), model.shape) + component.bbox.origin
    )


def flux(component):
    """Determine flux in every channel

    Parameters
    ----------
    component: `scarlet.Component` or `scarlet.ComponentTree`
        Component to analyze
    """
    model = component.get_model()
    return model.sum(axis=(1, 2))


def centroid(component):
    """Determine centroid of model

    Parameters
    ----------
    component: `scarlet.Component` or `scarlet.ComponentTree`
        Component to analyze
    """
    model = component.get_model()
    indices = np.indices(model.shape)
    centroid = np.array([np.sum(ind * model) for ind in indices]) / model.sum()
    return centroid + component.bbox.origin


class Moments:
    """Generate Moments for an image
    """
    def __init__(self, image, center=None):
        self._moments = {}
        self.n = -1
        self.image = image
        self.inv_image = 1/image.sum()

        y = np.arange(image.shape[0], dtype=float)
        x = np.arange(image.shape[1], dtype=float)

        X, Y = np.array(np.meshgrid(x, y))
        if center is None:
            cy, cx = (np.array(image.shape)-1) // 2
            center = np.array([cy, cx])
        Y -= center[0]
        X -= center[1]
        self.coords = Y, X
        self.moments = {}
        self.n = []

    def generate_moments(self, n):
        """Generate all of the nth order moments
        """
        Y, X = self.coords

        for i in range(n+1):
            j = n - i
            if n == 1:
                self.moments[(i, j)] = (X**i * Y**j * self.image).sum() * self.inv_image
            else:
                # Use the central moment for higher order moments
                _Y, _X = Y-self[(0, 1)], X-self[(1, 0)]
                self.moments[(i, j)] = (_X**i * _Y**j * self.image).sum() * self.inv_image
        self.n.append(n)

    def __getitem__(self, index):
        """Fetch a given moment
        `index=(i, j)` will return the `X**i`, `Y**j` moment,
        generating it if necessary.
        """
        assert len(index) == 2
        try:
            moment = self.moments[index]
        except KeyError:
            n = index[0] + index[1]
            self.generate_moments(n)
            moment = self.moments[index]
        return moment


class DeconvolvedMoment:
    """Use the DEIMOS algorithm to deconvolve the moments
    Using the DEIMOS algorithm from Melchior et al. 2012
    (https://arxiv.org/abs/1008.1076)
    deconvolve the moments based on the PSF of the image.
    """
    def __init__(self, moments, psf_moments):
        """Initialize the class
        Parameters
        ----------
        moments: `Moment`
            The moments of the image to deconvolve.
        psf_moments: `Moment`
            The moments of the PSF that the image is convolved with.
        """
        self.moments = moments
        self.psf = psf_moments
        self.deconvolved_moments = {}

    @property
    def coords(self):
        """The pixel coordinates for each source.
        `coords = (Y, X)`, where `X` and `Y` are the same
        shape as the input immage and are the coordinates
        of the `X` and `Y` positions respectively.
        """
        return self.moments.coords

    def __getitem__(self, index):
        """Fetch a given moment
        Similar to the `Moment` class, the `index=(i, j)`
        moment is generated (if necessary) and returned.
        """
        assert len(index) == 2
        try:
            moment = self.deconvolved_moments[index]
        except KeyError:
            i, j = index
            moment = self.moments[index]
            for k in range(i+1):
                for l in range(j+1):
                    if i != k or j != l:
                        coeff = -binomial_coeff(i, k) * binomial_coeff(j, l)
                        moment -= coeff * self[k, l] * self.psf[i-k, j-l]
            self.deconvolved_moments[index] = moment
        return moment


def binomial_coeff(n, m):
    """Return the binomial coefficent of n choose m
    """
    f = np.math.factorial
    return f(n)/(f(m) * f(n-m))


def get_ellipse_params(moments):
    """Calculate ellipse parameters from a given set of moments
    Parameters
    ----------
    moments: `Moment`
        The moments of the elliptical source
    Returns
    -------
    a: `float`
        Standard deviation of the Semi-major axis.
    b: `float`
        Standard deviation of the Semi-minor axis.
    theta: `float`
        Angle the ellipse makes with the x-axis
        (rotated counter-clockwise).
    """
    x2 = moments[2, 0]
    y2 = moments[0, 2]
    xy = moments[1, 1]

    mu_sum = x2 + y2
    mu_diff = x2 - y2
    mu_diff2 = mu_diff**2

    a = np.sqrt(0.5*mu_sum + np.sqrt(xy**2 + 0.25*mu_diff2))
    b = np.sqrt(0.5*mu_sum - np.sqrt(xy**2 + 0.25*mu_diff2))
    # Prevent division by zero when theta is 45 degrees
    if mu_diff == 0:
        theta = np.pi/4
    else:
        theta = 0.5 * np.arctan2(2*xy, mu_diff)
    # Make sure that the angle has the correct sign
    if np.sign(theta) == np.sign(xy):
        theta = -theta
    return theta, a, b
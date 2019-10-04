import numpy as np
from scipy.special import gamma, gammainc
from scipy.optimize import curve_fit


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


def gaussian2D(y, x, theta, sigma_x, sigma_y, amplitude=1):
    """Sample a 2D elliptical gaussian
    """
    cos2 = np.cos(theta)**2
    sin2 = np.sin(theta)**2
    sin2t = np.sin(2*theta)
    sx2 = sigma_x**2
    sy2 = sigma_y**2
    a = 0.5*(cos2/sx2 + sin2/sy2)
    b = 0.25*(-sin2t/sx2 + sin2t/sy2)
    c = 0.5*(sin2/sx2 + cos2/sy2)
    return amplitude * np.exp(-(a*x**2 + 2*b*x*y + c*y**2))


def gaussian_mixture(Y, X, *params):
    """Create a model as the sum of gaussians

    `params` should be a list of parameters of the form
    `params=[theta, sigma_x1, sigma_y1, amplitude1, sigma_x2, ...]`
    where each new gaussian component has a
    `sigma_x`, `sigma_y`, `amplitude` parameter specified.
    """
    assert len(params[1:]) % 3 == 0, "{0}".format(params)
    theta = params[0]
    K = len(params[1:]) // 3
    model = np.sum([
        gaussian2D(Y, X, theta, params[3*k+1], params[3*k+2], params[3*k+3]) for k in range(K)
    ], axis=0)
    return model


def gaussian_loss2D(coords, *params):
    """Loss function for a gaussian mixture model.

    This function just generates the model and flattens it
    so that `scipy.curve_fit` can use it.
    """
    y, x = coords
    model = gaussian_mixture(y, x, *params)
    return model.reshape(-1)


def _get_coords(shape, center=None):
    if center is None:
        cy, cx = (np.array(shape) - 1) // 2
        X = np.linspace(-cx, cx, shape[1])
        Y = np.linspace(-cy, cy, shape[0])
    else:
        X = np.arange(shape[1]) - center[1]
        Y = np.arange(shape[0]) - center[0]
    X, Y = np.meshgrid(X, Y)
    return (Y, X)


def fit_gaussian_mixture(image, center=None, components=2, init_params=None, bounds=None):
    """Fit an image as a mixture of gaussians
    """
    if init_params is None:
        init_params = [((n+1), (n+1), image.max()/(n+1)) for n in range(components)]
        init_params = [0] + [item for sublist in init_params for item in sublist]
    if bounds is None:
        xmins = [-np.pi/2] + [0] * len(init_params[1:])
        xmaxs = [np.pi/2] + [np.inf] * len(init_params[1:])
        bounds = [xmins, xmaxs]
    coords = _get_coords(image.shape, center)
    result = curve_fit(gaussian_loss2D, coords, image.reshape(-1), init_params, bounds=bounds)
    return result


def get_model(shape, func, *args, center=None, **kwargs):
    Y, X = _get_coords(shape, center)
    return func(Y, X, *args, **kwargs)


def sersic_hl(r, n, dr, intensity, half_light_radius):
    """Return a sersic model based on half-light radius
    """
    Ie = intensity
    Re = half_light_radius
    bn = get_bn(n)
    return Ie * np.exp(-bn * ((r/Re)**(1/n)))


def sersic(r, n, dr, R0, I0, center=None):
    """Return a sersic radial profile

    Since we know the flux at the center of the galaxy,
    use the sersic profile:
        I(r) = F/(2n\pi \Gamma(2n) r_0^2) e^{-r/r_0}^{1/n}

    See Rowe et al. 2015 (https://arxiv.org/pdf/1407.7676.pdf)
    for a description of the parameters.
    """
    # Normalization parameter
    norm = 1/(2*n*np.pi*gamma(2*n)*R0**2)
    # The integrated flux up to the first r>0
    # is given by the incomplete gamma function
    # \gamma(2n, (dr/R0)**(1/n)). Because the
    # Sersic model has a strong peak at r=0 we
    # set it by hand to the above flux, but
    # in order to scale the overall amplitude
    # we multiply the rest of the terms by I0/gamma.
    dr = 0.56 * dr
    norm = norm * I0 / gammainc(2*n, (dr/R0)**(1/n))
    result = norm * np.exp(-(r/R0)**(1/n))
    # Set the central pixel by hand, if an r==0 term exists
    if center is not None:
        result[center] = I0
    return result


def exponential(r, intensity, r_scale):
    """Return an exponential radial profile
    """
    I0 = intensity
    return I0 * np.exp(-r/r_scale)


def get_bn(n):
    """From SExtractor docs

    https://sextractor.readthedocs.io/en/latest/Model.html
    """
    # print(2*n - 1/3 +4/(405*n) + 46/(25515*n**2) + 131/(114175*n**3))
    return 2*n - 1/3 + 4/(405*n) + 46/(25515*n**2) + 131/(114175*n**3)


def _get_model(coords, func, q=1, theta=0, scale=1, **kwargs):
    y, x = coords
    a = 1
    b = 1/q
    r = np.sqrt(((x*np.cos(theta) + y*np.sin(theta))/a)**2 + ((x*np.sin(theta) - y*np.cos(theta))/b)**2)
    img = func(r, **kwargs)
    return img


def get_surface_brightness(shape, func, q=1, theta=0, scale=1, **kwargs):
    y_radius = (shape[0]-1) // 2
    x_radius = (shape[1]-1) // 2
    y = np.linspace(-y_radius*scale, y_radius*scale, shape[0])
    x = np.linspace(-x_radius*scale, x_radius*scale, shape[1])
    coords = np.array(np.meshgrid(x, y, indexing="ij"))
    return get_model(coords, func, q, theta, scale, **kwargs)


def sersic_loss(coords, *params):
    model = get_model(coords, *params)
    return model.reshape(-1)

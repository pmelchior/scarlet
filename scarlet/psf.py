import autograd.numpy as np
import autograd.scipy as scipy
from .bbox import Box


def moffat(y, x, alpha=4.7, beta=1.5, bbox=None):
    """Symmetric 2D Moffat function

    .. math::

        (1+\frac{(x-x0)^2+(y-y0)^2}{\alpha^2})^{-\beta}

    Parameters
    ----------
    y: float
        Vertical coordinate of the center
    x: float
        Horizontal coordinate of the center
    alpha: float
        Core width
    beta: float
        Power-law index
    bbox: Box
        Bounding box over which to evaluate the function

    Returns
    -------
    result: array
        A 2D circular gaussian sampled at the coordinates `(y_i, x_j)`
        for all i and j in `shape`.
    """
    Y = np.arange(bbox.shape[1]) + bbox.origin[1]
    X = np.arange(bbox.shape[2]) + bbox.origin[2]
    X, Y = np.meshgrid(X, Y)
    # TODO: has no pixel-integration formula
    return ((1 + ((X - x) ** 2 + (Y - y) ** 2) / alpha ** 2) ** -beta)[None, :, :]


def gaussian(y, x, sigma=1, integrate=True, bbox=None):
    """Circular Gaussian Function

    Parameters
    ----------
    y: float
        Vertical coordinate of the center
    x: float
        Horizontal coordinate of the center
    sigma: float
        Standard deviation of the gaussian
    integrate: bool
        Whether pixel integration is performed
    bbox: Box
        Bounding box over which to evaluate the function

    Returns
    -------
    result: array
        A 2D circular gaussian sampled at the coordinates `(y_i, x_j)`
        for all i and j in `shape`.
    """
    Y = np.arange(bbox.shape[1]) + bbox.origin[1]
    X = np.arange(bbox.shape[2]) + bbox.origin[2]

    def f(X):
        if not integrate:
            return np.exp(-(X ** 2) / (2 * sigma ** 2))
        else:
            sqrt2 = np.sqrt(2)
            return (
                np.sqrt(np.pi / 2)
                * sigma
                * (
                    scipy.special.erf((0.5 - X) / (sqrt2 * sigma))
                    + scipy.special.erf((2 * X + 1) / (2 * sqrt2 * sigma))
                )
            )

    return (f(Y - y)[:, None] * f(X - x)[None, :])[None, :, :]


class PSF:
    """Class to represent PSFs

    Parameters
    ----------
    X: array-like or method
        If `X` is an array, it represent an image of the PSF in every band.
        If `X` is a callable method, it describes a function that can generate a PSF when given coordinates (y, x).
    shape: tuple
        Shape of the 2D image to generate an PSF image for. Only used if `X` is a method.
    """

    def __init__(self, X, shape=None):
        if hasattr(X, "shape"):
            self._image = X.copy()
            self.normalize()
            self._func = None
            self.shape = X.shape
        elif hasattr(X, "__call__"):
            assert shape is not None, "Functional PSFs must set shape argument"
            self._image = None
            self._func = X
            self.shape = shape
        else:
            msg = "A PSF must be initialized with either an image or function"
            raise ValueError(msg)

    def __call__(self, y, x, bbox=None):
        """Generate analytic PSF image at coordinate (y,x)

        Parameters
        ----------
        y: float
            Vertical model frame coordinates for the center of PSF image
        x: float
            Horizontal model frame coordinates for the center of PSF image
        bbox: `~scarlet.Box`
            Bounding Box for the PSF image in model coordinates

        Returns
        -------
        image: array-like
            Centered image of the PSF at given coordinate.
        """
        if bbox is None:
            bbox = Box(self.shape)
        if self._func is not None:
            return self._func(y, x, bbox=bbox)
        return None

    @property
    def image(self):
        """Centered image of the PSF
        """
        if self._image is None:
            assert self.shape is not None, "Set PSF.shape first"
            y, x = self.shape[1] // 2, self.shape[2] // 2
            self._image = self.__call__(y, x)
            self.normalize()
        return self._image

    def normalize(self):
        """Normalize to PSF image in every band to unity
        """
        sums = self._image.sum(axis=(1, 2))
        self._image /= sums[:, None, None]
        return self

    def update_dtype(self, dtype):
        """Update data type of `image` to `dtype`
        """
        if self.image.dtype != dtype:
            self._image = self._image.astype(dtype)
        return self


def build_diff_images(image, box=None):
    """Build a differnce image

    In order to fit images it might be necessary to shift them
    by a fractional position. This method is used to build
    a difference kernel so that the new image is given by
    `image + dx*diff_x + dy+diff_y` for some floats `dx`, `dy`,
    where `dx`, `dy` are between -1 and 1.

    Parameters
    ----------
    image: `~numpy.array`
        3D image to shift.
    box: `~scarlet.bbox.Box`
        Box used to extract part of the image.

    Returns
    -------
    diff_x: 3D array
        The difference image in each channel for shifts in the x-direction.
    diff_y: 3D array
        The difference image in each channel for shifts in the y-direction.
    """
    diff_y = np.zeros(image.shape)
    diff_x = np.zeros(image.shape)

    diff_y[:, 1:-1] = 0.5*(image[:, :-2]-image[:, 2:])
    diff_x[:, :, 1:-1] = 0.5*(image[:, :, :-2]-image[:, :, 2:])

    if box is not None:
        diff_y = box.extract_from(diff_y)
        diff_x = box.extract_from(diff_x)
    return diff_y, diff_x


def match_sources(images, psfs, centers, min_overlap=1e-4, show=False):
    """Match each source in an image to the PSF in each band

    Parameters
    ----------
    images: `~numpy.array`
        3D observed image
    psfs: `~scarlet.psf.PSF`
        PSF object containing the PSF in each band
        (or a function that can be used to generate it at
        a given position).
    centers: list
        List of center locations for each source in `images`.
    min_chi2: float
        Minimum chi^2 required to label a source as a PSF.
    min_overlap: float
        Minimum amount of overlap a PSF must have in a band
        to be considered overlapping.
    show: bool
        Whether or not to show plots of the matching,
        which can be a useful diagnostic tool.
    """
    psf_shape = psfs.shape
    # Since we might add/remove some of these model fitting parameters
    # later, it is useful to keep track of the index for each parameter
    # that we fit.
    indices = ["sky", "sky_x", "sky_y", "dx", "dy", "psf"]
    i_sky = indices.index("sky")
    i_sky_x = indices.index("sky_x")
    i_sky_y = indices.index("sky_y")
    i_dx = indices.index("dx")
    i_dy = indices.index("dy")
    i_psf = indices.index("psf")
    # Total number of parameters other than PSFs from other sources
    n_base = len(indices)

    # Set the bounding box for the full PSF in each band
    psf_bbox = Box.from_moments(psfs.image)
    psf_center = np.asarray(psf_bbox.center)

    all_chi2 = np.zeros((len(centers), len(images)))
    # Fit each source
    for k, center in enumerate(centers):
        cy, cx = center
        # Create a stamp centered on the primary source in the image
        _center = np.asarray((psf_center[0],)+tuple(center))
        origin = tuple((_center-psf_center).astype(int))
        stamp_box = Box(psf_bbox.shape, origin=origin)
        stamps = stamp_box.extract_from(images)
        diff_y, diff_x = build_diff_images(psfs.image, psf_bbox)

        # Test all of the other sources for overlap and
        # (if necessary) add the portion of their PSF that
        # overlaps with the primary source to the model.
        others = []
        for m, other in enumerate(centers):
            if m == k:
                continue
            _box = Box.from_center(other, psf_shape)
            # Only add the PSF if there is an overlap
            if stamp_box.overlaps(_box):
                img = np.zeros(images.shape)
                img = _box.insert_into(img, psfs.image)
                other = stamp_box.extract_from(img)
                # Make sure that the overlap is significant
                if np.sum(other) > min_overlap:
                    others.append(other)

                    if show:
                        import matplotlib.pyplot as plt
                        plt.imshow(img[0])
                        plt.title("other {0}".format(m))
                        plt.show()

                        plt.imshow(others[-1][0])
                        plt.title("other {0} stamp".format(m))
                        plt.show()

        # Coordinates used to calcualte the linear sky background
        X = np.arange(stamp_box.start[-1], stamp_box.stop[-1]) - cx
        Y = np.arange(stamp_box.start[-2], stamp_box.stop[-2]) - cy
        X, Y = np.meshgrid(X, Y)
        x = X.reshape(-1)
        y = Y.reshape(-1)
        for c, stamp in enumerate(stamps):
            A = np.zeros((stamp.size, n_base+len(others)))
            # Constant background
            A[:, i_sky] = 1
            # Linear ramp on x-axis
            A[:, i_sky_x] = x
            A[:, i_sky_y] = y
            # Positional offset * Amplitude
            A[:, i_dx] = diff_x[c].reshape(-1)
            A[:, i_dy] = diff_y[c].reshape(-1)
            # Amplitude of the PSF
            _psf_bbox = Box(psf_bbox.shape[1:], psf_bbox.origin[1:])
            A[:, i_psf] = _psf_bbox.extract_from(psfs.image[c]).reshape(-1)

            # Include any overlaping PSFs from neighboring sources
            for n, other in enumerate(others):
                A[:, i_psf+1+n] = other[c].reshape(-1)
            result, residuals, rank, singular = np.linalg.lstsq(A, stamp.reshape(-1), rcond=None)
            all_chi2[k, c] = residuals[0]

            if show:
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots(figsize=(15, 5))
                model = np.zeros(stamp.shape)
                for col in range(A.shape[1]):
                    model += A[:, col].reshape(stamp.shape) * result[col]
                diff = stamp-model

                ax[0].imshow(stamp)
                ax[0].set_title("stamp")
                ax[1].imshow(model)
                ax[1].set_title("model")
                im = ax[2].imshow(diff)
                fig.colorbar(im, ax=ax[2])
                ax[2].set_title("residual")
                plt.show()
    return all_chi2

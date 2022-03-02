import autograd.numpy as np
import numpy.ma as ma
import proxmin.operators

from .bbox import Box, overlapped_slices
from .constraint import (
    ConstraintChain,
    L0Constraint,
    PositivityConstraint,
    MonotonicityConstraint,
    MonotonicMaskConstraint,
    SymmetryConstraint,
    CenterOnConstraint,
    NormalizationConstraint,
)
from .frame import Frame
from .model import Model, UpdateException
from .parameter import Parameter, prepare_param, relative_step
from .psf import PSF
from .wavelet import Starlet, starlet_reconstruction, get_multiresolution_support
from . import fft
from . import initialization


class Morphology(Model):
    """Morphology base class

    The class describes the 2D image of the spatial dependence of
    `~scarlet.FactorizedComponent`.

    Parameters
    ----------
    frame: `~scarlet.Frame`
        Characterization of the model
    parameters: list of `~scarlet.Parameter`
    bbox: `~scarlet.Box`
        2D bounding box of this model
    """

    def __init__(self, frame, *parameters, bbox=None):
        assert isinstance(frame, Frame)
        self.frame = frame

        if bbox is None:
            bbox = frame.bbox
        assert isinstance(bbox, Box)
        self.bbox = bbox

        super().__init__(*parameters)

    def shrink_box(self, image, thresh=0):
        # peel the onion
        size = max(image.shape)
        dist = 0
        while (
            np.all(image[dist, :] <= thresh)
            and np.all(image[-dist - 1, :] <= thresh)
            and np.all(image[:, dist] <= thresh)
            and np.all(image[:, -dist - 1] <= thresh)
        ):
            dist += 1
        newsize = initialization.get_minimal_boxsize(size - 2 * dist)
        if newsize < size:
            dist = (size - newsize) // 2
            # adjust bbox
            self.bbox.origin = tuple(o + dist for o in self.bbox.origin)
            self.bbox.shape = (newsize, newsize)


class ImageMorphology(Morphology):
    """Morphology from an image

    The class uses an arbitrary image as non-parametric model. To allow for subpixel
    offsets, a Fourier-based shifting transformation is available.

    Parameters
    ----------
    frame: `~scarlet.Frame`
        Characterization of the model
    image: 2D array or `~scarlet.Parameter`
        Image parameter
    bbox: `~scarlet.Box`
        2D bounding box for focation of the image in `frame`
    shift: None or `~scarlet.Parameter`
        2D shift parameter (in units of image pixels)
    resizing: bool
        Whether to resize the box dynamically
    """

    def __init__(
        self, frame, image, bbox=None, shifting=False, shift=None, resizing=True
    ):
        if isinstance(image, Parameter):
            assert image.name == "image"
        else:
            constraint = PositivityConstraint()
            image = Parameter(
                image, name="image", step=relative_step, constraint=constraint
            )

        if bbox is None:
            assert frame.bbox[1:].shape == image.shape
            bbox = Box(image.shape)
        else:
            assert bbox.shape == image.shape

        self.resizing = resizing
        self.shifting = shifting

        # create the shift parameter to allow for dynamic resizing
        if shift is None:
            shift = Parameter(np.zeros(2), name="shift", step=1e-2, fixed=self.shifting)
        else:
            assert shift.shape == (2,)
            if isinstance(shift, Parameter):
                assert shift.name == "shift"
            else:
                shift = Parameter(shift, name="shift", step=1e-2)

        parameters = (image, shift)
        super().__init__(frame, *parameters, bbox=bbox)

    def get_model(self, *parameters):
        image = self.get_parameter(0, *parameters)
        shift = self.get_parameter(1, *parameters)

        if self.shifting:
            image = fft.shift(image, shift, return_Fourier=False)
        return image

    def update(self):
        image = self._parameters[0]

        if not self.resizing or image.fixed:
            return

        # shrink the box?
        bbox = self.bbox.copy()
        self.shrink_box(image)
        if bbox != self.bbox:
            slice, _ = overlapped_slices(bbox, self.bbox)
            image = Parameter(
                image[slice],
                name=image.name,
                prior=image.prior,
                constraint=image.constraint,
                step=image.step / 2,
                fixed=image.fixed,
                m=image.m[slice] if image.m is not None else None,
                v=image.v[slice] if image.v is not None else None,
                vhat=image.vhat[slice] if image.vhat is not None else None,
            )

            # set new parameters
            self._parameters = (image,) + self._parameters[1:]

            raise UpdateException

        # grow the box?
        # because the PSF moves power across the box, the gradients at the edge
        # accummulate flux from beyond the box
        elif image.m is not None:
            # next adam gradient update
            gu = -image.m / np.sqrt(np.sqrt(ma.masked_equal(image.v, 0))) * image.step
            gu_pull = gu * (image > 0)  # check if model has flux at the edge at all
            edge_pull = np.array(
                (
                    gu_pull[:, 0].mean(),
                    gu_pull[:, -1].mean(),
                    gu_pull[0, :].mean(),
                    gu_pull[-1, :].mean(),
                )
            )

            # 0.1 compared to 1 at center
            if np.any(edge_pull > 0.1):
                # find next larger boxsize
                size = max(bbox.shape)
                newsize = initialization.get_minimal_boxsize(size + 1)
                pad_width = (newsize - size) // 2

                # Create new parameter for extended image
                image = Parameter(
                    np.pad(image, pad_width, mode="linear_ramp"),
                    name=image.name,
                    prior=image.prior,
                    constraint=image.constraint,
                    step=image.step / 2,
                    fixed=image.fixed,
                    m=np.pad(image.m, pad_width, mode="constant")
                    if image.m is not None
                    else None,
                    v=np.pad(image.v, pad_width, mode="constant")
                    if image.v is not None
                    else None,
                    vhat=np.pad(image.vhat, pad_width, mode="constant")
                    if image.vhat is not None
                    else None,
                )
                # set new parameters
                self._parameters = (image,) + self._parameters[1:]

                # adjust bbox
                self.bbox.origin = tuple(o - pad_width for o in self.bbox.origin)
                self.bbox.shape = (newsize, newsize)
                raise UpdateException


class ProfileMorphology(Morphology):
    """Morphology from a radial profile

    Parameters
    ----------
    frame: `~scarlet.Frame`
        Characterization of the model
    func:
        Radial profile function, signature func(R2, *parameters),
        where R2 is the squared radius.
    parameters: list of `~scarlet.Parameter`
        Parameters for the radial profile. Needs to include at least the following:
        * "radius": float
        * "center": 2D float
        * "ellipticity": 2D float
    boxsize: int
        Size of bounding box over which to evaluate the function, in frame pixels
    """

    def __init__(self, frame, func, *parameters, boxsize=None):

        if boxsize is None:
            boxsize = 15
        if boxsize % 2 == 0:
            boxsize += 1
        shape = (boxsize, boxsize)

        center = self.get_parameter("center", *parameters)
        assert center is not None and len(center) >= 2
        # define bbox around center
        origin = (
            int(round(center[-2])) - (boxsize // 2),
            int(round(center[-1])) - (boxsize // 2),
        )
        bbox = Box(shape, origin=origin)

        # retain center attribute
        self.center = center

        self._Y = np.arange(shape[-2], dtype="float") + origin[-2]
        self._X = np.arange(shape[-1], dtype="float") + origin[-1]

        self.f = func

        # make sure ellipticity stays in unit circle
        eps = self.get_parameter("ellipticity", *parameters)
        eps.prox = self._eps_prox

        super().__init__(frame, *parameters, bbox=bbox)

    def get_model(self, *parameters):
        center = self.get_parameter("center", *parameters)
        _Y = self._Y - center[-2]
        _X = self._X - center[-1]

        e = self.get_parameter("ellipticity", *parameters)
        if np.all(e == 0) and not parameters:  # need e for gradients even if 0
            R2 = _Y[:, None] ** 2 + _X[None, :] ** 2
        else:
            e1, e2 = e[0], e[1]
            __X = ((1 - e1) * _X[None, :] - e2 * _Y[:, None]) / np.sqrt(
                1 - (e1 ** 2 + e2 ** 2)
            )
            __Y = (-e2 * _X[None, :] + (1 + e1) * _Y[:, None]) / np.sqrt(
                1 - (e1 ** 2 + e2 ** 2)
            )
            R2 = __Y ** 2 + __X ** 2

        Rp = self.get_parameter("radius", *parameters)
        R2 /= Rp ** 2

        morph = self.f(R2, *parameters)
        morph /= morph.sum()

        return morph

    def _eps_prox(self, x, step):
        norm2 = (x ** 2).sum()
        if norm2 > 1:
            x /= np.sqrt(norm2)
        return x


class GaussianMorphology(ProfileMorphology):
    """Morphology from a Gaussian radial profile

    Parameters
    ----------
    frame: `~scarlet.Frame`
        Characterization of the model
    center: array or `~scarlet.Parameter`
        2D center parameter (in units of frame pixels)
        If it is to be optimized, it needs to be provided as a full defined `Parameter`.
    sigma: float or `~scarlet.Parameter`
        Standard deviation of the Gaussian in frame pixels
        If it is to be optimized, it needs to be provided as a full defined `Parameter`.
    ellipticity: array or ~scarlet.Parameter`
        Two-component ellipticity (e1,e2).
        If it is to be optimized, it needs to be provided as a full defined `Parameter`.
    boxsize: int
        Size of bounding box over which to evaluate the function, in frame pixels
    """

    def __init__(self, frame, center, sigma, ellipticity=(0, 0), boxsize=None):

        assert len(center) == 2
        self.center = prepare_param(center, name="center")
        radius = prepare_param(sigma, name="radius")
        assert len(ellipticity) == 2
        ellipticity = prepare_param(ellipticity, name="ellipticity")
        parameters = (self.center, radius, ellipticity)

        if boxsize is None:
            boxsize = int(np.ceil(10 * sigma))

        super().__init__(frame, self._f, *parameters, boxsize=boxsize)

    def _f(self, R2, *parameters):
        return np.exp(-R2 / 2)


# we need gamma and kv from scipy.special
from autograd.scipy.special import gamma
from scipy.optimize import fsolve

# but kv is not ported to autograd ...
import scipy.special
from autograd.extend import primitive, defvjp

kv = primitive(scipy.special.kv)
defvjp(kv, None, lambda ans, n, x: lambda g: g * (-kv(n - 1, x) - kv(n + 1, x)) / 2.0)


class SpergelMorphology(ProfileMorphology):
    """Morphology from a Spergel (2010) radial profile

    Parameters
    ----------
    frame: `~scarlet.Frame`
        Characterization of the model
    center: array or `~scarlet.Parameter`
        2D center parameter (in units of frame pixels)
        If it is to be optimized, it needs to be provided as a full defined `Parameter`.
    nu: float or `~scarlet.Parameter`
        Bessel function order.
        If it is to be optimized, it needs to be provided as a full defined `Parameter`.
    rhalf: float or `~scarlet.Parameter`
        Half-light radius in frame pixels.
        If it is to be optimized, it needs to be provided as a full defined `Parameter`.
    ellipticity: array or None
        Two-component ellipticity (e1,e2)
        If it is to be optimized, it needs to be provided as a full defined `Parameter`.
    integrate: bool
        Whether pixel integration is performed
    boxsize: int
        Size of bounding box over which to evaluate the function, in frame pixels
    """

    def __init__(self, frame, center, nu, rhalf, ellipticity=(0, 0), boxsize=None):

        assert len(center) == 2
        self.center = prepare_param(center, name="center")

        self._minimum_nu = -0.85
        self._maximum_nu = 4.0
        assert nu >= self._minimum_nu and nu <= self._maximum_nu
        nu = prepare_param(nu, name="nu")
        # make sure nu stays within limits
        nu.prox = self._nu_prox

        radius = prepare_param(rhalf, name="radius")

        assert len(ellipticity) == 2
        ellipticity = prepare_param(ellipticity, name="ellipticity")
        parameters = (self.center, nu, radius, ellipticity)

        if boxsize is None:
            boxsize = int(np.ceil(10 * rhalf))

        # compute the cnu function
        # see Table 1 in Spergel (2010) and the sentence below Eqn (8).
        # _nu = np.linspace(self._minimum_nu,
        #                   self._maximum_nu,
        #                   1000)
        #
        # def func(x, nu):
        #     return (1 + nu) * self._f_nu(x, nu + 1) - 0.25
        #
        # _cnu = [fsolve(lambda x: func(x, v), x0=0.2)[0] for v in _nu]
        # self._z = np.polyfit(_nu, _cnu, 4)  # fit a 4th order polynomial
        # instead of solving this every time, we simply store the result
        self._z = np.array(
            [-0.00788962, 0.0735303, -0.27770785, 0.99483285, 1.25227402]
        )

        super().__init__(frame, self._f, *parameters, boxsize=boxsize)

    def _f(self, R2, *parameters):
        nu = self.get_parameter("nu", *parameters)
        cnu = self._cnu(nu)

        x = np.maximum(np.sqrt(R2) * cnu, 1e-2)
        return self._f_nu(x, nu)

    def _f_nu(self, x, nu):
        # Eqn 3 in Spergel (2010).
        # kv is the modified Bessel function of the second kind.
        # gamma is the gamma function.
        return (x / 2) ** nu * kv(nu, x) / gamma(nu + 1)

    def _cnu(self, nu):
        z = self._z
        return z[0] * nu ** 4 + z[1] * nu ** 3 + z[2] * nu ** 2 + z[3] * nu + z[4]

    def _nu_prox(self, x, step):
        return max(min(self._maximum_nu, x), self._minimum_nu)


class PointSourceMorphology(Morphology):
    """Morphology from a PSF

    The class uses `frame.psf` as model, evaluated at `center`

    Parameters
    ----------
    frame: `~scarlet.Frame`
        Characterization of the model
    center: array or `~scarlet.Parameter`
        2D center parameter (in units of frame pixels)
    """

    def __init__(self, frame, center):

        assert frame.psf is not None and isinstance(frame.psf, PSF)
        self.psf = frame.psf

        # define bbox
        pixel_center = tuple(np.round(center).astype("int"))
        shift = (0, *pixel_center)
        bbox = self.psf.bbox + shift

        # parameters is simply 2D center
        self.center = prepare_param(center, name="center")
        super().__init__(frame, self.center, bbox=bbox)

    def get_model(self, *parameters):
        center = self.get_parameter(0, *parameters)
        box_center = np.mean(self.bbox.bounds[1:], axis=1)
        offset = center - box_center
        return self.psf.get_model(offset=offset)  # no "internal" PSF parameters here


class StarletMorphology(Morphology):
    """Morphology from a starlet representation of an image

    The class uses the starlet parameterization as an overcomplete, non-parametric model.

    Parameters
    ----------
    frame: `~scarlet.Frame`
        Characterization of the model
    image: 2D array
        Initial image to construct starlet transform
    bbox: `~scarlet.Box`
        2D bounding box for focation of the image in `frame`
    monotonic: bool
        Whether to constrain every starlet scale to be monotonic; otherwise they are
        hard-thresholded by `threshold`.
    threshold: float
        Lower bound on threshold for all but the last starlet scale
    """

    def __init__(self, frame, image, bbox=None, monotonic=False, threshold=0):

        if bbox is None:
            assert frame.bbox[1:].shape == image.shape
            bbox = Box(image.shape)

        self.monotonic = monotonic

        # Starlet transform of morphologies (n1,n2) with 3 dimensions: (scales+1,n1,n2)
        self.transform = Starlet.from_image(image)
        # The starlet transform is the model
        coeffs = self.transform.coefficients

        if not self.monotonic:
            # wavelet-scale norm
            starlet_norm = self.transform.norm
            # One threshold per wavelet scale: thresh*norm
            thresh_array = np.zeros(coeffs.shape) + threshold
            thresh_array *= starlet_norm[:, None, None]
            # We don't threshold the last scale
            thresh_array[-1] = 0
            constraint = ConstraintChain(
                PositivityConstraint(0), L0Constraint(thresh_array)
            )
        else:
            center = tuple(s // 2 for s in bbox.shape)
            constraint = MonotonicMaskConstraint(center, center_radius=1)

        coeffs = Parameter(coeffs, name="coeffs", step=1e-2, constraint=constraint)
        super().__init__(frame, coeffs, bbox=bbox)

    def get_model(self, *parameters):
        # Takes the inverse transform of parameters as starlet coefficients
        coeffs = self.get_parameter(0, *parameters)
        return starlet_reconstruction(coeffs)

    def update(self):
        coeffs = self.get_parameter(0)
        if coeffs.fixed:
            return

        # shrink the box?
        image = self.get_model()
        bbox = self.bbox.copy()
        self.shrink_box(image, thresh=1e-8)
        if bbox != self.bbox:
            slice, _ = overlapped_slices(bbox, self.bbox)
            center = tuple(s // 2 for s in self.bbox.shape)
            if self.monotonic:
                # non-monotonic can keep its constraint as it's independent of size
                constraint = MonotonicMaskConstraint(center, center_radius=1)
            coeffs = Parameter(
                coeffs[:, slice[0], slice[1]],
                name=coeffs.name,
                prior=coeffs.prior,
                constraint=constraint,
                step=coeffs.step,
                fixed=coeffs.fixed,
                m=coeffs.m[:, slice[0], slice[1]] if coeffs.m is not None else None,
                v=coeffs.v[:, slice[0], slice[1]] if coeffs.v is not None else None,
                vhat=coeffs.vhat[:, slice[0], slice[1]]
                if coeffs.vhat is not None
                else None,
            )

            # set new parameters
            self._parameters = (coeffs,) + self._parameters[1:]

            raise UpdateException


class ExtendedSourceMorphology(ImageMorphology):
    def __init__(
        self,
        frame,
        center,
        image,
        bbox=None,
        monotonic="angle",
        symmetric=False,
        min_grad=0,
        shifting=False,
        resizing=True,
    ):
        """Non-parametric image morphology designed for galaxies as extended sources.

        Parameters
        ----------
        frame: `~scarlet.Frame`
            The frame of the full model
        center: tuple
            Center of the source
        image: `numpy.ndarray`
            Image of the source.
        bbox: `~scarlet.Box`
            2D bounding box for focation of the image in `frame`
        monotonic: ['flat', 'angle', 'nearest'] or None
            Which version of monotonic decrease in flux from the center to enforce
        symmetric: `bool`
            Whether or not to enforce symmetry.
        min_grad: float in [0,1)
            Minimal radial decline for monotonicity (in units of reference pixel value)
        shifting: `bool`
            Whether or not a subpixel shift is added as optimization parameter
        resize: bool
            Whether to resize the box dynamically
        """

        constraints = []
        # backwards compatibility: monotonic was boolean
        if monotonic is True:
            monotonic = "angle"
        elif monotonic is False:
            monotonic = None
        if monotonic is not None:
            # most astronomical sources are monotonically decreasing
            # from their center
            constraints.append(
                MonotonicityConstraint(neighbor_weight=monotonic, min_gradient=min_grad)
            )

        if symmetric:
            # have 2-fold rotation symmetry around their center ...
            constraints.append(SymmetryConstraint())

        constraints += [
            # ... and are positive emitters
            PositivityConstraint(),
            # prevent a weak source from disappearing entirely
            CenterOnConstraint(),
            # break degeneracies between sed and morphology
            NormalizationConstraint("max"),
        ]
        morph_constraint = ConstraintChain(*constraints)
        image = Parameter(image, name="image", step=1e-2, constraint=morph_constraint)

        self.pixel_center = np.round(center).astype("int")
        if shifting:
            shift = Parameter(center - self.pixel_center, name="shift", step=1e-1)
        else:
            shift = None
        self.shift = shift

        super().__init__(
            frame, image, bbox=bbox, shifting=shifting, shift=shift, resizing=resizing
        )

    @property
    def center(self):
        if self.shift is not None:
            return self.pixel_center + self.shift
        else:
            return self.pixel_center

import autograd.numpy as np
import numpy.ma as ma

from .bbox import Box
from .constraint import (
    ConstraintChain,
    L0Constraint,
    PositivityConstraint,
    MonotonicityConstraint,
    SymmetryConstraint,
    CenterOnConstraint,
    NormalizationConstraint,
)
from .frame import Frame
from .model import Model, UpdateException
from .parameter import Parameter, relative_step
from .psf import PSF
from .wavelet import Starlet
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
        size = max(image.shape)

        if not self.resizing or image.fixed:
            return

        # shrink the box? peel the onion
        dist = 0
        while (
            np.all(image[dist, :] == 0)
            and np.all(image[-dist, :] == 0)
            and np.all(image[:, dist] == 0)
            and np.all(image[:, -dist] == 0)
        ):
            dist += 1

        newsize = initialization.get_minimal_boxsize(size - 2 * dist)
        if newsize < size:
            dist = (size - newsize) // 2
            # Create new parameter for smaller image
            image = Parameter(
                image[dist:-dist, dist:-dist],
                name=image.name,
                prior=image.prior,
                constraint=image.constraint,
                step=image.step / 2,
                fixed=image.fixed,
                m=image.m[dist:-dist, dist:-dist] if image.m is not None else None,
                v=image.v[dist:-dist, dist:-dist] if image.v is not None else None,
                vhat=image.vhat[dist:-dist, dist:-dist]
                if image.vhat is not None
                else None,
            )

            # set new parameters
            self._parameters = (image,) + self._parameters[1:]

            # adjust bbox
            self.bbox.origin = tuple(o + dist for o in self.bbox.origin)
            self.bbox.shape = (newsize, newsize)
            raise UpdateException

        # grow the box?
        # because the PSF moves power across the box, the gradients at the edge
        # accummulate flux from beyond the box
        if image.m is not None:
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
        if isinstance(center, Parameter):
            assert center.name == "center"
            self.center = center
        else:
            self.center = Parameter(center, name="center", step=3e-2)

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
    threshold: float
        Lower bound on threshold for all but the last starlet scale
    """

    def __init__(self, frame, image, bbox=None, threshold=0):

        if bbox is None:
            assert frame.bbox[1:].shape == image.shape
            bbox = Box(image.shape)

        # Starlet transform of morphologies (n1,n2) with 3 dimensions: (scales+1,n1,n2)
        self.transform = Starlet.fromImage(image)
        # The starlet transform is the model
        coeffs = self.transform.coefficients
        # wavelet-scale norm
        starlet_norm = self.transform.norm
        # One threshold per wavelet scale: thresh*norm
        thresh_array = np.zeros(coeffs.shape) + threshold
        thresh_array *= starlet_norm[:, None, None]
        # We don't threshold the last scale
        thresh_array[-1] = 0

        constraint = L0Constraint(thresh_array)

        coeffs = Parameter(coeffs, name="coeffs", step=1e-2, constraint=constraint)
        super().__init__(frame, coeffs, bbox=bbox)

    def get_model(self, *parameters):
        # Takes the inverse transform of parameters as starlet coefficients
        coeffs = self.get_parameter(0, *parameters)
        return Starlet.fromCoefficients(coeffs).image


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

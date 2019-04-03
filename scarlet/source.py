from __future__ import print_function, division
import numpy as np
import torch

from . import constraint as sc
from .config import Config
from .component import Component, ComponentTree
from . import operator
from .convolution import fft_convolve
from . import utils

import logging
logger = logging.getLogger("scarlet.source")


class Source(ComponentTree):
    """Base class for co-centered `~scarlet.component.Component`s.

    The class implements `update_center` to set all components with `shift_center > 0`
    to the flux-weighted mean center position of all components.
    """
    def __init__(self, components):
        """Constructor.

        Parameters
        ----------
        components: list of `~scarlet.component.Component` or `~scarlet.component.ComponentTree`
        """
        super(Source, self).__init__(components)

    def get_model(self, psfs=None, padding=3, trim=True):
        """Compute the model for this source.

        Returns
        -------
        `~numpy.array` with shape (B, Ny, Nx)
        """
        if len(self.components) > 1:
            model = torch.sum(torch.stack([c.get_model(trim=False) for c in self.components]), dim=0)
        else:
            model = self.components[0].get_model(trim=False)

        if psfs is not None:
            model = torch.stack([
                fft_convolve(model[b], psfs[b], padding=padding)
                for b in range(len(model))
            ])

        if trim:
            model = model[(slice(None), *self.components[0].bbox.slices)]

        return model

    def get_flux(self):
        """Get flux in every band
        """
        if len(self.components) > 1:
            return torch.sum(torch.stack([c.get_flux() for c in self.components]), dim=0)
        else:
            return self.components[0].get_flux()

    def update_center(self, center):
        """Center update to set all component centers to flux-weighted mean position.

        NOTE: Only components with `shift_center > 0` will be moved.
        """
        for c in self.components:
            c.update_center(center)

    def update_bbox(self, bbox=None, min_value=None):
        """Update the bounding boxes.

        If `bbox` is not `None` then all of the components
        have their bounding boxes set to `bbox`.
        Otherwise `min_value` must be specified and all of
        the components are checked to see if they have any flux on
        their edge larger than `min_value`. If they do then the
        bounding box is resized and trimmed.

        Parameters
        ----------
        bbox: `BoundingBox`
            The new bounding box for all of the components
        `min_value`: float
            Minimum value in a component to avoid being trimmed.
        """
        if bbox is not None:
            for c in self.components:
                c.update_bbox(bbox)
        elif min_value is not None:
            for c in self.components:
                c._bbox = utils.resize(c.get_model(trim=False), c.bbox, min_value)
        else:
            msg = "Either `bbox` or `min_value` must be set to update the bounding boxes"
            raise ValueError(msg)

    @staticmethod
    def point_source(center, img, constraints=None, config=None,
                     fix_sed=False, fix_morph=False, min_value=1e-2,
                     normalization=sc.Normalization.Smax):
        """
        center is always a tuple of integers
        """
        if config is None:
            config = Config()

        B, Ny, Nx = img.shape
        _y, _x = center
        # determine initial SED from peak position: amplitude is in sed
        sed = get_pixel_sed(img, center)
        morph = torch.zeros(img.shape[1:], dtype=img.dtype)
        # Turn on a single pixel at the peak: normalized S
        cy, cx = center
        morph[cy, cx] = 1
        bbox = utils.trim(morph)

        if constraints is None:
            constraints = (sc.MinimalConstraint(normalization),
                           sc.DirectMonotonicityConstraint(use_nearest=False),
                           sc.DirectSymmetryConstraint(),
                           sc.ResizeConstraint(min_value))

        component = Component(sed, morph, constraints=constraints, fix_sed=fix_sed,
                              fix_morph=fix_morph, config=config, center=center, bbox=bbox)
        component._normalize(normalization)
        return Source(component)

    @staticmethod
    def extended_source(center, img, bg_rms, constraints=None, symmetric=True, monotonic=True,
                        thresh=1., config=None, fix_sed=False, fix_morph=False,
                        normalization=sc.Normalization.Smax, min_value=1e-2):
        # Use a default configuration if config is not specified
        if config is None:
            config = Config()

        sed, morph = init_extended_source(center, img, bg_rms, thresh=thresh, symmetric=symmetric,
                                          monotonic=monotonic)
        bbox = utils.trim(morph)

        if constraints is None:
            constraints = (sc.MinimalConstraint(normalization),
                           sc.DirectMonotonicityConstraint(),
                           sc.DirectSymmetryConstraint(),
                           sc.ResizeConstraint(min_value))

        component = Component(sed, morph, constraints=constraints, fix_sed=fix_sed,
                              fix_morph=fix_morph, config=config, center=center, bbox=bbox)
        component._normalize(normalization)
        return Source(component)

    @staticmethod
    def multicomponent_source(center, img, bg_rms, flux_percentiles=[25], constraints=None,
                              symmetric=True, monotonic=True, thresh=1., config=None,
                              fix_sed=False, fix_morph=False, normalization=sc.Normalization.Smax):
        # Use a default configuration if config is not specified
        if config is None:
            config = Config()

        seds, morphs = init_multicomponent_source(center, img, bg_rms, thresh=thresh, symmetric=symmetric,
                                                  monotonic=monotonic)
        components = [
            Component(seds[k], morphs[k], constraints=constraints, fix_sed=fix_sed,
                      fix_morph=fix_morph, config=config, center=center)
            for k in range(len(seds))
        ]
        return Source(components)


class SourceInitError(Exception):
    """Error during source initialization
    """
    pass


def get_pixel_sed(img, position):
    """Get the SED at `position` in `img`

    Parameters
    ----------
    img: `~numpy.array`
        (Bands, Height, Width) array that contains a 2D image for each band
    position: array-like
        (y,x) coordinates of the source in the larger image

    Returns
    -------
    SED: `~numpy.array`
        SED for a single source
    """
    _y, _x = position
    sed = torch.zeros((img.shape[0],))
    sed[:] = img[:, _y, _x]
    if torch.all(sed <= 0):
        # If the flux in all bands is  <=0,
        # the new sed will be filled with NaN values,
        # which will cause the code to crash later
        msg = "Zero or negative flux at y={0}, x={1}"
        raise SourceInitError(msg.format(_y, _x))
    return sed


def get_integrated_sed(img, weight, p=1):
    """Calculate SED from weighted sum of the image in each band

    Parameters
    ----------
    img: `~numpy.array`
        (Bands, Height, Width) array that contains a 2D image for each band
    weight: `~numpy.array`
        (Height, Width) weights to apply to each pixel in `img`
    p: int
        power for the weight normalization: 1/(weight**p).sum()

    Returns
    -------
    SED: `~numpy.array`
        SED for a single source

    """
    B, Ny, Nx = img.shape
    sed = (img * weight**p).reshape(B, -1).sum(dim=1) / (weight**p).sum()
    if torch.all(sed <= 0):
        # If the flux in all bands is  <=0,
        # the new sed will be filled with NaN values,
        # which will cause the code to crash later
        msg = "Zero or negative flux under weight function"
        raise SourceInitError(msg)
    return sed


def get_best_fit_sed(img, S):
    """Calculate best fitting SED for multiple components.

    Solves min_A ||img - AS||^2 for the SED matrix A, assuming that img only
    contains a single source.

    Parameters
    ----------
    img: `~numpy.array`
        (Bands, Height, Width) array that contains a 2D image for each band
    S: `~numpy.array`
        (Components, Height, Width) array with the 2D image for each component
    """
    B, K = len(img), len(S)
    Y = img.reshape(B, -1)
    S_ = S.reshape(K, -1)
    return torch.mm(torch.mm(S_, S_.t()).inverse(), torch.mm(S_, Y.t()))


def init_extended_source(center, img, bg_rms, thresh=1., symmetric=True, monotonic=True):
    """Initialize the source that is symmetric and monotonic

    See `Source.ExtendedSource` for a description of the parameters
    """
    # determine initial SED from peak position
    B = img.shape[0]
    sed = get_pixel_sed(img, center)  # amplitude is in sed

    # build optimal detection coadd given the sed
    if torch.all(bg_rms > 0):
        bg_rms = bg_rms
        weights = torch.tensor([sed[b]/bg_rms[b]**2 for b in range(B)])
        jacobian = torch.tensor([sed[b]**2/bg_rms[b]**2 for b in range(B)]).sum()
        detect = torch.einsum('i,i...', weights, img) / jacobian

        # thresh is multiple above the rms of detect (weighted variance across bands)
        bg_cutoff = thresh * np.sqrt((weights**2 * bg_rms**2).sum()) / jacobian
    else:
        detect = np.sum(img, axis=0)
        bg_cutoff = 0

    morph = detect

    # symmetric, monotonic
    if symmetric:
        morph = operator.prox_uncentered_symmetry(morph, 0, center=center, use_soft=False)
    if monotonic:
        # use finite thresh to remove flat bridges
        prox_monotonic = operator.prox_strict_monotonic(morph.shape, use_nearest=False,
                                                        center=center, thresh=.1)
        morph = prox_monotonic(morph, 0).reshape(morph.shape)

    # trim morph to pixels above threshold
    mask = morph > bg_cutoff
    if mask.sum() == 0:
        msg = "No flux above threshold={2} for source at y={0} x={1}"
        raise SourceInitError(msg.format(center[0], center[1], bg_cutoff))
    morph[~mask] = 0
    ypix, xpix = np.where(mask)
    left, right, bottom, top = np.min(xpix), np.max(xpix), np.min(ypix), np.max(ypix)
    morph_slice = (slice(bottom, top), slice(left, right))

    # normalize to unity at peak pixel
    cy, cx = center
    center_morph = morph[cy, cx]
    morph /= center_morph

    # use mean sed from image, weighted with the morphology of each component
    try:
        # need p=2 to undo the intensity normalization, since we want Smax initially
        sed = get_integrated_sed(img[:, morph_slice[0], morph_slice[1]], morph[morph_slice], p=4)
    except SourceInitError:
        # keep the peak sed
        logger.INFO("Using peak SED for source at {0}".format(center))
    return sed, morph


def init_multicomponent_source(center, img, bg_rms, flux_percentiles=[25],
                               symmetric=True, monotonic=True, thresh=1.):
    center = np.array(center).astype(int)
    # Initialize the first component as an extended source
    sed, morph = init_extended_source(center, img, bg_rms, thresh=thresh,
                                      symmetric=symmetric, monotonic=monotonic)
    # create a list of components from base morph by layering them on top of
    # each other so that they sum up to morph
    K = len(flux_percentiles) + 1

    Ny, Nx = morph.shape
    morphs = torch.zeros((K, Ny, Nx), dtype=morph.dtype)
    morphs[0, :, :] = morph[:, :]
    max_flux = morph.max()
    percentiles_ = np.sort(flux_percentiles)
    last_thresh = 0
    for k in range(1, K):
        perc = percentiles_[k-1]
        flux_thresh = perc*max_flux/100
        mask_ = morph > flux_thresh
        morphs[k-1][mask_] = flux_thresh - last_thresh
        morphs[k][mask_] = morph[mask_] - flux_thresh
        last_thresh = flux_thresh

    # renormalize morphs: initially Smax
    for k in range(K):
        morphs[k] /= morphs[k].max()

    # optimal SEDs given the morphologies, assuming img only has that source
    seds = get_best_fit_sed(img, morphs)

    return seds, morphs

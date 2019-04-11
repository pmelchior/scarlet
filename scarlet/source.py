from __future__ import print_function, division
import numpy as np
import torch

from .component import Component, ComponentTree
from . import operator
from . import update
from . import observation

import logging
logger = logging.getLogger("scarlet.source")


class SourceInitError(Exception):
    """Error during source initialization
    """
    pass


def get_pixel_sed(sky_coord, scene, observations):
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
    sed = torch.zeros(scene.B).float()
    band = 0
    for obs in observations:
        pixel = obs.get_pixel(sky_coord)
        sed[band:band+obs.B] = obs.images[:, pixel[0], pixel[1]]
        band += obs.B
    if torch.all(sed <= 0):
        # If the flux in all bands is  <=0,
        # the new sed will be filled with NaN values,
        # which will cause the code to crash later
        msg = "Zero or negative flux at y={0}, x={1}"
        raise SourceInitError(msg.format(*sky_coord))
    return sed


def get_best_fit_seds(morphs, scene, observations):
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
    K = len(morphs)
    seds = torch.tensor((K, scene.B))
    band = 0
    _morph = morphs.reshape(K, -1)
    for k, obs in enumerate(observations):
        images = obs.get_scene(scene)
        data = images.reshape(obs.B, -1)
        sed = torch.mm(torch.mm(_morph, _morph.t()).inverse(), torch.mm(_morph, data.t()))
        seds[k, band:band+obs.B] = sed
    return seds


def init_extended_source(sky_coord, scene, observations, bg_rms, obs_idx=0,
                         thresh=1., symmetric=True, monotonic=True):
    """Initialize the source that is symmetric and monotonic
    See `Source.ExtendedSource` for a description of the parameters
    """
    try:
        iter(observations)
    except TypeError:
        observations = [observations]
    # determine initial SED from peak position
    sed = get_pixel_sed(sky_coord, scene, observations)  # amplitude is in sed
    morph, bg_cutoff = observation.build_detection_coadd(sed, bg_rms, observations[obs_idx], scene, thresh)
    center = scene.get_pixel(sky_coord)

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
        raise SourceInitError(msg.format(*sky_coord, bg_cutoff))
    morph[~mask] = 0

    # normalize to unity at peak pixel
    cy, cx = center
    center_morph = morph[cy, cx]
    morph /= center_morph
    return sed, morph


def init_multicomponent_source(sky_coord, scene, observations, bg_rms, flux_percentiles=None, obs_idx=0,
                               thresh=1., symmetric=True, monotonic=True):
    if flux_percentiles is None:
        flux_percentiles = [25]
    # Initialize the first component as an extended source
    sed, morph = init_extended_source(sky_coord, scene, observations, bg_rms, obs_idx,
                                      thresh, symmetric, monotonic)
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
    seds = get_best_fit_seds(morphs, scene, observations)

    return seds, morphs


class PointSource(Component):
    """Create a point source.
    Point sources are initialized with the SED of the center pixel,
    and the morphology of a single pixel (the center) turned on.
    While the source can have any `constraints`, the default constraints are
    symmetry and monotonicity.
    """
    def __init__(self, scene, sky_coord, observations, **component_kwargs):
        # this ignores any broadening from the PSFs ...
        B, Ny, Nx = scene.shape
        morph = torch.tensor(Ny, Nx)
        pixel = scene.get_pixel(sky_coord)
        morph[pixel] = 1
        self.pixel_center = pixel

        sed = torch.tensor(B)
        b0 = 0
        for obs in observations:
            pixel = obs.get_pixel(sky_coord)
            sed[b0:b0+obs.B] = obs.img[(None, *pixel)]
            b0 += obs.B

        super(PointSource, self).__init__(sed, morph, **component_kwargs)

    def update(self):
        update.symmetric_fit_center(self.morph)  # update the center position
        update.symmetric(self, self.pixel_center)  # make the morph perfectly symmetric
        update.monotonic(self, self.pixel_center)  # make the morph monotonically decreasing
        update.positive(self)  # Make the SED and morph non-negative
        update.normalize(self)
        return self


class ExtendedSource(Component):
    def __init__(self, scene, sky_coord, observations, bg_rms, obs_idx=0, thresh=1,
                 symmetric=True, monotonic=True, **component_kwargs):
        self.symmetric = symmetric
        self.monotonic = monotonic
        self.coords = sky_coord
        center = scene.get_pixel(sky_coord)
        self.pixel_center = center

        sed, morph = init_extended_source(sky_coord, scene, observations, bg_rms, obs_idx,
                                          thresh, symmetric, monotonic)

        super().__init__(sed, morph, **component_kwargs)

    def update(self):
        if self.symmetric:
            update.symmetric_fit_center(self.morph)  # update the center position
            center = self.coords
            update.symmetric(self, center)  # make the morph perfectly symmetric
        elif self.monotonic:
            update.fit_pixel_center(self)
            center = self.pixel_center
        if self.monotonic:
            update.monotonic(self, center)  # make the morph monotonically decreasing
        update.positive(self)  # Make the SED and morph non-negative
        update.normalize(self)
        return self


class MultiComponentSource(ComponentTree):
    """Create an extended source with multiple components layered vertically.
    Uses `~scarlet.source.ExtendedSource` to define the overall morphology,
    then erodes the outer footprint until it reaches the specified size percentile.
    For the narrower footprint, it evaluates the mean value at the perimeter and
    sets the inside to the perimeter value, creating a flat distribution inside.
    The subsequent component(s) is/are set to the difference between the flattened
    and the overall morphology.
    The SED for all components is calculated as the best fit of the multi-component
    morphology to the multi-band image in the region of the source.
    """
    def __init__(self, scene, sky_coord, observations, bg_rms, obs_idx=0, thresh=1,
                 symmetric=True, monotonic=True, **component_kwargs):
        seds, morphs = init_multicomponent_source(sky_coord, scene, observations, bg_rms, obs_idx,
                                                  thresh, symmetric, monotonic)

        class MultiComponent(Component):
            def __init__(self, sed, morph, symmetric, monotonic, **kwargs):
                self.symmetric = symmetric
                self.monotonic = monotonic
                super().__init__(sed, morph, **kwargs)

            def update(self):
                if self.symmetric:
                    update.symmetric_fit_center(self.morph)  # update the center position
                    center = self.coords
                    update.symmetric(self, center)  # make the morph perfectly symmetric
                elif self.monotonic:
                    update.fit_pixel_center(self)
                    center = self.pixel_center
                if self.monotonic:
                    update.monotonic(self, center)  # make the morph monotonically decreasing
                update.positive(self)  # Make the SED and morph non-negative
                update.normalize(self)
                return self

        components = [
            MultiComponent(seds[k], morphs[k], symmetric, monotonic, **component_kwargs)
            for k in range(len(seds))
        ]
        super().__init__(components)

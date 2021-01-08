import astropy
import logging
import numpy as np

from .bbox import Box
from .psf import PSF, ImagePSF
from . import interpolation

logger = logging.getLogger("scarlet.frame")


class Frame:
    """Spatial and spectral characteristics of the data

    Attributes
    ----------
    shape: tuple
        shape tuple (Channel, Height, Width)
    wcs: TBD
        World Coordinates
    psfs: `scarlet.PSF` or its arguments
        PSF in each channel
    channels: list of hashable elements
        Names/identifiers of spectral channels
    dtype: `numpy.dtype`
        Dtype to represent the data.
    """

    def __init__(self, shape, channels, wcs=None, psfs=None, dtype=np.float32):
        self._bbox = Box(shape)
        assert len(channels) == self.C
        self.channels = channels

        if wcs is not None:
            assert isinstance(wcs, astropy.wcs.WCS)
            self.wcs = wcs.celestial  # only use celestial portion
        else:
            self.wcs = None

        if psfs is None:
            logger.warning("No PSF specified. Possible, but dangerous!")
            self._psfs = None
        else:

            if isinstance(psfs, PSF):
                self._psfs = psfs
            else:
                self._psfs = PSF(psfs)

        self.dtype = dtype

    @property
    def bbox(self):
        """The `~scarlet.bbox.Box` of this `Frame`.
        """
        return self._bbox

    @property
    def shape(self):
        return self._bbox.shape

    @property
    def C(self):
        """Number of channels in the model
        """
        return self._bbox.shape[0]

    @property
    def Ny(self):
        """Number of pixel in the y-direction
        """
        return self._bbox.shape[1]

    @property
    def Nx(self):
        """Number of pixels in the x-direction
        """
        return self._bbox.shape[2]

    @property
    def psf(self):
        return self._psfs

    def get_pixel(self, sky_coord):
        """Get the pixel coordinate from a world coordinate

        Parameters
        ----------
        sky_coord: tuple, array
            Coordinates on the sky
        """
        sky = np.array(sky_coord, dtype=np.float).reshape(-1, 2)

        if self.wcs is not None:
            pixel = np.array(self.wcs.world_to_pixel_values(sky)).reshape(-1, 2)
            # y/x instead of x/y:
            pixel = np.flip(pixel, axis=-1)
        else:
            pixel = sky

        if pixel.size == 2:  # only one coordinate pair
            return pixel[0]
        return pixel

    def get_sky_coord(self, pixel):
        """Get the sky coordinate from a pixel coordinate

        Parameters
        ----------
        pixel: tuple, array
            Coordinates in the pixel space
        """
        pix = np.array(pixel, dtype=np.float).reshape(-1, 2)

        if self.wcs is not None:
            # x/y instead of y/x:
            pix = np.flip(pix, axis=-1)
            sky = np.array(self.wcs.pixel_to_world_values(pix))
        else:
            sky = pix

        if sky.size == 2:
            return sky[0]
        return sky

    def convert_pixel_to(self, target, pixel=None):
        """Converts pixel coordinates from this frame to `target` Frame

        Parameters
        ----------
        target: `~scarlet.Frame`
            target frame
        pixel: `tuple` or list ((y1,y2, ...), (x1, x2, ...))
            pixel coordinates in this frame
            If not set, convert all pixels in this frame

        Returns
        -------
        coord_target: `tuple`
            coordinates at the location of `coord` in the target frame
        """
        if pixel is None:
            y, x = np.indices(self.shape[-2:], dtype=np.float)
            pixel = np.stack((y.flatten(), x.flatten()), axis=1)

        ra_dec = self.get_sky_coord(pixel)
        pixel_ = target.get_pixel(ra_dec)

        if pixel_.size == 2:  # only one coordinate pair
            return pixel_[0]
        return pixel_

    @staticmethod
    def from_observations(observations, target_psf=None, obs_id=None, coverage="union"):
        """Generates a suitable model frame for a set of observations.

        This method generates a frame from a set of observations by indentifying the highest resolution
        and the smallest PSF and use them to construct a common frome for all observations in the set.

        Parameters
        ----------
        observations: array of `scarlet.Observation` objects
            array that contains Observations to match onto a common frame
        target_psfs: `scarlet.PSF`
            Target PSF to which oll observations are to be deconvolved.
            If set to None, uses the smallest PSF across all observations and channels.
        obs_id: int
            index of the reference observation
            If set to None, uses the observation with the smallest pixels.
        coverage: "union" or "intersection"
            Sets the frame to incorporate the pixels covered by any observation ('union')
            or by all observations ('intersection').
        """
        from scarlet.observation import LowResObservation

        assert coverage in ["union", "intersection"]

        if not hasattr(observations, "__iter__"):
            observations = (observations,)

        # Array of pixel sizes for each observation
        pix_tab = []
        # Array of psf size for each psf of each observation
        fat_psf_size = None
        small_psf_size = None
        channels = []
        # Create frame channels and find smallest and largest psf
        for c, obs in enumerate(observations):
            # Concatenate all channels
            channels = channels + obs.frame.channels
            # concatenate all pixel sizes
            pix_tab.append(
                interpolation.get_pixel_size(interpolation.get_affine(obs.frame.wcs))
            )
            h_temp = interpolation.get_pixel_size(
                interpolation.get_affine(obs.frame.wcs)
            )
            # Looking for the sharpest and the fatest psf
            psfs = obs.frame.psf.get_model()._data
            for psf in psfs:
                psf_size = interpolation.get_psf_size(psf) * h_temp
                if (fat_psf_size is None) or (psf_size > fat_psf_size):
                    fat_psf_size = psf_size
                if (obs_id is None) or (c == obs_id):
                    if (target_psf is None) and (
                        (small_psf_size is None) or (psf_size < small_psf_size)
                    ):
                        small_psf_size = psf_size
                        target_psf_temp = ImagePSF(psf[np.newaxis, :, :])
                        psf_h = h_temp

        # Find a reference observation. Either provided by obs_id or as the observation with the smallest pixel
        if obs_id is None:
            obs_ref = observations[np.where(pix_tab == np.min(pix_tab))[0][0]]
        else:
            # Frame defined from obs_id
            obs_ref = observations[obs_id]

        # Scale of the smallest pixel
        h = interpolation.get_pixel_size(interpolation.get_affine(obs_ref.frame.wcs))

        # If needed and psf is not provided: interpolate psf to smallest pixel
        if target_psf is None:
            # If the reference PSF is not at the highest pixel resolution, make it!
            if psf_h > h:
                angle, h = interpolation.get_angles(target_wcs, obs.frame.wcs)
                target_psf = PSF(
                    interpolation.sinc_interp_inplace(target_psf_temp, psf_h, h, angle)
                )
            else:
                target_psf = target_psf_temp

        # Determine overlap with respect to the frame of obs_ref
        ref_box = obs_ref.frame.bbox[-2:]
        for c, obs in enumerate(observations):
            if obs is not obs_ref:
                # Make observations with a different wcs LowResObservation
                # TODO (see #220)
                if type(obs) is not LowResObservation:
                    observations[c] = obs.get_LowRes()

                # Limits that include all observations relative to target_wcs
                obs_coord = obs.frame.convert_pixel_to(obs_ref.frame)
                y_min = np.min(obs_coord[:, 0])
                x_min = np.min(obs_coord[:, 1])
                y_max = np.max(obs_coord[:, 0])
                x_max = np.max(obs_coord[:, 1])
                new_box = Box.from_bounds((y_min, y_max + 1), (x_min, x_max + 1))
                if coverage == "union":
                    ref_box |= new_box
                else:
                    ref_box = new_box & ref_box

        # Margin in pixels
        # Padding by the size of the psf
        fat_pixel_size = fat_psf_size / h
        ny, nx = ref_box.shape
        target_shape = (
            len(channels),
            np.round((ny + fat_pixel_size)).astype("int"),
            np.round((nx + fat_pixel_size)).astype("int"),
        )
        target_origin = np.round(np.array(ref_box.origin) - fat_pixel_size / 2).astype(
            "int"
        )

        # create new wcs as a copy from obs_ref, but shifted to new origin
        target_wcs = obs_ref.frame.wcs.deepcopy()
        target_wcs.wcs.crpix -= target_origin
        target_frame = Frame(
            target_shape, psfs=target_psf, channels=channels, wcs=target_wcs
        )

        # Match observations to this frame
        for obs in observations:
            obs.match(target_frame)

        return target_frame

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
    psf: `scarlet.PSF` or its arguments
        PSF in each channel
    channels: list of hashable elements
        Names/identifiers of spectral channels
    dtype: `numpy.dtype`
        Dtype to represent the data.
    """

    def __init__(self, shape, channels, wcs=None, psf=None, dtype=np.float32):
        self._bbox = Box(shape)
        assert len(channels) == self.C
        self.channels = channels

        if wcs is not None:
            assert isinstance(wcs, astropy.wcs.WCS)
            self.wcs = wcs
        else:
            self.wcs = None

        if psf is None:
            logger.warning("No PSF specified. Possible, but dangerous!")
            self._psf = None
        else:

            if isinstance(psf, PSF):
                self._psf = psf
            else:
                self._psf = PSF(psf)

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
        return self._psf

    def get_pixel(self, sky_coord):
        """Get the pixel coordinate from a world coordinate

        Parameters
        ----------
        sky_coord: tuple, array
            Coordinates on the sky
        """
        sky = np.array(sky_coord, dtype=np.float64).reshape(-1, 2)

        if self.wcs is not None:
            wcs_ = self.wcs.celestial  # only use celestial portion
            pixel = np.array(wcs_.world_to_pixel_values(sky)).reshape(-1, 2)
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
        pix = np.array(pixel, dtype=np.float64).reshape(-1, 2)

        if self.wcs is not None:
            wcs_ = self.wcs.celestial  # only use celestial portion
            # x/y instead of y/x:
            pix = np.flip(pix, axis=-1)
            sky = np.array(wcs_.pixel_to_world_values(pix))
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
            y, x = np.indices(self.shape[-2:], dtype=np.float64)
            pixel = np.stack((y.flatten(), x.flatten()), axis=1)

        ra_dec = self.get_sky_coord(pixel)
        pixel_ = target.get_pixel(ra_dec)

        if pixel_.size == 2:  # only one coordinate pair
            return pixel_[0]
        return pixel_

    @staticmethod
    def from_observations(
        observations, model_psf=None, model_wcs=None, obs_id=None, coverage="union"
    ):
        """Generates a suitable model frame for a set of observations.

        This method generates a frame from a set of observations by identifying the highest resolution
        and the smallest PSF and use them to construct a common frame for all observations.

        Parameters
        ----------
        observations: array of `scarlet.Observation` objects
            array that contains Observations to match onto a common frame
        model_psf: `scarlet.PSF`
            PSF of the model frame, to which all observations are to be deconvolved.
            If None, uses the smallest PSF across all observations and channels.
        model_wcs: `astropy.wcs.WCS`
            WCS for the model frame. If None, uses transformation of the observation
            with the smallest pixels.
        obs_id: int
            index of the reference observation
            If set to None, uses the observation with the smallest pixels.
        coverage: "union" or "intersection"
            Sets the frame to incorporate the pixels covered by any observation ('union')
            or by all observations ('intersection').
        """
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
            channels = channels + obs.channels
            # concatenate all pixel sizes
            h_temp = interpolation.get_pixel_size(interpolation.get_affine(obs.wcs))
            pix_tab.append(h_temp)
            # Looking for the sharpest and the fatest psf
            psf = obs.psf.get_model()._data
            for psf in psf:
                psf_size = interpolation.get_psf_size(psf) * h_temp
                if (fat_psf_size is None) or (psf_size > fat_psf_size):
                    fat_psf_size = psf_size
                if (obs_id is None) or (c == obs_id):
                    if (model_psf is None) and (
                        (small_psf_size is None) or (psf_size < small_psf_size)
                    ):
                        small_psf_size = psf_size
                        model_psf_temp = ImagePSF(psf[np.newaxis, :, :])
                        psf_h = h_temp

        # Find a reference observation. Either provided by obs_id or as the observation with the smallest pixel
        if obs_id is None:
            obs_ref = observations[np.where(pix_tab == np.min(pix_tab))[0][0]]
        else:
            # Frame defined from obs_id
            obs_ref = observations[obs_id]

        # Reference wcs
        if model_wcs is None:
            model_wcs = obs_ref.wcs

        # Scale of the smallest pixel
        h = interpolation.get_pixel_size(interpolation.get_affine(model_wcs))

        # If needed and psf is not provided: interpolate psf to smallest pixel
        if model_psf is None:
            # If the reference PSF is not at the highest pixel resolution, make it!
            if psf_h > h:
                angle, h = interpolation.get_angles(model_wcs, obs.wcs)
                model_psf = PSF(
                    interpolation.sinc_interp_inplace(model_psf_temp, psf_h, h, angle)
                )
            else:
                model_psf = model_psf_temp

        # Dummy frame for WCS computations
        model_shape = (len(channels), 0, 0)
        model_frame = Frame(
            model_shape, channels=channels, psf=model_psf, wcs=model_wcs
        )

        # Determine overlap of all observations in pixel coordinates of the model frame
        for c, obs in enumerate(observations):

            if model_frame.wcs is obs.wcs:
                this_box = obs_ref.bbox[-2:]
            else:
                obs_coord = obs.convert_pixel_to(model_frame)
                y_min = np.floor(np.min(obs_coord[:, 0])).astype("int")
                x_min = np.floor(np.min(obs_coord[:, 1])).astype("int")
                y_max = np.ceil(np.max(obs_coord[:, 0])).astype("int")
                x_max = np.ceil(np.max(obs_coord[:, 1])).astype("int")
                this_box = Box.from_bounds((y_min, y_max + 1), (x_min, x_max + 1))

            if c == 0:
                model_box = this_box
            else:
                if coverage == "union":
                    model_box |= this_box
                else:
                    model_box &= this_box

        # pad by the size of the widest psf to prevent leakage across the frame edge
        ny, nx = model_box.shape
        pad_size = fat_psf_size / h / 2
        offset = (np.round(pad_size).astype("int"), np.round(pad_size).astype("int"))
        model_box -= offset
        model_box.shape = tuple(s + 2 * o for s, o in zip(model_box.shape, offset))

        # move the reference pixel of the model wcs to the 0/0 pixel of the new shape
        model_wcs = model_wcs.deepcopy()
        model_wcs.wcs.crpix -= model_box.origin
        model_wcs.array_shape = model_box.shape

        # recreate the model frame with the correct shape
        frame_shape = (len(channels), *model_box.shape)
        model_frame = Frame(
            frame_shape, channels=channels, psf=model_psf, wcs=model_wcs
        )

        # Match observations to this frame
        for obs in observations:
            obs.match(model_frame)

        return model_frame

import astropy
import logging
import numpy as np

from .bbox import Box
from .psf import PSF, ImagePSF
from . import interpolation
from . import resampling

logger = logging.getLogger("scarlet.frame")


class Frame:
    """Spatial and spectral characteristics of the data

    Attributes
    ----------
    shape_or_box: tuple
        shape tuple (Channel, Height, Width) or image/cube with a shape
    wcs: TBD
        World Coordinates
    psfs: `scarlet.PSF` or its arguments
        PSF in each channel
    channels: list of hashable elements
        Names/identifiers of spectral channels
    dtype: `numpy.dtype`
        Dtype to represent the data.
    """

    def __init__(self, shape_or_box, channels, wcs=None, psfs=None, dtype=np.float32):
        if isinstance(shape_or_box, Box):
            self._bbox = shape_or_box
        else:
            self._bbox = Box(shape_or_box)

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
    def origin(self):
        return self._bbox.origin

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

    @staticmethod
    def from_observations(
        observations, target_psf=None, target_wcs=None, obs_id=None, coverage="union"
    ):
        """Generates a frame from a set of observations.

        By default, this method will generate a frame from a set of observations by indentifying the highest resolution
        and the smallest PSF and use them to construct a common frome for all observations in the set.

        Parameters
        ----------
        observations: array of `scarlet.Observation` objects
            array that contains Observations to match onto a common frame
        psfs: `scarlet.PSF` or its arguments
            Target PSF to which oll observations are to be deconvolved.If set to None we use the smallest PSF across
            all observations and channels.
        wcs: TBD
            World Coordinates of the target frame
        obs_id:
            index of the reference observation. If not set to None, the observation with the smallest PSF and smallest
            pixel is used
        coverage: `string`
            Sets the frame to incorporate all the pixels covered by all the model ('union')
            or sets the frame to incorporate only the pixels vovered by all the observations ('intersection').
            Default is 'union'.
        """
        assert coverage in ["union", "intersection"]
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
        # Reference wcs
        if target_wcs is None:
            target_wcs = obs_ref.frame.wcs
        # Scale of the smallest pixel
        h = interpolation.get_pixel_size(interpolation.get_affine(target_wcs))

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

        # Margin in pixels
        fat_pixel_size = (fat_psf_size / h).astype(int)
        # Padding by the size of the psf
        if fat_pixel_size % 2 != 0:
            fat_pixel_size += 1

        # Matching observations together with the target_wcs so as to create a common frame\
        # Box for the reference observation
        ref_box = obs_ref.frame.bbox
        from .observation import LowResObservation

        target_frame = Frame(
            (len(channels), 0, 0), psfs=target_psf, channels=channels, wcs=target_wcs
        )
        for c, obs in enumerate(observations):
            # Make observations with a different wcs LowResObservation
            if (obs is not obs_ref) and (type(obs) is not LowResObservation):
                observations[c] = obs.get_LowRes()
                # Limits that include all observations relative to target_wcs
                obs_coord = resampling.get_to_common_frame(obs.frame, target_frame)
                y_min = np.min(obs_coord[0])
                x_min = np.min(obs_coord[1])
                y_max = np.max(obs_coord[0])
                x_max = np.max(obs_coord[1])
                new_box = Box(
                    (obs.frame.C, y_max - y_min + 1, x_max - x_min + 1),
                    origin=(0, y_min, x_min),
                )
                if coverage == "union":
                    ref_box |= new_box
                else:
                    ref_box = new_box & ref_box

        _, ny, nx = ref_box.shape
        frame_shape = (
            len(channels),
            np.int((ny + fat_pixel_size)),
            np.int((nx + fat_pixel_size)),
        )
        _, o_y, o_x = ref_box.origin
        fbox = Box(
            frame_shape,
            origin=(
                0,
                np.int(o_y - fat_pixel_size / 2),
                np.int(o_x - fat_pixel_size / 2),
            ),
        )
        target_frame._bbox = fbox

        # Match observations to this frame
        for obs in observations:
            obs.match(target_frame)

        return target_frame

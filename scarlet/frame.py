import numpy as np
from .psf import PSF
from .bbox import Box
from . import interpolation
import logging
from. import resampling

logger = logging.getLogger("scarlet.frame")


class Frame(Box):
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

    def __init__(
        self, shape_or_box, wcs=None, psfs=None, channels=None, dtype=np.float32
    ):

        if isinstance(shape_or_box, Box):
            super().__init__(shape_or_box.shape, shape_or_box.origin)
        else:
            super().__init__(shape_or_box)

        self.wcs = wcs

        if psfs is None:
            logger.warning("No PSF specified. Possible, but dangerous!")
            self._psfs = None
        else:
            if isinstance(psfs, PSF):
                self._psfs = psfs
            else:
                self._psfs = PSF(psfs)

        assert channels is None or len(channels) == self.shape[0]
        self.channels = channels
        self.dtype = dtype



    @property
    def C(self):
        """Number of channels in the model
        """
        return self.shape[0]

    @property
    def Ny(self):
        """Number of pixel in the y-direction
        """
        return self.shape[1]

    @property
    def Nx(self):
        """Number of pixels in the x-direction
        """
        return self.shape[2]

    @property
    def psf(self):
        return self._psfs

    def get_pixel(self, sky_coord):
        """Get the pixel coordinate from a world coordinate
        If there is no WCS associated with the `Scene`,
        meaning the data frame and model frame are the same,
        then this just returns `sky_coord`.
        """
        if self.wcs is not None:
            if self.wcs.naxis == 3:
                coord = self.wcs.wcs_world2pix(*sky_coord, 0, 0)
            elif self.wcs.naxis == 2:
                coord = self.wcs.wcs_world2pix(*sky_coord, 0)
            else:
                raise ValueError(
                    "Invalid number of wcs dimensions: {0}".format(self.wcs.naxis)
                )
            return tuple(int(c.item()) for c in coord)

        return tuple(int(coord) for coord in sky_coord)

    def get_sky_coord(self, pixel):
        """Get the world coordinate for a pixel coordinate
        If there is no WCS associated with the `Scene`,
        meaning the data frame and model frame are the same,
        then this just returns `pixel`.
        """
        if self.wcs is not None:
            if self.wcs.naxis == 3:
                coord = self.wcs.wcs_pix2world(*pixel, 0, 0)
            elif self.wcs.naxis == 2:
                coord = self.wcs.wcs_pix2world(*pixel, 0)
            else:
                raise ValueError(
                    "Invalid number of wcs dimensions: {0}".format(self.wcs.naxis)
                )
            return tuple(c.item() for c in coord)
        return tuple(pixel)

    @staticmethod
    def from_observations(observations, target_psf = None, target_wcs = None, obs_id = None, coverage = 'union'):
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
        assert coverage in ['union', 'intersection']
        # Array of pixel sizes for each observation
        pix_tab = []
        # Array of psf size for each psf of each observation
        fat_psf_size = None
        small_psf_size = None
        channels = []
        #Create frame channels and find smallest and largest psf
        for c, obs in enumerate(observations):
            #Concatenate all channels
            channels = channels + obs.frame.channels
            #concatenate all pixel sizes
            pix_tab.append(interpolation.get_pixel_size(interpolation.get_affine(obs.frame.wcs)))
            h_temp = interpolation.get_pixel_size(interpolation.get_affine(obs.frame.wcs))
            # Looking for the sharpest and the fatest psf
            for psf in obs.frame._psfs.image:
                psf_size = get_psf_size(psf)*h_temp
                if (fat_psf_size is None) or (psf_size > fat_psf_size):
                    fat_psf_size = psf_size
                if (obs_id is None) or (c == obs_id):
                    if (target_psf is None) and ((small_psf_size is None) or (psf_size < small_psf_size)):
                        small_psf_size = psf_size
                        target_psf_temp = PSF(psf[np.newaxis, :, :])
                        psf_h = h_temp

        # Find a reference observation. Either provided by obs_id or as the observation with the smallest pixel
        if obs_id is None:
            obs_ref = observations[np.where(pix_tab == np.min(pix_tab))[0][0]]
        else:
            #Frame defined from obs_id
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
                target_psf = PSF(interpolation.sinc_interp_inplace(target_psf_temp, psf_h, h, angle))
            else:
                target_psf = target_psf_temp

        # Matching observations together with the target_wcs so as to create a common frame
        y_min, x_min, y_max, x_max = 0, 0, 0, 0
        for c, obs in enumerate(observations):
            # Make observations with a different wcs LowResObservation
            if (obs.frame.wcs is not target_wcs) and (type(obs) is not 'LowResObservation'):
                observations[c] = obs.make_LowRes()
            # Limits that include all observations relative to target_wcs
            obs_coord = resampling.get_to_common_frame(obs, target_wcs)
            if int(np.min(obs_coord[0])) < y_min:
                y_min = np.min(obs_coord[0])
            if int(np.min(obs_coord[1])) < x_min:
                x_min = np.min(obs_coord[1])
            #The upper limit is the bottom right corner
            if int(np.max(obs_coord[0])) > y_max:
                y_max = np.max(obs_coord[0])
            if int(np.max(obs_coord[1])) > x_max:
                x_max = np.max(obs_coord[1])

        # Margin in pixels
        fat_pixel_size = (fat_psf_size/h).astype(int)
        #Padding by the size of the psf
        if fat_pixel_size % 2 != 0:
            fat_pixel_size += 1

        offset = np.array([y_min, x_min])
        # Shape of the box that encompasses all observations
        ny = (y_max - offset[0] + 1).astype(int)
        nx = (x_max - offset[1] + 1).astype(int)

        coord_frame = (np.indices((ny,nx)))
        footprint = 0
        for obs in observations:
            if obs.frame.wcs != target_wcs:
                coord_2obs = np.around(resampling.convert_coordinates(coord_frame, target_wcs, obs.frame.wcs))
            else:
                coord_2obs = coord_frame
            footprint += (coord_2obs[0] >= -np.finfo(float).eps) * (coord_2obs[1] >= -np.finfo(float).eps) *\
                (coord_2obs[0] <= obs.frame.shape[-2]) * (coord_2obs[1] <= obs.frame.shape[-1])

        if coverage is 'union':
            coord_cover = np.where(footprint != 0)

        elif coverage is 'intersection':
            coord_cover = (np.where(footprint == np.max(footprint)))
        #Corners of the footprint with padding by the psf size
        y_fmin = np.min(coord_cover[0]) - fat_pixel_size / 2 + offset[0]
        x_fmin = np.min(coord_cover[1]) - fat_pixel_size / 2 + offset[1]
        y_fmax = np.max(coord_cover[0]) + fat_pixel_size / 2 + offset[0]
        x_fmax = np.max(coord_cover[1]) + fat_pixel_size / 2 + offset[1]
        # Shape of the resulting frame
        ny = (y_fmax - y_fmin + 1).astype(int)
        nx = (x_fmax - x_fmin + 1).astype(int)
        frame_shape =(len(channels), ny, nx)

        fbox = Box(frame_shape, origin = (0,np.around(y_fmin).astype(int), np.around(x_fmin).astype(int)))
        frame = Frame(fbox, wcs=target_wcs, psfs=target_psf, channels=channels)

        # Match observations to this frame
        for obs in observations:
            obs.match(frame)

        return frame


def get_psf_size(psf):
    """ Measures the size of a psf by computing the size of the area in 3 sigma around the center.

    This is an approximate method to estimate the size of the psf for setting the size of the frame,
    which does not require a precise measurement.

    Parameters
    ----------
        PSF: `scarlet.PSF` object
            PSF for whic to compute the size
    Returns
    -------
        sigma3: `float`
            radius of the area inside 3 sigma around the center in pixels
    """
    # Normalisation by maximum
    psf_frame = psf/np.max(psf)

    # Pixels in the FWHM set to one, others to 0:
    psf_frame[psf_frame>0.5] = 1
    psf_frame[psf_frame<=0.5] = 0

    # Area in the FWHM:
    area = np.sum(psf_frame)

    # Diameter of this area
    d = 2*(area/np.pi)**0.5

    # 3-sigma:
    sigma3 = 3*d/(2*(2*np.log(2))**0.5)

    return sigma3
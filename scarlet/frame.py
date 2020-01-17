import numpy as np
from .psf import PSF
from .bbox import Box
from . import interpolation
import logging

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
            self = shape_or_box
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
        print(*sky_coord)
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

        channels = []
        #Check that pixels are square and create frame channels
        for obs in observations:
            channels = channels + obs.channels

        # Find target psf and wcs if not set by arguments
        if obs_id is None:
            if target_wcs is None:
                h = None
                for obs in observations:
                    # Finds the smallest pixel scale
                    pix = interpolation.get_pixel_size(interpolation.get_affine(obs.wcs))
                    if (h is None) or (pix < h):
                        target_wcs = obs.wcs
                        h = pix

            if target_psf is None:
                psf_size = None
                for obs in observations:
                    # Finds the sharpest PSF (might not be on the sharpest grid though)
                    for psf in obs.psfs.image:
                        size = get_psf_size(psf)*interpolation.get_pixel_size(interpolation.get_affine(obs.wcs))
                        if (psf_size is None) or size < psf_size:
                            psf_size = size
                            psf_h = interpolation.get_pixel_size(interpolation.get_affine(obs.wcs))
                            target_psf = PSF(psf[np.newaxis, :, :])
                            print('zizi', obs.channels, size)
                if psf_h > interpolation.get_pixel_size(interpolation.get_affine(target_wcs)):
                    coord_lr = [range(target_psf.shape[-2]), range(target_psf.shape[-1])]
                    ny_hr, nx_hr = target_psf.shape[-2]*psf_h/h, target_psf.shape[-1]*psf_h/h
                    if (ny_hr % 2) == 0:
                        ny_hr += 1
                    if (nx_hr % 2) == 0:
                        nx_hr += 1
                    coord_hr = [range(ny_hr), range(nx_hr)]
                    angle, h = interpolation.get_angles(target_wcs, obs.wcs)
                    target_psf = PSF(interpolation.sinc_interp(target_psf, coord_hr, coord_lr, angle = angle))
                    print('zezette')

        else:
            # Sets frame properties from obs_id using sharpest psf in observation `obs_id`
            assert obs_id in range(len(observations)), '`obs_id` should be an integer between 0 and `len(observations)`'
            target_obs = observations[obs_id]
            target_wcs = target_obs.wcs.cdelt[-1]
            for psf in target_obs.psfs.image:
                size = get_psf_size(psf)
                if (psf_size is None) or (size < psf_size):
                    psf_size = size
                    target_psf = PSF(psf[np.newaxis, :, :])

        # Matching observations together so as to create a common frame
        fat_psf = None
        for c, obs in enumerate(observations):
            if (obs.wcs is not target_wcs) and (type(obs) is not 'LowResObservation'):
                observations[c] = obs.make_LowRes()



        shape = (6,100,200)
        print(target_wcs, target_psf.shape, channels)
        return Frame(shape, wcs=target_wcs, psfs=target_psf, channels=channels)


def get_boundaries(frame_wcs, observation):
    """ Extracts the boundaries of an observation relative to a given frame

    Parameters:
        frame_wcs: wcs
            reference frame for matching coordinates
        observations: `Observation` object
            observation which boundaries we want to extract.
    """


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
    psf /= np.max(psf)

    # Pixels in the FWHM set to one, others to 0:
    psf[psf>0.5] = 1
    psf[psf<=0.5] = 0

    # Area in the FWHM:
    area = np.sum(psf)

    # Diameter of this area
    d = 2*(area/np.pi)**0.5

    # 3-sigma:
    sigma3 = 3*d/(2*(2*np.log(2))**0.5)

    return sigma3
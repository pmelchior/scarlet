import numpy as np
from .psf import PSF
from .bbox import Box
import logging
logger = logging.getLogger("scarlet.frame")


class Frame(Box):
    """Spatial and spectral characteristics of the data

    Attributes
    ----------
    shape: tuple
        (channels, Ny, Nx) shape of the model image
    wcs: TBD
        World Coordinates
    psf: `scarlet.PSF`
        PSF in each band
    channels: list of hashable elements
        Names/identifiers of spectral channels
    dtype: `numpy.dtype`
        Dtype to represent the data.
    """
    def __init__(self, shape_or_box, wcs=None, psf=None, channels=None, dtype=np.float32):

        if isinstance(shape_or_box, Box):
            self = shape_or_box
        else:
            super().__init__(shape_or_box)

        self.wcs = wcs

        if psf is None:
            logger.warning('No PSF specified. Possible, but dangerous!')
            self._psf = None
        else:
            if isinstance(psf, PSF):
                self._psf = psf
            else:
                self._psf = PSF(psf)

        assert channels is None or len(channels) == self.shape[0]
        self.channels = channels
        self.dtype = dtype


    @property
    def psf(self):
        return self._psf

    def get_pixel(self, sky_coord):
        """Get the pixel coordinate from a world coordinate
        If there is no WCS associated with the `Scene`,
        meaning the data frame and model frame are the same,
        then this just returns the `sky_coord`
        """
        if self.wcs is not None:
            if self.wcs.naxis == 3:
                coord = self.wcs.wcs_world2pix(*sky_coord, 0, 0)
            elif self.wcs.naxis == 2:
                coord = self.wcs.wcs_world2pix(*sky_coord, 0)
            else:
                raise ValueError("Invalid number of wcs dimensions: {0}".format(self.wcs.naxis))
            return tuple(int(c.item()) for c in coord)

        return tuple(int(coord) for coord in sky_coord)

    def get_sky_coord(self, pixel):
        """Get the world coordinate for a pixel coordinate
        If there is no WCS associated with the `Scene`,
        meaning the data frame and model frame are the same,
        then this just returns the `sky_coord`
        """
        if self.wcs is not None:
            if self.wcs.naxis == 3:
                coord = self.wcs.wcs_pix2world(*pixel, 0, 0)
            elif self.wcs.naxis == 2:
                coord = self.wcs.wcs_pix2world(*pixel, 0)
            else:
                raise ValueError("Invalid number of wcs dimensions: {0}".format(self.wcs.naxis))
            return tuple(c.item() for c in coord)

        return tuple(pixel)

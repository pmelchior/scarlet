from __future__ import print_function, division
import numpy as np
from functools import partial

import proxmin
from . import transformations
from . import operators

import logging
logger = logging.getLogger("scarlet.source")

class SourceInitError(Exception):
    pass

def get_peak_sed(img, center=None, epsilon=0):
    """Get the SED at position `center` in `img`

    Parameters
    ----------
    img: `~numpy.array`
        (Bands, Height, Width) data array that contains a 2D image for each band
    center: array-like
        (y,x) coordinates of the source in the larger image
    epsilon: float
        Random offset to add to the sed (may be useful with multiple components)

    Returns
    -------
    SED: `~numpy.array`
        SED for a single source
    """
    _y, _x = center
    sed = np.zeros((img.shape[0],))
    sed[:] = img[:,_y,_x]
    if np.all(sed<=0):
        # If the flux in all bands is  <=0,
        # the new sed will be filled with NaN values,
        # which will cause the code to crash later
        msg = "Zero or negative flux at the peak for source at y={0}, x={1}"
        raise SourceInitError(msg.format(_y, _x))
    # ensure proper normalization
    sed = proxmin.operators.prox_unity_plus(sed, 0)
    if epsilon>0:
        sed += np.random.rand(len(sed))*epsilon
        # Normalize again
        sed = proxmin.operators.prox_unity_plus(sed, 0)
    return sed

def get_integrated_sed(img):
    """Calculated the SED by summing the flux in the image in each band
    """
    pass

def init_peak(source, blend, img, epsilon=0):
    """Initialize the source using only the peak pixel

    Parameters
    ----------
    source: `~scarlet.source.Source`
        `Source` to initialize.
    blend: `~scarlet.blend.Blend`
        `Blend` that contains the source.
        This may be necessary for some initialization functions to access parameters
        inside the `Blend` class.
    img: `~numpy.array`
        (Bands, Height, Width) data array that contains a 2D image for each band
    epsilon: float
        Random offset to add to the sed (may be useful with multiple components)

    This implementation initializes the sed from the pixel in
    the center of the frame and sets morphology to only comprise that pixel,
    which works well for point sources and poorly resolved galaxies.

    Returns
    -------
    None.
    But `source.sed` and `source.morph` are set.
    """
    # init with SED of the peak pixels
    # TODO: what should we do if peak is saturated?
    B = img.shape[0]
    cx, cy = source.Nx // 2, source.Ny // 2
    _y, _x = source.center_int
    for k in range(source.K):
        source.sed[k] = get_peak_sed(img, source.center_int, epsilon)
    source.morph = np.zeros((source.K, source.Ny, source.Nx))
    # Turn on a single pixel at the peak
    for k in range(source.K):
        # Make each component's radius one pixel larger
        ymin = max(0, cy-k)
        ymax = min(source.Ny, cy+k+1)
        xmin = max(0, cx-k)
        xmax = min(source.Nx, cx+k+1)
        _ymin = max(0, _y-k)
        _ymax = min(_y+source.Ny, _y+k+1)
        _xmin = max(0,_x-k)
        _xmax = min(_x+source.Nx, _x+k+1)
        source.morph[k, ymin:ymax, xmin:xmax] = img[:,_ymin:_ymax,_xmin:_xmax].sum(axis=0)
    source.morph = source.morph.reshape(source.K, source.Ny*source.Nx)

def init_templates(source, blend, img, symmetric=True, monotonic=True, thresh=0.5):
    """Initialize a single component template

    Parameters
    ----------
    source: `~scarlet.source.Source`
        `Source` to initialize.
    blend: `~scarlet.blend.Blend`
        `Blend` that contains the source.
        This may be necessary for some initialization functions to access parameters
        inside the `Blend` class.
    img: `~numpy.array`
        (Bands, Height, Width) data array that contains a 2D image for each band
    symmetric: `bool`, default=`True`
        Whether or not to make the template symmetric
    monotonic: `bool`, default=`True`
        Whether or not to make the template monotonically decreasing from the peak
    thresh: `float`, default=0.5
        Default fraction of the background RMS to use as the low flux cutoff

    This implementation initializes the sed from the pixel in
    the center of the frame and sets morphology to a template that might be
    symmetric and/or monotonic, depending on the parameters passed to the method.

    Returns
    -------
    None.
    But `source.sed` and `source.morph` are set.
    """
    assert source.K==1
    B = img.shape[0]
    source.sed[0] = get_peak_sed(img, source.center_int)
    _y, _x = source.center_int
    Ny = 2*min(img.shape[1]-_y, _y)-1
    Nx = 2*min(img.shape[2]-_x, _x)-1

    # If the source is on the edge, extend the image with its reflection
    # to model the hidden portion in x and/or y
    if Ny<=1:
        _Ny = img.shape[1]
        Ny = 2*img.shape[1]-1
        _img = np.zeros((img.shape[0], Ny, img.shape[2]))
        _img[:,:_Ny-1] = np.fliplr(img[:,1:])
        _img[:,_Ny-1:] = img[:]
        _y = _Ny - 1
    else:
        _img = img.copy()
    if Nx<=1:
        _Nx = img.shape[2]
        Nx = 2*img.shape[2]-1
        __img = _img.copy()
        _img = np.zeros((_img.shape[0], _img.shape[1], Nx))
        _img[:,:,:_Nx-1] = np.fliplr(__img[:,:,1:])
        _img[:,:,_Nx-1:] = __img
        _x = _Nx - 1
    cx, cy = Nx // 2, Ny // 2

    morph = np.zeros((Ny,Nx))
    # use the band with maximum flux for the source
    band = np.argmax(_img[:,_y,_x])
    morph[:] = _img[band,_y-cy:_y+cy+1,_x-cx:_x+cx+1]/source.sed[0,band]
    morph = morph.reshape((morph.size,))
    morph[morph<0] = 0
    # Apply the appropriate constraints
    if symmetric:
        # Make the model symmetric
        symmetric = morph[::-1]
        morph = np.min([morph, symmetric], axis=0)
    if monotonic:
        # Make the model monotonic
        prox_monotonic = operators.prox_strict_monotonic((Ny, Nx), thresh=0, use_nearest=False)
        morph = prox_monotonic(morph.reshape(morph.size,), 0)
    # Trim the source to set the new size
    morph = morph.reshape(Ny,Nx)
    cutoff = blend._bg_rms[band]*thresh
    cuts = morph>cutoff
    # Make sure that the source has at least one source
    # above the cutoff value
    if np.sum(cuts)==0:
        msg = "Source centered at y={0}, x={1} has no flux above the cutoff ({2})"
        raise SourceInitError(msg.format(_y, _x, cutoff))

    ypix, xpix = np.where(cuts)
    if len(ypix)==0:
        ypix, xpix = np.where(morph>0)
    Ny = np.max(ypix)-np.min(ypix)
    Nx = np.max(xpix)-np.min(xpix)
    Ny += 1 - Ny % 2
    Nx += 1 - Nx % 2
    _cx = Nx//2
    _cy = Ny//2
    morph = morph[cy-_cy:cy+_cy+1, cx-_cx:cx+_cx+1]
    source.resize([Ny, Nx])
    source.morph[0] = morph.reshape((1,morph.size))

def init_bulge_disk(source, blend, img, symmetric=True, monotonic=True, thresh=0.5,
                    color_offset=0.02, disk_ratio=0.5):
    """Initialize a Bulge-Disk Model

    Parameters
    ----------
    source: `~scarlet.source.Source`
        `Source` to initialize.
    blend: `~scarlet.blend.Blend`
        `Blend` that contains the source.
        This may be necessary for some initialization functions to access parameters
        inside the `Blend` class.
    img: `~numpy.array`
        (Bands, Height, Width) data array that contains a 2D image for each band
    symmetric: bool, default=`True`
        Whether or not to make the components symmetric
    monotonic: bool, default=`True`
        Whether to make the components monotonically decreasing from the center
    thresh: `float`, default=0.5
        Default fraction of the background RMS to use as the low flux cutoff
    color_offset: float, default=`0.1`
        The disk is made bluer by subtracting a line from the initial SED,
        where the bluest SED is increased by `disk_offset` and the reddest
        SED in decreased by `disk_offset`.
    disk_ratio: float, default=`0.5`
        Ratio of the bulge size over the disk size, so a `disk_ratio` of 0.5 (default)
        means the disk is twice the size of the bulge.

    This implementation initializes the sed from the pixel in
    the center of the frame and sets morphology to a template that might be
    symmetric and/or monotonic, depending on the parameters passed to the method.

    Returns
    -------
    None.
    But `source.sed` and `source.morph` are set.
    """
    assert source.K==2
    B = img.shape[0]
    _y, _x = source.center_int
    Ny = 2*min(img.shape[1]-_y, _y)-1
    Nx = 2*min(img.shape[2]-_x, _x)-1

    # Initialize the bulge SED
    source.sed[0] = get_peak_sed(img, source.center_int)
    # Make the disk bluer
    disk_sed = np.linspace(-color_offset, color_offset, len(source.sed[0]))
    disk_sed = source.sed[0]-disk_sed
    source.sed[1] = proxmin.operators.prox_unity_plus(disk_sed, 0)

    # If the source is on the edge, extend the image with its reflection
    # to model the hidden portion in x and/or y
    if Ny<=1:
        _Ny = img.shape[1]
        Ny = 2*img.shape[1]-1
        _img = np.zeros((img.shape[0], Ny, img.shape[2]))
        _img[:,:_Ny-1] = np.fliplr(img[:,1:])
        _img[:,_Ny-1:] = img[:]
        _y = _Ny - 1
    else:
        _img = img.copy()
    if Nx<=1:
        _Nx = img.shape[2]
        Nx = 2*img.shape[2]-1
        __img = _img.copy()
        _img = np.zeros((_img.shape[0], _img.shape[1], Nx))
        _img[:,:,:_Nx-1] = np.fliplr(__img[:,:,1:])
        _img[:,:,_Nx-1:] = __img
        _x = _Nx - 1
    cx, cy = Nx // 2, Ny // 2

    morph = np.zeros((Ny,Nx))
    # use the band with maximum flux for the source
    band = np.argmax(_img[:,_y,_x])
    morph[:] = _img[band,_y-cy:_y+cy+1,_x-cx:_x+cx+1]/source.sed[0,band]
    morph = morph.reshape((morph.size,))
    morph[morph<0] = 0
    # Apply the appropriate constraints
    if symmetric:
        # Make the model symmetric
        symmetric = morph[::-1]
        morph = np.min([morph, symmetric], axis=0)
    if monotonic:
        # Make the model monotonic
        prox_monotonic = operators.prox_strict_monotonic((Ny, Nx), thresh=0, use_nearest=False)
        morph = prox_monotonic(morph.reshape(morph.size,), 0)
    # Trim the source to set the new size
    morph = morph.reshape(Ny,Nx)
    
    cutoff = blend._bg_rms[band]*thresh
    cuts = morph>cutoff
    # Make sure that the source has at least one source
    # above the cutoff value
    if np.sum(cuts)==0:
        msg = "Source centered at y={0}, x={1} has no flux above the cutoff ({2})"
        raise SourceInitError(msg.format(_y, _x, cutoff))

    ypix, xpix = np.where(cuts)
    Ny = np.max(ypix)-np.min(ypix)
    Nx = np.max(xpix)-np.min(xpix)
    Ny += 1 - Ny % 2
    Nx += 1 - Nx % 2
    _cx = Nx//2
    _cy = Ny//2
    morph = morph[cy-_cy:cy+_cy+1, cx-_cx:cx+_cx+1]
    source.resize([Ny, Nx])
    # Make the bulge size smaller
    # TODO: improve this algorithm
    x = np.arange(Nx)
    y = np.arange(Ny)
    X,Y = np.meshgrid(x,y)
    X = X - _cx
    Y = Y - _cy
    distance = np.sqrt(X**2+Y**2)
    _morph = np.zeros_like(morph)
    cut = distance<min(Ny*disk_ratio/2,Nx*disk_ratio/2)
    _morph[cut] = morph[cut]*2/3
    source.morph[0] = _morph.reshape((1,_morph.size))
    _morph = morph-_morph
    if monotonic:
        prox_monotonic = operators.prox_strict_monotonic((Ny, Nx), thresh=0, use_nearest=False)
        _morph = prox_monotonic(_morph.reshape(_morph.size,), 0)
    source.morph[1] = _morph.reshape((1,_morph.size))

class Source(object):
    """A single source in a blend
    """
    def __init__(self, center, shape, K=1, psf=None, constraints=None, fix_sed=False, fix_morph=False,
                 fix_frame=False, shift_center=0.2, init_func=init_templates):
        """Constructor

        Parameters
        ----------
        center: array-like
            (y,x) coordinates of the source in the larger image
        shape: tuple
            Shape (B, Ny, Nx) of the frame that contains the source.
            This can be (and usually is) smaller than the size of the full blend
        K: int, default='1'
            Number of components with the same center position
        psf: array-like or `~scarlet.transformations.GammaOp`, default=`None`
            2D image of the psf in a single band (Height, Width),
            or 2D image of the psf in each band (Bands, Height, Width),
            or `~scarlet.transformations.GammaOp` created from a psf array.
        constraints: dict or list of dicts, default=`None`
            Each key in `constraints` contains any parameters
            (such as a treshold for "l0") needed by the proximal operator.
            If K>1, a list of dicts can be given, one for each component.
        fix_sed: bool or list of bools, default=`False`
            Whether or not the SED is fixed, or can be updated
            If K>1, a list of bools can be given, one for each component.
        fix_morph: bool or list of bools, default=`False`
            Whether or not the morphology is fixed, or can be updated
            If K>1, a list of bools can be given, one for each component.
        fix_frame: bool or list of bools, default=`False`
            Whether or not the frame dimensions are fixed, or can be updated
            If K>1, a list of bools can be given, one for each component.
        shift_center: float, default=0.2
            Amount to shift the differential image in x and y to fit
            changes in position.
        init_func: func, default=`init_templates`
            Function to initialize all components of the source.
        """

        # set size of the source frame
        assert len(shape) == 3
        self.B = shape[0]
        size = shape[1:]
        self._set_frame(center, size)
        size = (self.Ny, self.Nx)
        self.fix_frame = fix_frame

        # create containers
        self.K = K
        self.sed = np.zeros((self.K, self.B))
        self.morph = np.zeros((self.K, self.Ny*self.Nx))

        # set up psf and translations matrices
        if isinstance(psf, transformations.GammaOp):
            self._gammaOp = psf
        else:
            self._gammaOp = transformations.GammaOp(self.shape, psf=psf)

        # set center coordinates and translation operators
        # needs to have GammaOp set up first
        self.set_center(center)
        self.shift_center = shift_center

        # updates for sed or morph?
        if hasattr(fix_sed, '__iter__') and len(fix_sed) == self.K:
            self.fix_sed = fix_sed
        else:
            self.fix_sed = [fix_sed] * self.K
        if hasattr(fix_morph, '__iter__') and len(fix_morph) == self.K:
            self.fix_morph = fix_morph
        else:
            self.fix_morph = [fix_morph] * self.K

        # set sed and morph constraints
        self.set_constraints(constraints)
        self.init_func = init_func

    @property
    def Nx(self):
        """Width of the frame
        """
        return self.right-self.left

    @property
    def Ny(self):
        """Height of the frame
        """
        return self.top - self.bottom

    @property
    def shape(self):
        """Shape of the source image (Band, Height, Width)
        """
        return (self.B, self.Ny, self.Nx)

    @property
    def image(self):
        """Reshaped morphology into an array of 2D images for each component
        """
        morph_shape = (self.K, self.Ny, self.Nx)
        return self.morph.reshape(morph_shape) # this *should* be a view

    @property
    def center_int(self):
        """Rounded (not truncated) integer pixel position of the center
        """
        return np.round(self.center).astype('int')

    @property
    def has_psf(self):
        """Whether the source has a psf
        """
        return self._gammaOp.psf is not None

    def get_slice_for(self, im_shape):
        """Return the slice of the source frame in the full multiband image

        In other words, return the slice so that
        self.image[k][slice] corresponds to image[self.bb],
        where image has shape (Band, Height, Width).

        Parameters
        ----------
        im_shape: tuple
            Shape of the full image
        """
        NY, NX = im_shape[1:]

        left = max(0, -self.left)
        bottom = max(0, -self.bottom)
        right = self.Nx - max(0, self.right - NX)
        top = self.Ny - max(0, self.top - NY)
        return (slice(None), slice(bottom, top), slice(left, right))

    def get_model(self, combine=True, Gamma=None, use_sed=True):
        """Get the model of all components for the current source

        Parameters
        ----------
        combine: bool, default=`True`
            Whether or not to combine all of the components into a single model
        Gamma: `~scarlet.transformations.GammaOp`, default=`None`
            Gamma transformation to convolve with PSF and perform linear transform.
            If `Gamma` is `None` then `self.Gamma` is used.
        use_sed: bool, default=`True`
            Whether to use the SED to create a multi-color model or model of just the morphology

        Returns
        -------
        model: `~numpy.array`
            (Bands, Height, Width) image of the model
        """
        if Gamma is None:
            Gamma = self.Gamma
        if use_sed:
            sed = self.sed
        else:
            sed = np.ones_like(self.sed)
        # model for all components of this source
        if not self.has_psf:
            model = np.empty((self.K, self.B, self.Ny*self.Nx))
            for k in range(self.K):
                model[k] = np.outer(sed[k], Gamma.dot(self.morph[k]))
        else:
            model = np.zeros((self.K, self.B, self.Ny*self.Nx))
            for k in range(self.K):
                for b in range(self.B):
                    model[k,b] += sed[k,b] * Gamma[b].dot(self.morph[k])

        # reshape the image into a 2D array
        model = model.reshape(self.K, self.B, self.Ny, self.Nx)

        if combine:
            model = model.sum(axis=0)
        return model

    def _set_frame(self, center, size):
        """Create a frame and bounding box

        To save memory and computation time, each source is contained in a small
        subset of the entire blended image. This method takes the coordinates of
        the source and the size of the frame and creates a bonding box (`self.bb`).

        Parameters
        ----------
        center: array-like
            (y,x) coordinates of the center of the source in the full image
        size: float or array-like
            Either a (height,width) shape or a single size to create a
            square (size,size) frame.

        Returns
        -------
        None.
        But it defines `self.bottom`, `self.top`, `self.left`, `self.right` as
        the edges of the frame, `self.bb` as the slices (bounding box) containing the frame,
        and `self.center` as the center of the frame.

        """
        assert len(center) == 2
        self.center = np.array(center)
        if hasattr(size, '__iter__'):
            size = size[:2]
        else:
            size = (size,) * 2
        # make cutout of in units of the original image frame (that defines xy)
        # ensure odd pixel number
        y_, x_ = self.center_int
        self.bottom, self.top = y_ - int(size[0]//2), y_ + int(size[0]//2) + 1
        self.left, self.right = x_ - int(size[1]//2), x_ + int(size[1]//2) + 1

        # since slice wrap around if start or stop are negative, need to sanitize
        # start values (stop always postive)
        self.bb = (slice(None), slice(max(0, self.bottom), self.top), slice(max(0, self.left), self.right))

    def set_center(self, center):
        """Given a (y,x) `center`, update the frame and `Gamma`
        """
        if not hasattr(self, '_init_center'):
            self._init_center = np.array([center[0], center[1]])
        size = (self.Ny, self.Nx)
        self._set_frame(center, size)

        # update translation operator
        dx = self.center - self.center_int
        self.Gamma = self._gammaOp(dx, self.shape)

    def resize(self, size):
        """Resize the frame

        Set the new frame size and update relevant parameters like the morphology,
        Gamma matrices, and constraint operators.

        Parameters
        ----------
        size: float or array-like
            Either a (height,width) shape or a single size to create a
            square (size,size) frame.
        """
        # store old edge coordinates
        top, right, bottom, left = self.top, self.right, self.bottom, self.left
        self._set_frame(self.center, size)

        # check if new size is larger or smaller
        new_slice_y = slice(max(0, bottom - self.bottom),
                            min(self.top - self.bottom, self.top - self.bottom - (self.top - top)))
        old_slice_y = slice(max(0, self.bottom - bottom), min(top - bottom, top - bottom - (top - self.top)))
        if top-bottom == self.Ny:
            new_slice_y = old_slice_y = slice(None)
        new_slice_x = slice(max(0, left - self.left),
                            min(self.right - self.left, self.right - self.left - (self.right - right)))
        old_slice_x = slice(max(0, self.left - left), min(right - left, right - left - (right - self.right)))
        if right-left == self.Nx:
            new_slice_x = old_slice_x = slice(None)
        new_slice = (slice(None), new_slice_y, new_slice_x)
        old_slice = (slice(None), old_slice_y, old_slice_x)

        if new_slice != old_slice:
            # change morph
            _morph = self.morph.copy().reshape((self.K, top-bottom, right-left))
            self.morph = np.zeros((self.K, self.Ny, self.Nx))

            self.morph[new_slice] = _morph[old_slice]
            self.morph = self.morph.reshape((self.K, self.Ny*self.Nx))

            # update GammaOp and center (including subpixel shifts)
            self.set_center(self.center)

            # set constraints
            self.set_constraints(self.constraints)

    def init_source(self, blend, img):
        """Initialize the source
        Parameters
        ----------
        blend: `~scarlet.blend.Blend`
            `Blend` that contains the source.
            This may be necessary for some initialization functions to access parameters
            inside the `Blend` class.
        img: `~numpy.array`
            (Bands, Height, Width) data array that contains a 2D image for each band

        Call `init_func` with the current blend parameters

        Returns
        -------
        None.
        But `self.sed` and `self.morph` are set.
        """
        self.init_func(self, blend, img)

    def get_morph_error(self, weights):
        """Get error in the morphology

        This error estimate uses linear error propagation and assumes that the
        source was isolated (it ignores blending).

        CAVEAT: If the source has a PSF, the inversion of the covariance matrix
        is likely instable.

        Parameters
        ----------
        weights: `~numpy.array`
            Weights of the images in each band (Bands, Height, Width).

        Returns
        -------
        me: `~numpy.array`
            Error in morphology for each pixel
        """
        w = np.zeros(self.shape)
        w[self.get_slice_for(weights.shape)] = weights[self.bb]
        w = w.reshape(self.B, -1)
        # prevent zeros from messing up:
        # set them at a very small value, and zero them out at the end
        mask = (w.sum(axis=0) == 0).flatten()
        if mask.sum():
            w[:,mask] = 1e-3 * w[:,~mask].min(axis=1)[:,None]

        # compute direct error propagation assuming only this source SED(s)
        # and the pixel covariances: Sigma_morph = diag((A^T Sigma^-1 A)^-1)
        # CAVEAT: If done on the entire A matrix, degeneracies in the linear
        # solution arise and substantially amplify the error estimate:
        # Instead, estimate noise for each component separately:
        # simple multiplication for diagonal pixel covariance matrix
        if not self.has_psf:
            me = [1./np.sqrt(np.dot(a.T, np.multiply(w, a[:,None]))) for a in self.sed]
        else:
            # see Blend.steps_f for details for the complete covariance matrix
            import scipy.sparse
            Sigma_pix = scipy.sparse.diags(w.flatten(), 0)
            PA = [scipy.sparse.bmat([[self.sed[k,b] * self.Gamma[b]] for b in range(self.B)])
                    for k in range(self.K)]
            Sigma_s = [PAk.T.dot(Sigma_pix.dot(PAk)) for PAk in PA]
            me = [np.sqrt(np.diag(np.linalg.inv(Sigma_sk.toarray()))) for Sigma_sk in Sigma_s]

            # TODO: the matrix inversion is instable if the PSF gets wide
            # possible options: Tikhonov regularization or similar
        if mask.sum():
            for mek in me:
                mek[mask] = 0
        return me

    def get_sed_error(self, weights):
        """Get error in the SED's

        This error estimate uses linear error propagation and assumes that the
        source was isolated (it ignores blending).

        Parameters
        ----------
        weights: `~numpy.array`
            Weights of the images in each band (Bands, Height, Width).

        Returns
        -------
        error: `~numpy.array`
            Estimated error in the SED.
        """
        w = np.zeros(self.shape)
        w[self.get_slice_for(weights.shape)] = weights[self.bb]
        w = w.reshape(self.B, -1)
        # NOTE: zeros weights would only be a problem if an entire band is missing

        # See explanation in get_morph_error and Blend.steps_f
        if not self.has_psf:
            return [1./np.sqrt(np.dot(s,np.multiply(w.T, s[None,:].T))) for s in self.morph]
        else:
            import scipy.sparse
            Sigma_pix = scipy.sparse.diags(w.flatten(), 0)
            model = self.get_model(combine=False, use_sed=False)
            PS = [scipy.sparse.block_diag([model[k,b,:,:].reshape((1,-1)).T for b in range(self.B)])
                        for k in range(self.K)]
            return [np.sqrt(np.diag(np.linalg.inv(PSk.T.dot(Sigma_pix.dot(PSk)).toarray()))) for PSk in PS]

    def set_constraints(self, constraints):
        """Set the constraints for each component in the source

        Parameters
        ----------
        constraints: dict, or list of dicts
            Each key in `constraints` contains any parameters
            (such as a treshold for "l0") needed by the proximal operator.
            If a list of dicts is given, each elements refers to one component.

        Returns
        -------
        None.
        But `self.constraints`, `self.progs_g` (ADMM-like proximal operators),
        `self.Ls` (linear matrices for each proxs_g),
        `self.prox_sed` (prox_f proximal operator for the SED's),
        and `self.prox_morph` (prox_f for morphologies) are set.
        """
        if constraints is None:
            self.constraints = {}
        else:
            self.constraints = constraints
        # one set of constraints for each component
        if not isinstance(self.constraints, list):
            self.constraints = [self.constraints,] * self.K

        self.proxs_g_A = [None] * self.K
        self.proxs_g_S = [[],] * self.K
        self.LA = [None] * self.K
        self.LS = [[],] * self.K

        if self.constraints is None:
            self.proxs_g_A = [None] * self.K
            self.proxs_g_S = [None] * self.K
            self.LA = [None] * self.K
            self.LS = [None] * self.K
            return

        self.prox_sed = [proxmin.operators.prox_unity_plus] * self.K
        self.prox_morph = [[],] * self.K

        # superset of all constraint keys (in case they are different)
        keys = set(self.constraints[0].keys())
        for k in range(1, self.K):
            keys |= set(self.constraints[k].keys())
        shape = (self.Ny, self.Nx)
        # because of the constraint matrices being costly to construct:
        # iterate over keys, not over components
        for c in keys:

            if c =="+":
                for k in range(self.K):
                    if c in self.constraints[k].keys():
                        # with positivity for positive source, the center
                        # needs to have some flux.
                        self.prox_morph[k].append(operators.prox_center_on)
                        self.prox_morph[k].append(proxmin.operators.prox_plus)

            # Note: don't use hard/soft thresholds with _plus (non-negative) because
            # that is either happening with prox_plus before or is not indended
            if c == "l0":
                for k in range(self.K):
                    if c in self.constraints[k].keys():
                        thresh = self.constraints[k][c]
                        self.prox_morph[k].append(partial(proxmin.operators.prox_hard, thresh=thresh))

            elif c == "l1":
                for k in range(self.K):
                    if c in self.constraints[k].keys():
                        thresh = self.constraints[k][c]
                        self.prox_morph[k].append(partial(proxmin.operators.prox_soft, thresh=thresh))

            elif c == "m":
                for k in range(self.K):
                    if c in self.constraints[k].keys():
                        kwargs = {}
                        if self.constraints[k][c] is not None:
                            kwargs.update(self.constraints[k][c])
                        self.prox_morph[k].append(operators.prox_strict_monotonic(shape, **kwargs))

            elif c == "M":
                # positive gradients
                # NOTE: we're using useNearest from component k=0 only!
                M = transformations.getRadialMonotonicOp(shape, useNearest=self.constraints[0][c])
                for k in range(self.K):
                    if c in self.constraints[k].keys():
                        self.LS[k].append(M)
                        self.proxs_g_S[k].append(proxmin.operators.prox_plus)
                    else:
                        self.LS[k].append(None)
                        self.proxs_g_S[k].append(None)

            elif c == "S":
                # zero deviation of mirrored pixels
                S = transformations.getSymmetryOp(shape)
                for k in range(self.K):
                    if c in self.constraints[k].keys():
                        self.LS[k].append(S)
                        self.proxs_g_S[k].append(proxmin.operators.prox_zero)
                    else:
                        self.LS[k].append(None)
                        self.proxs_g_S[k].append(None)

            elif c == "C":
                # cone method for monotonicity: exact but VERY slow
                useNearest = self.constraints.get("M", False)
                G = transformations.getRadialMonotonicOp(shape, useNearest=useNearest).toarray()
                for k in range(self.K):
                    if c in self.constraints[k].keys():
                        self.proxs_g_S[k].append(partial(operators.prox_cone, G=G))
                    else:
                        self.proxs_g_[k].append(None)
                    self.LS[k].append(None)
            elif c == "X":
                # l1 norm on gradient in X for TV_x
                cx = int(self.Nx)
                Gx = proxmin.transformations.get_gradient_x(shape, cx)
                for k in range(self.K):
                    if c in self.constraints[k].keys():
                        self.LS[k].append(Gx)
                        thresh = self.constraints[k][c]
                        self.proxs_g_S[k].append(partial(proxmin.operators.prox_soft, thresh=thresh))
                    else:
                        self.LS[k].append(None)
                        self.proxs_g_S[k].append(None)
            elif c == "Y":
                # l1 norm on gradient in Y for TV_y
                cy = int(self.Ny)
                Gy = proxmin.transformations.get_gradient_y(shape, cy)
                for k in range(self.K):
                    if c in self.constraints[k].keys():
                        self.LS[k].append(Gy)
                        thresh = self.constraints[k][c]
                        self.proxs_g_S[k].append(partial(proxmin.operators.prox_soft, thresh=thresh))
                    else:
                        self.LS[k].append(None)
                        self.proxs_g_S[k].append(None)

        # with several projection operators in prox_morph:
        # use AlternatingProjections to link them together
        for k in range(self.K):
            # dummy operator: identity
            if len(self.prox_morph[k]) == 0:
                self.prox_morph[k] = proxmin.operators.prox_id
            elif len(self.prox_morph[k]) == 1:
                self.prox_morph[k] = self.prox_morph[k][0]
            else:
                self.prox_morph[k] = proxmin.operators.AlternatingProjections(self.prox_morph[k], repeat=1)

    def remove_component(self, idx, ref_idx=None):
        """Remove a component from the Source

        Parameters
        ----------
        idx: int
            Index of the component in the source to remove
        ref_idx: int, defaul=`None`
            Index of the primary component that `idx` is degenerate with.
            If `ref_idx` is not `None`, the morphology of component `int` is
            added to the morphology of component `ref_idx`.

        Returns
        -------
        None
        """
        # Add the flux from the degenerate component to the primary component
        if ref_idx is not None:
            self.morph[ref_idx] += self.morph[idx]
        # Delete the degenerate morphology and SED
        self.morph = np.delete(self.morph, (idx), axis=0)
        self.sed = np.delete(self.sed, (idx), axis=0)
        # Clear out all of the parameters for the degenerate component
        self.K -= 1
        del self.constraints[idx]
        del self.prox_morph[idx]
        del self.proxs_g_A[idx]
        del self.proxs_g_S[idx]
        del self.LA[idx]
        del self.LS[idx]
        del self.fix_sed[idx]
        del self.fix_morph[idx]

    def remove_degenerate_components(self, sed_diff=1e-5, idx=0):
        """Remove all degenerate components

        Parameters
        ----------
        sed_diff: float, default=1e-5
            Maximum difference between component SED's to consider
            them degenerate.
        idx: int, default=0
            Initial component to begin degenerate search.
            Because the algorithm removes components from the source
            list in place, it is easier to make remove_degenerates a
            recursive function and that is called again after
            degenerate components have been removed.
            To save processing time, we being with the next component
            (`idx`) from the component that just had it's degenerates removed.

        Returns
        -------
        result: bool
            Whether or not any degenerate components were removed
        """
        for l in range(idx, self.K-1):
            degenerates = []
            for ll in range(l+1,self.K):
                diff = np.sum((self.sed[l]-self.sed[ll])**2)
                if diff<sed_diff:
                    degenerates.append(ll)
            # Remove any degenerates for the current source, then restart
            if len(degenerates) > 0:
                for degenerate_idx in degenerates:
                    self.remove_component(degenerate_idx, l)
                logger.warn("Removed degenerate components {0} from source at {1}".format(
                    degenerates, self.center))
                self.remove_degenerate_components(sed_diff, l+1)
                return True
        return False

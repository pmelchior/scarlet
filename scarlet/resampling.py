import autograd.numpy as np
from . import interpolation




def convolve_band(model, psf, padding = 3):
    """Convolve the model in a single band
    """
    psf_fft = np.fft.fft2(np.fft.ifftshift(psf))
    _model = np.pad(model, padding, 'constant')
    model_fft = np.fft.fft2(np.fft.ifftshift(_model))
    convolved_fft = model_fft * psf_fft
    convolved = np.fft.ifft2(convolved_fft)
    result = np.fft.fftshift(np.real(convolved))
  #  (bottom, top), (left, right) = padding
    result = result#[bottom:-top, left:-right]
    return result



def conv2D_fft(shape, xm, ym, p, h, scp):
    '''

    shape:
        xm, ym: coordinate of the low resolution location where to compute mapping
        p: PSF kernel
        h: pixel size
    RETURN:
    ------
        result: vector for convolution and resampling of the high resolution plane into pixel (xm,ym) at low resolution
    '''

    nl, n1, n2 = shape
    ker = np.zeros((n1,n2))
    x,y = np.where(ker == 0)

    ker[x,y] = interpolation.sinc2D((xm-x)/h,(ym-y)/h)

    return scp.fftconvolve(ker, p, mode = 'same')*h/np.pi

def make_operator(shape, coord_lr, p):
    '''
    INPUTS:
    ------
        shape: size (number of pixels) of the high resolution scene
            Type: tuple of 2 numbers
        coord_lr: coordinates of the overlapping pixels from the low resolution grid in the high resolution grid frame.
            Type: tuple of 1-D numpy array. One array fair each dimension of the 2-D plane.
        p: convolution kernel (PSF)
            Type: 2-D numpy array
    OUTPUTS:
    -------
        mat: the convolution-resampling matrix
    '''
    nl,n1,n2 = shape
    a, b = np.where(np.zeros((n1,n2))==0)
    A, B = coord_lr
    mat = np.zeros((n1*n2, B.size))

    h = a[1]-a[0]
    if h == 0:
        h = b[1]-b[0]
    assert h !=0
    import scipy.signal as scp
    for m in range(np.size(B)):
            mat[:, m] = conv2D_fft(shape, A[m], B[m], p, h, scp).flatten()*A.size/a.size
            #mat[:, m] /= np.sum(mat[:,m])

    return mat




def linorm2D(S, nit):
    """
      Estimates the inverse of the Lipschitz constant of a matrix SS.T

      INPUTS:
          A: matrix
          nit: number of iterations

      OUTPUTS:
          xn: float
            inverse of the Lipschitz constant of SS.T

       EXAMPLES

    """

    n1, n2 = np.shape(S)
    x0 = np.random.rand(1, n1)
    x0 = x0 / np.sqrt(np.sum(x0 ** 2))

    for i in range(nit):
        x = np.dot(x0, S)
        xn = np.sqrt(np.sum(x ** 2))
        xp = x / xn
        y = np.dot(xp, S.T)
        yn = np.sqrt(np.sum(y ** 2))

        if yn < np.dot(y, np.t(x0)):
            break
        x0 = y / yn

    return 1./xn


def match_patches(shape_hr, shape_lr, wcs_hr, wcs_lr):

    '''
    Finds the region of overlap between two datasets and creates a mask for the region as well as the pixel coordinates for the dataset pixels inside the overlap.
    INPUTS:
    ------
        shape_hr, shape_lr: cube images for the two datasets to match. The pixel grid from im_hr is used as a reference grid for the combination of both sets.
        wcs_hr, wcs_lr: WCS of the Low and High resolution fields respectively

    RETURN:
    ------
        mask:mask of overlapping pixel in the high resolution frame.
        coordlr_over_lr: coordinates of the overlap in low resolution.
        coordlr_over_hr: coordinates of the overlaps at low resolution in the high resolution frame.
    '''



    if np.size(shape_hr) == 3:
        b,n1,n2 = shape_hr
    elif np.size(shape_hr) == 2:
        n1, n2 = shape_hr
    else:
        raise ValueError('Wrong dimensions for reference image')

    if np.size(shape_lr) == 3:
        B,N1,N2 = shape_lr
    elif np.size(shape_lr) == 2:
        N1, N2 = shape_lr
    else:
        raise ValueError('Wrong dimensions for low resolution image')

    assert wcs_hr != None
    assert wcs_lr != None

    im_hr = np.zeros((n1,n2))
    im_lr = np.zeros((N1,N2))

    # Coordinates of pixels in both frames
    x_hr, y_hr = np.where(im_hr * 0 == 0)
    X_lr, Y_lr = np.where(im_lr * 0 == 0)

    if np.size(wcs_lr.array_shape) == 2:
        ra_lr, dec_lr = wcs_lr.all_pix2world(Y_lr, X_lr, 0, ra_dec_order = True)
    elif np.size(wcs_lr.array_shape) == 3:
        ra_lr, dec_lr = wcs_lr.all_pix2world(Y_lr, X_lr, 0, 0, ra_dec_order = True)
    if np.size(wcs_hr.array_shape) == 2:
        ra_hr, dec_hr = wcs_hr.all_pix2world(x_hr, y_hr, 0, ra_dec_order=True)
    elif np.size(wcs_hr.array_shape) == 3:
        ra_hr, dec_hr = wcs_hr.all_pix2world(x_hr, y_hr, 0, 0, ra_dec_order=True)

    # Coordinates of the low resolution pixels in the high resolution frame
    if np.size(wcs_hr.array_shape) == 2:
        Y_hr, X_hr = wcs_hr.all_world2pix(ra_lr, dec_lr, 0, ra_dec_order = True)
    elif np.size(wcs_hr.array_shape) == 3:
        Y_hr, X_hr = wcs_hr.all_world2pix(ra_lr, dec_lr, 0, 0, ra_dec_order = True)

    # Coordinates of the high resolution pixels in the low resolution frame
    if np.size(wcs_lr.array_shape) == 2:
        y_lr, x_lr = wcs_lr.all_world2pix(ra_hr, dec_hr, 0, ra_dec_order = True)
    # Coordinates of the high resolution pixels in the low resolution frame
    elif np.size(wcs_lr.array_shape) == 3:
        y_lr, x_lr, l = wcs_lr.all_world2pix(ra_hr, dec_hr, 0, 0, ra_dec_order = True)

    # Mask of low resolution pixels in the overlap at low resolution:
    over_lr = ((Y_hr>0) * (Y_hr<n2) * (X_hr>0) * (X_hr<n1))
    # Mask of low resolution pixels in the overlap at high resolution:
    over_hr = ((y_lr>0) * (y_lr<N2) * (x_lr>0) * (x_lr<N1))


    mask = over_hr.reshape(n1,n2)

    if np.sum(mask) == 0:
        raise ValueError('No overlap found between datasets. Check your coordinates and/or WCSs.')

    # Coordinates of low resolution pixels in the overlap at low resolution:
    xlr_over_lr = Y_lr[(over_lr == 1)]
    ylr_over_lr = X_lr[(over_lr == 1)]
    coordlr_over_lr = (ylr_over_lr, xlr_over_lr)
    # Coordinates of low resolution pixels in the overlap at high resolution:
    xlr_over_hr = Y_hr[(over_lr == 1)]
    ylr_over_hr = X_hr[(over_lr == 1)]
    coordlr_over_hr = (ylr_over_hr, xlr_over_hr)



    return mask, coordlr_over_lr, coordlr_over_hr


def match_psfs(psf_hr, psf_lr, wcs_hr, wcs_lr):
    '''
    INPUTS:
    ------
        psf_hr: centered psf of the high resolution scene
        psf_lr: centered psf of the low resolution scene
        wcs_hr: wcs of the high resolution scene
        wcs_lr: wcs of the low resolution scene
    OUTPUTS:
    -------
        psf_match_hr: high rresolution psf at mactching size
        psf_match_lr: low resolution psf at matching size and resolution
    '''

    nhr1, nhr2 = psf_hr.shape
    nlr1, nlr2 = psf_lr.shape
    if np.size(wcs_hr.array_shape) == 2:
        wcs_hr.wcs.crval = 0., 0.
        wcs_hr.wcs.crpix = nhr1 / 2., nhr2 / 2.
    elif np.size(wcs_hr.array_shape) == 3:
        wcs_hr.wcs.crval = 0., 0., 0.
        wcs_hr.wcs.crpix = nhr1 / 2., nhr2 / 2., 0.
    if np.size(wcs_lr.array_shape) == 2:
        wcs_lr.wcs.crval = 0., 0.
        wcs_lr.wcs.crpix = nlr1 / 2., nlr2 / 2.
    elif np.size(wcs_lr.array_shape) == 3:
        wcs_lr.wcs.crval = 0., 0., 0.
        wcs_lr.wcs.crpix = nlr1 / 2., nlr2 / 2., 0


    mask, p_lr, p_hr = match_patches(psf_hr.shape, psf_lr.data.shape, wcs_hr, wcs_lr)

    cmask = np.where(mask == 1)

    n_p = np.int((np.size(cmask[0]))**0.5)
    psf_match_lr = interpolation.sinc_interp(cmask, p_hr[::-1], (psf_lr).flatten()).reshape(n_p, n_p)

    psf_match_hr = psf_hr[np.int((nhr1-n_p)/2):np.int((nhr1+n_p)/2),np.int((nhr2-n_p)/2):np.int((nhr2+n_p)/2)]

    psf_match_hr /= np.max(psf_match_hr)
    psf_match_lr /= np.max(psf_match_lr)
    return psf_match_hr, psf_match_lr
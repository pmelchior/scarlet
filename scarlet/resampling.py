import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as scp
from . import interpolation



def conv2D_fft(shape, xm, ym, p, h):
    '''

    shape:
        xm, ym: coordinate of the low resolution location where to compute mapping
        p: PSF kernel
        h: pixel size
    RETURN:
    ------
        result: vector for convolution and resampling of the high resolution plane into pixel (xm,ym) at low resolution
    '''


    ker = np.zeros((shape[0], shape[1]))
    x,y = np.where(ker == 0)

    ker[x,y] = sinc2D((xm-x)/h,(ym-y)/h)

    return scp.fftconvolve(ker, p, mode = 'same')*h/np.pi

def make_mat(shape, coord_hr, coord_lr, p):
    '''
    INPUT:
    -----
    shape: size of the whole scene
    coord_hr: coordinates of overlapping pixels in the reference frame
    coord_lr: coordinates of overlapping pixels in the low resolution frame
    p: convolution kernel (PSF)
    RETURN:
    ------
         mat: the convolution-resampling matrix
    '''
    a, b = coord_hr
    A, B = coord_lr
    mat = np.zeros((a.size, B.size))

    h = a[1]-a[0]
    if h == 0:
        h = b[1]-b[0]
    assert h !=0

    for m in range(np.size(B)):
            mat[:, m] = conv2D_fft(shape, A[m], B[m], p, h)[a,b]#.flatten()
            mat[:, m] /= np.sum(mat[:,m])

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

        if yn < np.dot(y, x0.T):
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
        mask:
        coord_hr_model: coordinates of overlapping pixels in the model frame
        coord_hr_over: coordinates of overlapping pixels int he high resolution frame
        coord_lr_over: coordinates of overlapping pixels in the low resolution frame
        coord_lr_over: low resolution coordinates of overlapping pixels in the high resolution frame
    '''

    #shapes


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

    im_hr = np.zeros((n1,n2))
    im_lr = np.zeros((N1,N2))

    # Coordinates of pixels in both frames
    x_hr, y_hr = np.where(im_hr * 0 == 0)
    X_lr, Y_lr = np.where(im_lr * 0 == 0)


    ra_lr, dec_lr = wcs_lr.all_pix2world(Y_lr, X_lr, 0)

    # Coordinates of the low resolution pixels in the high resolution frame
    Y_hr, X_hr = wcs_hr.all_world2pix(ra_lr, dec_lr,0)

    # Limits of the frame encapsulating both datasets
    frame_lim = np.array([np.min(np.concatenate((x_hr,X_hr))), np.max(np.concatenate((x_hr, X_hr))),
                 np.min(np.concatenate((y_hr, y_hr))), np.max(np.concatenate((y_hr, Y_hr)))]).astype(int)

    # Model frame: Grid for the whole frame encapsulating both observations
    frame = np.zeros((frame_lim[1]-frame_lim[0]+1, frame_lim[3]-frame_lim[2]+1))
    s1, s2 = frame.shape

    # Coordinates of the model grid in the high resolution frame
    yf = np.linspace(frame_lim[0], frame_lim[1], frame_lim[1]-frame_lim[0]+1)
    xf = np.linspace(frame_lim[2], frame_lim[3], frame_lim[3]-frame_lim[2]+1)
    xf, yf = np.meshgrid(xf, yf)
    xf = xf.flatten().astype(int)
    yf = yf.flatten().astype(int)

    # Ra,Dec positions of the pixels in the model frame
    raf, decf = wcs_hr.all_pix2world(yf, xf, 0)
    # Coordinates of the model grid in the low resolution frame
    yf_lr, xf_lr = wcs_lr.all_world2pix(raf, decf, 0)

    ''' Location of pixels in the reference frame that are in the overlap (boolean)'''
    loc = (yf<n2)*(yf>=0)*(xf<n1)*(xf>=0)*(yf_lr<N2)*(yf_lr>=0)*(xf_lr<N1)*(xf_lr>=0)


    # Mask of overlapping regions
    frame = loc.reshape(s1,s2)
    x, y = np.where(frame*0 == 0)


    ''' pixel coordinates of the overlap in each dataset '''
    # Coordinates of the overlapping pixels in the model grid
    coord_hr_model = np.where(frame == 1)


    # Coordinates of the overlapping high resolution pixels in the high resolution frame
    xhr_over = xf[frame[x.astype(int), y.astype(int)] == 1]
    yhr_over = yf[frame[x.astype(int), y.astype(int)] == 1]

    coord_hr_over = (xhr_over, yhr_over)
    # Re, Dec coordinate of overlapping pixels at high resolution
    ra_over, dec_over = wcs_hr.all_pix2world(yhr_over, xhr_over, 0)

    # Coordinates of the overlapping pixels at high resolution in the low resolution frame
    # (useless but we need them to find low resolution overlap)
    ylr_over, xlr_over = wcs_lr.all_world2pix(ra_over, dec_over, 0)

    # Overlap region in the low resolution frame
    im_lr[xlr_over.astype(int), ylr_over.astype(int)] = 1

    # Coordinate of overlapping pixels at low resolution in the low resolution frame
    coord_lr_over = np.where(im_lr == 1)


    # Sanity check: low resolution pixels in high resolution frame
    RA_over, DEC_over = wcs_lr.all_pix2world(coord_lr_over[1], coord_lr_over[0], 0)

    Y_over, X_over = wcs_hr.all_world2pix(RA_over, DEC_over, 0)

    coord_lr_hr = X_over, Y_over

    return frame, coord_hr_model, coord_hr_over, coord_lr_over, coord_lr_hr
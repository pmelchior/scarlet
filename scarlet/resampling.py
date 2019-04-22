import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as scp
from . import interpolation
import warnings
warnings.simplefilter("ignore")



def Image(f, a, b):
    '''creates a 2-D array from 1-D array with coordinates (a,b)

    Parameters:
    ----------
    f: function that takes 2 1-D arrays of the same length as input
    a,b: numpy 1-D arrays
        coordinates of point at wich f is evaluated

    Returns:
    -------
    Img: numpy array
        a 2-D array filled with values of f(a,b)
    '''
    n1 = np.int(np.sqrt(a.size))
    n2 = np.int(np.sqrt(b.size))
    Img = np.zeros((n1, n2))
    xgrid, ygrid = np.where(np.zeros((n1, n2)) == 0)

    Img[xgrid, ygrid] = f(a, b)
    return Img


def conv2D_fft(xk, yk, xm, ym, p):
    '''Computes a line of the resampling-convolution matrix at position (xm, ym) in the low resolution frame


     Parameters:
     ----------
     xk, yk: 1-D numpy arrays
        positions in 2-D of the samples in the high resolution frame
     xm, ym: ints
        position of the low resolution frame for which the resampling is computed
     p: numpy 2-D arrray
        samples of the psf

     Returns:
     -------
     Result: 2-D numpy array of the size of xk and yx
        interpolated samples of Fm at positions (a,b)

     '''
    assert xk.size == yk.size

    h = xk[1]-xk[0]
    assert h != 0

    #Resampling kernel
    ker = np.zeros((np.int(xk.size**0.5), np.int(yk.size**0.5)))
    x,y = np.where(ker == 0)
    ker[x,y] = interpolation.sinc2D((xm-xk)/h,(ym-yk)/h)

    return scp.fftconvolve(ker, p, mode = 'same')*h/np.pi

def make_mat2D_fft(a, b, A, B, p):
    '''Creates the resampling-convolution matrix

     Parameters:
     ----------
     a, b: 1-D numpy arrays
        positions of the high resolution samples, which is also the sampling of the model grid
     A, B: numpy 1-D arrays
        positions of the low resolution samples
     p: numpy 2-D array
        samples of the psf

     Returns:
     -------
     mat: numpy 2-D array
        matrix for resampling and convolution.

     '''
    mat = np.zeros((a.size, B.size))
    h = a[1]-a[0]
    assert h!=0
    t0 = time.clock()
    for m in range(np.size(B)):
            mat[:, m] = conv2D_fft(a, b, A[m], B[m], p)
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

def match_patches(x0,y0,WLR, WHR, excess):

    '''
    :param x0, y0: coordinates of the center of the patch in High Resolutions pixels
    :param WLR, WHR: WCS of the Low and High resolution fields respectively
    :param excess: half size of the box
    :return:
    x_HR, y_HR: pixel coordinates of the grid for the High resolution patch
    X_LR, Y_LR: pixel coordinates of the grid for the Low resolution grid
    X_HR, Y_HR: pixel coordinates of the Low resolution grid in units of the High resolution patch
    '''

    xstart = x0 - excess
    xstop = x0 + excess
    ystart = y0 - excess
    ystop = y0 + excess

    XX = np.linspace(xstart, xstop, xstop-xstart + 1)
    YY = np.linspace(ystart, ystop, xstop-xstart + 1)

    x, y = np.meshgrid(XX, YY)
    x_HR = x.flatten().astype(int) + 0.5
    y_HR = y.flatten().astype(int) + 0.5

    # LR coordinates

    Ramin, Decmin = WHR.wcs_pix2world(ystart, xstart, 0)
    Ramax, Decmax = WHR.wcs_pix2world(ystop, xstop, 0)
    Ymin, Xmin = WLR.wcs_world2pix(Ramin, Decmin, 0)
    Ymax, Xmax = WLR.wcs_world2pix(Ramax, Decmax, 0)

    X = np.linspace(np.int(Xmin), np.int(Xmax), np.int(Xmax)-np.int(Xmin)+1)
    Y = np.linspace(np.int(Ymin), np.int(Ymax), np.int(Ymax)-np.int(Ymin)+1)

    X, Y = np.meshgrid(X, Y)
    X_LR = X.flatten().astype(int) + 0.5
    Y_LR = Y.flatten().astype(int) + 0.5
    Ra_LR, Dec_LR = WLR.wcs_pix2world(Y_LR, X_LR, 0)  # type:
    Y_HR, X_HR = WHR.wcs_world2pix(Ra_LR, Dec_LR, 0)

    return x_HR, y_HR, X_LR, Y_LR, X_HR, Y_HR

def make_patches(x_HR, y_HR, X_LR, Y_LR, Im_HR, Im_LR):
    '''
    :param x_HR, y_HR: Coordinates of the High resolution grid
    :param X_LR, Y_LR: Coordinates of the Low resolution grid
    :param Im_HR: High resolution FoV
    :param Im_LR: Low resolution FoV
    :return: Patch_HR, Patch_LR
    '''

    x_HR = x_HR.astype(int)
    y_HR = y_HR.astype(int)
    X_LR = X_LR.astype(int)
    Y_LR = Y_LR.astype(int)


    N1 = np.max(X_LR)-np.min(X_LR)+1
    N2 = np.max(Y_LR)-np.min(Y_LR)+1

    n1 = np.max(x_HR)-np.min(x_HR)+1
    n2 = np.max(y_HR)-np.min(y_HR)+1

    cut_HR = Im_HR[x_HR, y_HR].reshape(n2,n1)
    cut_LR = Im_LR[X_LR, Y_LR].reshape(N2,N1)

    cut_HR /= np.sum(cut_HR)/(n1 * n2)
    cut_LR /= np.sum(cut_LR)/(N1 * N2)

    return cut_HR, cut_LR
import numpy as np
import scipy.signal as scp


def sinc2D(x,y):
    return np.sinc(x)*np.sinc(y)

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

def make_mat2D_fft(shape, coord_lr, p):
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
    n1,n2 = shape
    a, b = np.where(np.zeros((n1,n2))==0)
    A, B = coord_lr
    mat = np.zeros((n1*n2, B.size))

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

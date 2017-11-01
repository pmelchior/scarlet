import numpy as np
import galsim
from . import nmf

def getDiffKernel(P, P0, nx=50, ny=50, pixel_scale=1.):
    PD = galsim.Convolve(P, galsim.Deconvolve(P0))
    PDimg = PD.drawImage(nx=nx,ny=ny, scale=pixel_scale, method='no_pixel')
    return PDimg

def getDiffKernels(psf_bands, P0, radius_cut=None, pixel_scale=1.):
    kernels = []
    diff_kernels = []
    reconv_kernels = []
    psf_error = 0

    for i in range(len(psf_bands)):
        P = psf_bands[i]
        ny,nx = P.image.array.shape
        x,y = np.meshgrid(np.arange(nx),np.arange(ny))
        # radius wrt center of central pixel
        r = np.sqrt((x-0.5*nx + 0.5)**2 + (y-0.5*ny + 0.5)**2)

        PDimg = getDiffKernel(P, P0, nx=nx, ny=ny, pixel_scale=pixel_scale)
        kernels.append(P.image.array)
        diff_kernels.append(PDimg.array)

        # just for testing
        P0img = P0.drawImage(nx=nx,ny=ny, scale=pixel_scale, method='no_pixel')
        P0FromImage = galsim.InterpolatedImage(P0img)

        # truncate difference kernel below 3e-3
        # to make the more compact
        #mask = np.abs(PDimg.array) < thresh
        if radius_cut is not None:
            mask = r > radius_cut
            PDimg.array[mask] = 0
        """
        # construct reconvolved kernel with Galsim
        PDFromImage = galsim.InterpolatedImage(PDimg)
        P0PD = galsim.Convolve(P0FromImage, PDFromImage)
        P0PDimg = P0PD.drawImage(nx=nx,ny=ny, scale=pixel_scale, method='no_pixel')
        reconv_kernels.append(P0PDimg.array)
        error = ((reconv_kernels[i]-kernels[i])**2).sum()
        psf_error += error
        """

    diff_kernels = np.array(diff_kernels)
    B = len(psf_bands)
    N,M = ny,nx
    P = nmf.adapt_PSF(diff_kernels, B, (N,M))
    reconv_kernels = np.empty_like(diff_kernels)
    for b in range(B):
        reconv_kernels[b] = P[b].dot(P0img.array.flatten()).reshape(N,M)
        error = ((reconv_kernels[b]-kernels[b])**2).sum()
        print (b,error)
        psf_error += error

    return np.array(kernels), np.array(diff_kernels), np.array(reconv_kernels), psf_error

def matchPSFs(psfs, fwhm=2., pixel_scale=1., radius_cut=10.):
    B = len(psfs)

    # check that all images have the same shape
    nxy = np.empty((B,2), dtype='int64')
    for b in range(B):
            nxy[b] = psfs[b].shape
    max_xy = np.max(nxy, axis=0)

    # if not: extend them
    psf_bands = []
    for b in range(B):
        if all(nxy[b] == max_xy):
            _img = psfs[b]
        else:
            N,M = nxy[b]
            dxy = (max_xy - nxy[b])/2
            _img = np.zeros(max_xy)
            _img[dxy[0]:N+dxy[0],dxy[1]:M+dxy[1]] = psfs[b]
        psf_img = galsim.ImageF(_img, scale=pixel_scale)
        psf_bands.append(galsim.InterpolatedImage(psf_img))

    P0 = galsim.Gaussian(fwhm=fwhm)
    return getDiffKernels(psf_bands, P0, radius_cut=radius_cut, pixel_scale=pixel_scale)

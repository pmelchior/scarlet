# Scarlet

This package performs source separation (aka "deblending") on astronomical multi-band images of stars and galaxies by modeling each source with a SED and a non-parametric morphology (or multiple such components per source). The minimization is done using the block-SDMM algorithm described in [Moolekamp and Melchior 2017](https://arxiv.org/abs/1708.09066), and the development of this package is part of the [LSST Science Pipeline](https://pipelines.lsst.io).

Because of its generic utility, this package provides a stand-alone implementation that contains the core components of the deblending algorithm, while the [meas_deblender](https://github.com/lsst/meas_deblender) package contains a wrapper to implement the algorithms in the LSST stack. 

The API is not fully stable, so feel free to contact the authors [fred3m](https://github.com/fred3m) and [pmelchior](https://github.com/pmelchior) for guidance. For bug reports and feature request, open an issue.

If you make use of scarlet, please acknowledge [Moolekamp and Melchior 2017](https://arxiv.org/abs/1708.09066) and Melchior et al. (in prep.), which describe in detail the algorithms and constraints used in this package.

## Example use

```python
import numpy as np
import scarlet

# load image data as image cube with B bands, Ny x Nx pixels
img = np.empty((B,Ny,Nx))

# detect objects in img with sep (https://github.com/kbarbary/sep)
def makeCatalog(img):
    detect = img.mean(axis=0) # simple average for detection
    bkg = sep.Background(detect)
    catalog = sep.extract(detect, 1.5, err=bkg.globalrms)
    bg_rms = np.array([sep.Background(band).globalrms for band in img])
    return catalog, bg_rms
catalog, bg_rms = makeCatalog(img)

# constraints on morphology:
# "S": symmetry
# "m": monotonicity (with neighbor pixel weighting)
# "+": non-negativity
constraints = {"S": None, "m": {'use_nearest': False}, "+": None}

# initial size of the box around each object
# will be adjusted as needed
shape = (B, 15, 15)
sources = [scarlet.Source((obj['y'],obj['x']), shape, constraints=constraints) for obj in catalog]
blend = scarlet.Blend(sources, img, bg_rms=bg_rms)

# if you have per-pixel weights:
weights = np.empty((B,Ny,Nx))
blend = scarlet.Blend(sources, img, bg_rms=bg_rms, weights=weights)

# if you have per-band PSF kernel images:
# Note: These need to be difference kernels to a common minimum 
pdiff = [PSF[b] for b in range(B)]
psf = scarlet.transformations.GammaOp(shape, pdiff)
blend = scarlet.Blend(sources, img, bg_rms=bg_rms, psf=psf)

# run the fitter for 200 steps (or until convergence)
blend.fit(200)

# render the multi-band model: has same shape as img
model = blend.get_model()

# inspect the components
for source in sources:
    model = source.get_model()
```
# Scarlet

This package performs source separation (aka "deblending") on multi-band images. It's geared towards optical astronomy, where scenes are composed of stars and galaxies, but it is straightforward to apply it to other imaging data.

Separation is achieved through a constrained matrix factorization, which models each source with a Spectral Energy Distribution (SED) and a non-parametric morphology, or multiple such components per source. In astronomy jargon, the code performs forced photometry (with PSF matching if needed) using an optimal weight function given by the signal-to-noise weighted morphology across bands. The approach works well if the sources in the scene have different colors and can be further strengthened by imposing various additional constraints/priors on each source. The minimization itself is done using the proximal block-SDMM algorithm described in [Moolekamp & Melchior 2017](https://arxiv.org/abs/1708.09066).

Because of its generic utility, this package provides a stand-alone implementation that contains the core components of the source separation algorithm. However, the development of this package is part of the [LSST Science Pipeline](https://pipelines.lsst.io);  the [meas_deblender](https://github.com/lsst/meas_deblender) package contains a wrapper to implement the algorithms here for the LSST stack.

The API is reasonably stable, but feel free to contact the authors [fred3m](https://github.com/fred3m) and [pmelchior](https://github.com/pmelchior) for guidance. For bug reports and feature request, open an issue.

If you make use of scarlet, please acknowledge [Moolekamp and Melchior 2017](https://arxiv.org/abs/1708.09066) and Melchior et al. (in prep.), which describe in detail the algorithms and constraints used in this package.

## Prerequisites

The code runs on python>=2.7. In addition, you'll need

* numpy
* scipy
* pybind11
* [proxmin](https://github.com/pmelchior/proxmin)

## Example use

```python
import numpy as np
import scarlet

# load image data as image cube with B bands, Ny x Nx pixels
img = np.empty((B,Ny,Nx))

# detect objects in img
# we use the python SExtractor clone sep (https://github.com/kbarbary/sep)
# but use whatever you prefer
def makeCatalog(img):
    detect = img.mean(axis=0) # simple average for detection
    bkg = sep.Background(detect)
    catalog = sep.extract(detect, 1.5, err=bkg.globalrms)
    bg_rms = np.array([sep.Background(band).globalrms for band in img])
    return catalog, bg_rms
catalog, bg_rms = makeCatalog(img)

# define sources in the model (this is where the magic happens...)
# for convenience, we predefine PointSource and ExtendedSource classes.
# 1) by default, ExtendedSources are initialized and constrained
# as strictly positive, symmetric, and monotonic.
# WARNING: coordinates are expected to use numpy/C ordering: (y,x)
sources = [scarlet.ExtendedSource((obj['y'],obj['x']), img, bg_rms) for obj in catalog]

# 2) if you want to change the constraints (but not the initialization)
# e.g. add a l0 sparsity penalty at approx the noise level
import scarlet.constraints as sc
constraints = sc.SimpleConstraint() & sc.DirectMonotonicityConstraint() & sc.SymmetryConstraint() & sc.L0Constraint(bg_rms.sum())
sources = [scarlet.ExtendedSource((obj['y'],obj['x']), img, bg_rms, constraints=constraints) for obj in catalog]

# 3) if you have per-band PSF kernel images:
# Note: These need to be difference kernels to a common minimum size
pdiff = [PSF[b] for b in range(B)]
psf = scarlet.transformations.GammaOp(pdiff)
sources = [scarlet.ExtendedSource((obj['y'],obj['x']), img, bg_rms, psf=psf) for obj in catalog]

# 4) if you want more control over source initialization and constraints,
# make your own class:
class MySource(scarlet.Source):
    def __init__(self, *args, **kwargs):
        # determine the initial sed and morphology, center, constraints
        # then construct the source, e.g:
        super(MySource, self).__init__(sed, morph, center=center, constraints=constraints, psf=psf, fix_sed=False, fix_morph=False, fix_frame=False, shift_center=0.2)

# define blended scene from all sources
blend = scarlet.Blend(sources, img, bg_rms=bg_rms)

# if you have per-pixel weights:
weights = np.empty((B,Ny,Nx))
blend = scarlet.Blend(sources, img, bg_rms=bg_rms, weights=weights)

# run the fitter for 200 steps (or until convergence)
blend.fit(200)

# render the multi-band model: has same shape as img
model = blend.get_model()

# render single component of the blend model in scene
k = 0
model_k = blend.get_model(k)

# inspect the source in their own frame (centered in bounding box)
for source in sources:
    model = source.get_model()
```

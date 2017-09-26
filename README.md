# NMF Deblender

This package is deblends astronomical multi-band images of stars and galaxies by modeling each source with a SED and a non-parametric morphology (or multiple such components per source). The minimization is done using the block-SDMM algorithm described in [Moolekamp and Melchior 2017](https://arxiv.org/abs/1708.09066), and the development of this package is part of the [LSST Science Pipeline](https://pipelines.lsst.io).

Because of its generic utility, this package provides a stand-alone implementation that contains the core components of the deblender algorithm, while the [meas_deblender](https://github.com/lsst/meas_deblender) package contains a wrapper to implement the algorithms in the LSST stack. At this time the package is not well documented and under heavy development, so it is recommended to contact the authors [fred3m](https://github.com/fred3m) and [pmelchior](https://github.com/pmelchior) for advice on using the package, its current limitations, and its planned upgrades.

# Acknowledging this package
If you are making use of the deblender package, please acknowledge [Moolekamp and Melchior 2017](https://arxiv.org/abs/1708.09066) and Melchior et al. (in prep.), which describe in detail the concept as well as algorithms and constraints used in this deblender.

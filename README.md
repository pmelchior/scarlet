# NMF Deblender

This package is designed to deblend astronomical images of stars and galaxies by modeling each source as an SED (or multiple components, each with an individual SED) multiplied by the morphology of each component. The minimization is done using the block SDMM algorithm described in [Moolekamp and Melchior 2017](https://arxiv.org/abs/1708.09066), and the development of this package is part of the [LSST Science Pipeline](https://pipelines.lsst.io).

Because of it's more general use than other parts of the LSST stack and the co-development for WFIRST, this stand alone package contains the core components of the deblender algorithm while the [meas_deblender](https://github.com/lsst/meas_deblender) package contains a wrapper to implement the algorithms in the LSST stack. At this time the package is not well documented and under current development, so it is recommended to contact the authors [fred3m](https://github.com/fred3m) and [pmelchior](https://github.com/pmelchior) for advice on using the package, it's current limitations, and it's planned upgrades.

# Acknowledging this package
If you are making use of the deblender package, please acknowledge [Moolekamp and Melchior 2017](https://arxiv.org/abs/1708.09066) and Melchior and Moolekamp 2017 (in Press), which will describe in detail the algorithms and constraints used in this deblender.

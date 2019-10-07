[![](https://travis-ci.org/fred3m/scarlet.svg?branch=master)](https://travis-ci.org/fred3m/scarlet)
[![](https://img.shields.io/github/license/fred3m/scarlet.svg)](https://github.com/fred3m/scarlet/blob/master/LICENSE.md)
[![DOI](https://img.shields.io/badge/DOI-10.1016%2Fj.ascom.2018.07.001-blue.svg)](https://doi.org/10.1016/j.ascom.2018.07.001)
[![arXiv](https://img.shields.io/badge/arxiv-1802.10157-red.svg)](https://arxiv.org/abs/1802.10157)

# Scarlet

This package performs source separation (aka "deblending") on multi-band images. It's geared towards optical astronomy, where scenes are composed of stars and galaxies, but it is straightforward to apply it to other imaging data.

**For the full documentation see [the docs](https://pmelchior.github.io/scarlet/).**

Separation is achieved through a constrained matrix factorization, which models each source with a Spectral Energy Distribution (SED) and a non-parametric morphology, or multiple such components per source. In astronomy jargon, the code performs forced photometry (with PSF matching if needed) using an optimal weight function given by the signal-to-noise weighted morphology across bands. The approach works well if the sources in the scene have different colors and can be further strengthened by imposing various additional constraints/priors on each source.

The minimization itself uses the proximal gradient method (PGM). In short, we iteratively compute gradients of the likelihood (or of the posterior if priors are included), perform a downhill step, and project the outcome on a sub-manifold that satisfies one or multiple non-differentiable constraints for any of the sources.

This package provides a stand-alone implementation that contains the core components of the source separation algorithm. However, the development of this package is part of the [LSST Science Pipeline](https://pipelines.lsst.io);  the [meas_deblender](https://github.com/lsst/meas_deblender) package contains a wrapper to implement the algorithms here for the LSST stack.

The API is reasonably stable, but feel free to contact the authors [fred3m](https://github.com/fred3m) and [pmelchior](https://github.com/pmelchior) for guidance. For bug reports and feature request, open an issue.

If you make use of scarlet, please acknowledge [Melchior et al. (2018)](https://doi.org/10.1016/j.ascom.2018.07.001), which describes in detail the concepts and algorithms used in this package:
```
@ARTICLE{scarlet,
   author = {{Melchior}, Peter and {Moolekamp}, Fred and {Jerdee}, Maximilian and {Armstrong}, Robert and
	{Sun}, Ai-Lei and {Bosch}, James and {Lupton}, Robert},
    title = "{SCARLET: Source separation in multi-band images by Constrained Matrix Factorization}",
  journal = {Astronomy and Computing},
   volume = "24",
    pages = {129 - 142},
     year = "2018",
     month = jul,
     issn = {2213-1337},
      doi = {10.1016/j.ascom.2018.07.001},
      url = {http://www.sciencedirect.com/science/article/pii/S2213133718300301},
 keywords = {Methods, Data analysis, Techniques, Image processing, Galaxies, Non-negative matrix factorization}
archivePrefix = "arXiv",
   eprint = {1802.10157},
 primaryClass = "astro-ph.IM"
}
```

## Prerequisites

The code runs on python>=3.5. In addition, you'll need

* numpy
* pybind11
* autograd
* [proxmin](https://github.com/pmelchior/proxmin)

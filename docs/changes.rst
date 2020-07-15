1.1 (2020-07-14)
----------------

General
^^^^^^^
- Introduction of `Spectrum` and `Morphology` classes (both inherited from `Factor`) to modularize `FactorizedComponent`. They implement `get_model` member functions which create the representation of the spectrum and morphology model from whatever it uses as parameters.
- `Component.sed` has been renamed to `Component.spectrum`.
- `Component.morph` has been renamed to `Component.morphology`.
- `FunctionComponent` has been deprecated.
- Initialization and plotting functions are now consistent for multi-observation data sets.

New Features
^^^^^^^^^^^^
- One can now e.g. combine a custom spectrum model with an existing `PointSourceMorphology`.
- `Box` can now be sliced for lower dimensional subset; two Boxes can be combined with `@` to create a higher-dimensional box.
- `display` functions can now show the source boxes.

1.0 (2019-12-22)
--------------------

General
^^^^^^^
- Complete overhaul of the modeling code. It now allows for arbitrary `Parameter`
  instances to generate a source model (e.g. point-source or Sersic fitting).
- Each `Parameter` can be further constrained by proximal constraints or priors.
- New optimizer from the `proxmin` package: `adaprox` is an adaptive proximal gradient
  method that doesn't require Lipschitz constants and uses different steps sizes for
  each element of an optimization parameter.

New Features
^^^^^^^^^^^^
- `Prior` can now be attached to a `Parameter` and its gradient will be added to likelihood gradients.
- `PointSource` performs optimization of centroid position and flux assuming the model PSF.
- `Fourier` helps with the bookkeeping involved with calculating FFTs to different shapes.
- Most `scarlet` objects can be pickled. That allows to store sources and reload sources.
- `Component` has `Box` to confine its footprint and save memory.
- `scarlet.display` has methods to `show_scene` and `show_sources` which allow fast inspection.
- `scarlet.measure` has methods to perform measurements on the component models.

API Changes
^^^^^^^^^^^
- `Frame` and `Observation` strongly prefer all arguments (`weights`, `psfs`, `channels`, `wcs`) to be set.
- `Frame.psfs` (or `Observation.Frame.psfs`) now stores a `PSF` object, and the `PSF.image` method needs to be called
  to access the PSF image cube.
- `ExtendedSource` does not accept `bg_rms` keyword. This information is derived from
  `observation.weights`.


0.5 (2019-06-26)
----------------

General
^^^^^^^

- Completely restructured code, including using `autograd` package to calculate gradients.
- PSF convolutions are now performed on the model of the entire blend as opposed to
  individually for each source.
- `Blend` no longer fits for source positions. Instead it is up to the user to implement a
  centering algorithm, such as centroiding or the `scarlet.update.fix_pixel_center`.
- Updates to all of the docs and tutorials to match the new API.

New Features
^^^^^^^^^^^^
- Multi-resolution deblending.
- A `BoundingBox` container to define rectangular regions in the pixel grid, a `trim` function
  to strip away all pixels lower than a cutoff value, and a `flux_at_edge` method to determine if
  a model has flux on the edge of the image.
- A `Scene` class to handle the metadata for the deblended model.
- An `Observation` class to provide mappings from the model scene to a set
  of observations with potentially multiple filter bands.
- A `LowResObservation` class for observations with lower resolution than the blended model,
  requiring resampling and reprojection.
- A `BlendFlag` has been introduced to keep track of things that might have gone wrong
  while debending.
- A gradient `Prior` class has been created to allow users to update the gradient of a parameter
  in a source or `Component`.
- Color normalization imports from `astropy.visualization`, with the addition of `img_to_channels` to
  map any number of filter bands to an RGB that can be displayed.
- A `CombinedExtendedSource` initializes sources with multiple observations at different pixel
  resolutions.
- `Frame` issues warnings when PSF is not specified or not normalized.
- `Frame.channels` is used to identify channels in multiple observations.
- PSF convolutions and ffts of image cubes are performed using ndimensional fft along selected axes for better performance.

API Changes
^^^^^^^^^^^
- `Scene` is a is a confusing name and has been renamed to `Frame`.
- `Observation.get_model` is a confusing name and has been renamed to `Observation.render`.
- `Blend` does not have a `frame` argument any more, it inherits its frame from sources.
- `Component` is now the base class for sources and `Source` has been removed
  `PointSource`, `ExtendedSource`, and `MultiComponentSource` are now inherited from `Component`.
- The `Constraint` class and module were removed in place of an update method that
  has been added to `Component`s (and thus sources). User defined constraints should now inherit
  from `Component` or one of its subclasses and overwrite the `update` method. Constraints are
  now applied using functions from the `scarlet.update` module or similar user defined update
  functions.
- `get_model` has been simplified for `ComponentTree` and `Components` to always return the
  entire scene with the component added in place, using the `Scene` target `PSF`. To get a
  model in the same space as observations requires calling `Observation.get_model` and passing
  the high resolution/best seeing model.
- `Frame` has a `channels` argument instead of `filter_curves`.
- The old `resampling.py` module has been renamed `interpolation.py` and a new `resampling.py`
  used for multi-resolution resampling/reprojection.
- Sources and components are no longer centered in a small patch that is reprojected
  into the model frame. Instead components can exist anywhere on an image and constraints that
  require a center, such as symmetry and monotonicity, can use the new `uncentered_operator` method.
- `make_operator` is now a method of the `LowResObservation` class as should be.


0.45 (2019-3-27)
----------------

General
^^^^^^^

- Tests have been added for the `operator`, `constraint`, `resample`, and `transformation`
  modules. Tests are run on Travis CI with each new build.

New Features
^^^^^^^^^^^^

- Convolutions can now be done in Fourier space and/or real
  space by setting the `use_fft` option in `config.Config`.

- A new internal function was added to project images into larger or
  smaller images by slicing and/or padding.

- Interpolation kernels have been implemented for fractional pixel shifts using
  bilinear, cubic spline, and Lanczos algorithms.

0.4 (2019-2-15)
---------------

General
^^^^^^^

- Dropped python 2 support

New Features
^^^^^^^^^^^^

- Initialization of `PointSource`, `ExtendedSource`, and
  `MultiComponentSource` now take `normalization` as an
  input parameter, which selects the normalization used
  to break the color/morphology degeneracy. The default is to
  use `constraint.Normalization.Smax`, which normalizes
  `S` (morphology) so that the peak pixel always has a value
  of one.

- The `exact_lipschitz` option has been added to `Config`.
  This allows the user to recalculate the `Lipschitz` constant
  (used for calculating step sizing) in each iteration as opposed
  to an approximation (the default) used for speed.

API Changes
^^^^^^^^^^^

- The default value of the `Config` parameter `accelerated`
  is now `False`. This was done because in some cases
  acceleration caused the optimization to diverge, and because
  the new S matrix normalization causes the code to run faster
  than the old accelerated version for most blends.

- Due to the additional normalization parameters, a `get_flux`
  method has been added to properly get the flux of an object
  in each band.


Bug Fixes
^^^^^^^^^

- Monotonicity would break if the bounding box for a `Source` was not odd.
  The shape is now forced to be odd when the `Source` is initialized.

- `fix_sed` and `fix_morph` were not correctly passing the SED and morphology
  though correctly, but this behavior has been corrected.

- Installations that do not have access to get the current commit using git
  will now truncate the release number to the subversion. This was needed for
  binary installs (like the LSST-DM stack).

Other Changes and Additions
^^^^^^^^^^^^^^^^^^^^^^^^^^^

- The names of the sub modules `operators`, `constraints`, `transformations`,
  have been changed to the singular `operator`, `constraint`, `transformation`.

- Internally the logic that separates `Component`, `Source`, and `Blend` objects
  has been clarified. A `Source` is just a collection of components with relatively
  no internal logic other than initialization. `ComponentTree` is a hierarchical
  list of components that replaces the old `ComponentList` class, making it easier
  to have more complicated objects and improving the internal interface to them.

- The internal resizing and re-centering algorithms have been updated.

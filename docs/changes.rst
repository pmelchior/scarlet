
0.5 (unreleased)
----------------

General
^^^^^^^

- Completely restructured code, including using `autograd` package to calculate gradients.
- PSF convolutions are now performed on the model of the entire blend as opposed to
  individually for each source.
- It is possible to perform deblending on images with different orientations and resolutions.
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

API Changes
^^^^^^^^^^^
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
- `Blend` no longer fits for source positions. Instead it is up to the user to implement a
  centering algorithm, such as centroiding or the `scarlet.update.fix_pixel_center`.
- The old `resampling.py` module has been renamed `interpolation.py` and a new `resampling.py`
  used for multi-resolution resampling/reprojection.
- Sources and components are no longer centered in a small patch that is reprojected
  into the model frame. Instead components can exist anywhere on an image and constraints that
  require a center, such as symmetry and monotonicity, can use the new `uncentered_operator` method.

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
----------------

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

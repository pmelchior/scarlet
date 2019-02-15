0.5 (unreleased)
----------------
- no changes yet

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

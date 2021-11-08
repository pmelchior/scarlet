# Support for CONDA_PREFIX (conda-forge compilers)
if [[ -n "$CONDA_PREFIX" ]]; then
    export EIGEN_INCLUDE=$CONDA_PREFIX/include/eigen3
fi

# Inside conda-build (stackvana)
if [[ "$CONDA_BUILD" == "1" ]]; then
    export EIGEN_INCLUDE=$PREFIX/include/eigen3
fi

install() {
  PYDEST="$PREFIX/lib/python"
  PYTHONPATH="$PYDEST:$PYTHONPATH" \
     eval python setup.py install --single-version-externally-managed --record record.txt  $PYSETUP_INSTALL_OPTIONS
  install_ups
}

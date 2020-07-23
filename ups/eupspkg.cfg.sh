# If MACOSX_DEPLOYMENT_TARGET is not set, we force it to correspond to the
# greater of 10.9 or the target used when compiling Python.
# This works around DM-5409, wherein mpi4py was attempting to use an OS X 10.5
# SDK, based on querying Anaconda, and failing; and DM-6133, wherein distutils
# refuses to let us target an earlier SDK than Python was compiled with.

# Support for CONDA_PREFIX (conda-forge compilers)
if [[ -n "$CONDA_PREFIX" ]]; then
    export EIGEN_INCLUDE=$CONDA_PREFIX/include/eigen3
fi

# Inside conda-build (stackvana)
if [[ "$CONDA_BUILD" == "1" ]]; then
    export EIGEN_INCLUDE=$PREFIX/include/eigen3
fi

if [ -z "$MACOSX_DEPLOYMENT_TARGET" ]; then
    MIN_DEPLOYMENT_TARGET=9
    CFG_DEPLOYMENT_TARGET=$(python -c "import sysconfig; print((sysconfig.get_config_var('MACOSX_DEPLOYMENT_TARGET') or '10.$MIN_DEPLOYMENT_TARGET').split('.')[1])")
    export MACOSX_DEPLOYMENT_TARGET=10.$((MIN_DEPLOYMENT_TARGET>CFG_DEPLOYMENT_TARGET?MIN_DEPLOYMENT_TARGET:CFG_DEPLOYMENT_TARGET))
fi

install() {
  PYDEST="$PREFIX/lib/python"
  PYTHONPATH="$PYDEST:$PYTHONPATH" \
     eval python setup.py install --single-version-externally-managed --record record.txt  $PYSETUP_INSTALL_OPTIONS
  install_ups
}

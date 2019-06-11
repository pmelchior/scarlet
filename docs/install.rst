:tocdepth: 1

Installation
============

*SCARLET* has several dependencies that must be installed prior to installation:

#. numpy_
#. proxmin_ (proximal algorithms used to minimize the likelihood)
#. pybind11_ (integrate C++ code into python)
#. peigen_ (used to load the Eigen_ headers if they are not already installed)
#. autograd_ (needed to calculate gradients during optimization)

Optional Dependencies (required to build docs)

#. matplotlib_ (required to use the plotting functionality in `scarlet.display`)
#. astropy_ (required for visualization, some of the tutorials, and sample data)
#. scipy_ (required for nearest neighbor monotonicity and estimating a target PSF kernel)
#. sphinx_ (required to build the docs)
#. sphinx_rtd_theme_ (required to use the Read the Docs theme)
#. nbsphinx_ (required to compile notebooks)
#. numpydoc_ (allow for numpy style docstrings)

From Source
-----------
First download the github repo:
::

    git clone https://github.com/fred3m/scarlet.git

then install using:
::

    python setup.py install

*SCARLET* requires the Eigen_ library headers, which are downloaded automatically when using the
command above.
If you already have a local version of Eigen_ and don't want to download the headers, use

::

    python setup.py build_ext -I<full path to your Eigen header files>
    python setup.py install

.. warning::
    `build_ext` does not accept relative paths, so `<full path to your Eigen header files>`
    must be a full path.

.. _numpy: http://www.numpy.org
.. _proxmin: https://github.com/pmelchior/proxmin/tree/master/proxmin
.. _pybind11: https://pybind11.readthedocs.io/en/stable/
.. _peigen: https://github.com/fred3m/peigen
.. _Eigen: http://eigen.tuxfamily.org/index.php?title=Main_Page
.. _autograd: https://github.com/HIPS/autograd
.. _matplotlib: https://matplotlib.org
.. _astropy: http://www.astropy.org
.. _sphinx: http://www.sphinx-doc.org/en/master/
.. _sphinx_rtd_theme: https://sphinx-rtd-theme.readthedocs.io/en/latest/
.. _nbsphinx: https://nbsphinx.readthedocs.io/en/0.4.2/
.. _numpydoc: https://numpydoc.readthedocs.io/en/latest/
.. _scipy: https://www.scipy.org/

:tocdepth: 1

Installation
============

*SCARLET* has several dependencies that must be installed prior to installation:

#. numpy_
#. scipy_
#. proxmin_ (proximal algorithms used to minimize the likelihood)
#. pybind11_ (integrate C++ code into python)

Optional Dependencies (required to build docs)

#. matplotlib_ (required to use the plotting functionality in `scarlet.display`)
#. astropy_ (required for some of the tutorials and sample data)
#. galsim_ (required for PSF de-convolution and matching)

From Source
-----------

::

    git clone https://github.com/fred3m/scarlet.git
    python setup.py

From pip
--------
(not yet active, but coming soon)

::

    pip install scarlet

.. _numpy: http://www.numpy.org
.. _scipy: https://www.scipy.org
.. _proxmin: https://github.com/pmelchior/proxmin/tree/master/proxmin
.. _pybind11: https://pybind11.readthedocs.io/en/stable/
.. _matplotlib: https://matplotlib.org
.. _astropy: http://www.astropy.org
.. _galsim: https://github.com/GalSim-developers/GalSim

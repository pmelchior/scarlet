# This uses the code at
# https://github.com/pybind/python_example/blob/master/setup.py
# as a template to integrate pybind11

import os
from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext
import sys
import setuptools
import subprocess

pybind11_path = None
if "PYBIND11_DIR" in os.environ:
    pybind11_path = os.environ["PYBIND11_DIR"]
eigen_path = None
if "EIGEN_DIR" in os.environ:
    eigen_path = os.environ["EIGEN_DIR"]

packages = find_packages()

class get_pybind_include(object):
    """Helper class to determine the pybind11 include path
    The purpose of this class is to postpone importing pybind11
    until it is actually installed, so that the ``get_include()``
    method can be invoked. """

    def __init__(self, user=False):
        self.user = user

    def __str__(self):
        if pybind11_path is not None:
            return os.path.join(os.environ["PYBIND11_DIR"], "include")
        else:
            import pybind11

            return pybind11.get_include(self.user)


class get_eigen_include(object):
    """Helper class to determine the peigen include path
    The purpose of this class is to postpone importing peigen
    until it is actually installed, so that the ``get_include()``
    method can be invoked.
    """

    def __init__(self, user=False):
        self.user = user

    def __str__(self):
        if eigen_path is not None:
            return os.path.join(os.environ["EIGEN_DIR"], "include")
        else:
            import peigen

            return peigen.header_path


ext_modules = [
    Extension(
        "scarlet.operators_pybind11",
        ["scarlet/operators_pybind11.cc"],
        include_dirs=[
            get_pybind_include(),
            get_pybind_include(user=True),
            get_eigen_include(),
        ],
        language="c++",
    )
]


# As of Python 3.6, CCompiler has a `has_flag` method.
# cf http://bugs.python.org/issue26689
def has_flag(compiler, flagname):
    """Return a boolean indicating whether a flag name is supported on
    the specified compiler.
    """
    import tempfile

    with tempfile.NamedTemporaryFile("w", suffix=".cpp") as f:
        f.write("int main (int argc, char **argv) { return 0; }")
        try:
            compiler.compile([f.name], extra_postargs=[flagname])
        except setuptools.distutils.errors.CompileError:
            return False
    return True


def cpp_flag(compiler):
    """Return the -std=c++[11/14] compiler flag.
    The c++14 is prefered over c++11 (when it is available).
    """
    if has_flag(compiler, "-std=c++14"):
        return "-std=c++14"
    elif has_flag(compiler, "-std=c++11"):
        return "-std=c++11"
    else:
        raise RuntimeError(
            "Unsupported compiler -- at least C++11 support " "is needed!"
        )


class BuildExt(build_ext):
    """A custom build extension for adding compiler-specific options."""

    c_opts = {
        "msvc": ["/EHsc"],
        "unix": [],
    }

    if sys.platform == "darwin":
        c_opts["unix"] += ["-stdlib=libc++", "-mmacosx-version-min=10.7"]

    def build_extensions(self):
        ct = self.compiler.compiler_type
        opts = self.c_opts.get(ct, [])
        if ct == "unix":
            opts.append('-DVERSION_INFO="%s"' % self.distribution.get_version())
            opts.append(cpp_flag(self.compiler))
            if has_flag(self.compiler, "-fvisibility=hidden"):
                opts.append("-fvisibility=hidden")
        elif ct == "msvc":
            opts.append('/DVERSION_INFO=\\"%s\\"' % self.distribution.get_version())
        for ext in self.extensions:
            ext.extra_compile_args = opts
        build_ext.build_extensions(self)


install_requires = ["numpy", "proxmin>=0.6.9", "autograd>=1.3"]
# Only require the pybind11 and peigen packages if
# the C++ headers are not already installed
if pybind11_path is None:
    install_requires.append("pybind11>=2.2")
if eigen_path is None:
    install_requires.append("peigen>=0.0.9")


setup(
    name="scarlet",
    packages=packages,
    description="Blind Source Separation using proximal matrix factorization",
    author="Fred Moolekamp and Peter Melchior",
    author_email="peter.m.melchior@gmail.com",
    url="https://github.com/pmelchior/scarlet",
    keywords=["astro", "deblending", "photometry", "nmf"],
    ext_modules=ext_modules,
    install_requires=install_requires,
    cmdclass={"build_ext": BuildExt},
    zip_safe=False,
    setup_requires=['setuptools_scm', 'setuptools_scm_git_archive'],
    use_scm_version={'write_to': 'scarlet/_version.py'},
)

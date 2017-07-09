# This uses the code at
# https://github.com/pybind/python_example/blob/master/setup.py
# as a template to integrate pybind11

import os
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import sys
import setuptools
import subprocess

# Use the firt 7 digits of the git hash to set the version
__version__ = '0.0.'+subprocess.check_output(['git', 'rev-parse', 'HEAD'])[:7].decode("utf-8")

packages = []
for root, dirs, files in os.walk('.'):
    if not root.startswith('./build') and '__init__.py' in files:
        packages.append(root[2:])
print("Packages:", packages)

print('Packages:', packages)

class get_pybind_include(object):
    """Helper class to determine the pybind11 include path
    The purpose of this class is to postpone importing pybind11
    until it is actually installed, so that the ``get_include()``
    method can be invoked. """

    def __init__(self, user=False):
        self.user = user

    def __str__(self):
        import pybind11
        return pybind11.get_include(self.user)

ext_modules = [
    Extension(
        'deblender.proximal_utils',
        ['deblender/proximal_utils.cc'],
        include_dirs=[
            get_pybind_include(),
            get_pybind_include(user=True)
        ],
        language='c++'
    )
]

# As of Python 3.6, CCompiler has a `has_flag` method.
# cf http://bugs.python.org/issue26689
def has_flag(compiler, flagname):
    """Return a boolean indicating whether a flag name is supported on
    the specified compiler.
    """
    import tempfile
    with tempfile.NamedTemporaryFile('w', suffix='.cpp') as f:
        f.write('int main (int argc, char **argv) { return 0; }')
        try:
            compiler.compile([f.name], extra_postargs=[flagname])
        except setuptools.distutils.errors.CompileError:
            return False
    return True


def cpp_flag(compiler):
    """Return the -std=c++[11/14] compiler flag.
    The c++14 is prefered over c++11 (when it is available).
    """
    if has_flag(compiler, '-std=c++14'):
        return '-std=c++14'
    elif has_flag(compiler, '-std=c++11'):
        return '-std=c++11'
    else:
        raise RuntimeError('Unsupported compiler -- at least C++11 support '
                           'is needed!')


class BuildExt(build_ext):
    """A custom build extension for adding compiler-specific options."""
    c_opts = {
        'msvc': ['/EHsc'],
        'unix': [],
    }

    if sys.platform == 'darwin':
        c_opts['unix'] += ['-stdlib=libc++', '-mmacosx-version-min=10.7']

    def build_extensions(self):
        ct = self.compiler.compiler_type
        opts = self.c_opts.get(ct, [])
        if ct == 'unix':
            opts.append('-DVERSION_INFO="%s"' % self.distribution.get_version())
            opts.append(cpp_flag(self.compiler))
            if has_flag(self.compiler, '-fvisibility=hidden'):
                opts.append('-fvisibility=hidden')
        elif ct == 'msvc':
            opts.append('/DVERSION_INFO=\\"%s\\"' % self.distribution.get_version())
        for ext in self.extensions:
            ext.extra_compile_args = opts
        build_ext.build_extensions(self)

setup(
  name = 'deblender',
  packages = packages,
  version = __version__,
  description = 'NMF Deblender',
  author = 'Fred Moolekamp and Peter Melchior',
  author_email = 'fred.moolekamp@gmail.com',
  url = 'https://github.com/fred3m/deblender',
  keywords = ['astro', 'deblender', 'photometry', 'nmf'],
  ext_modules=ext_modules,
  install_requires=['proxmin>=0.1', 'pybind11>=1.7', 'numpy', 'scipy'],
  cmdclass={'build_ext': BuildExt},
  zip_safe=False
)

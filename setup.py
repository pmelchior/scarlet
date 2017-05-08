from setuptools import setup
import os

packages = []
for root, dirs, files in os.walk('.'):
    if not root.startswith('./build') and '__init__.py' in files:
        packages.append(root[2:])

print('Packages:', packages)

setup(
  name = 'deblender',
  packages = packages,
  version = '0.0',
  description = 'NMF Deblender',
  author = 'Fred Moolekamp and Peter Melchior',
  author_email = 'fred.moolekamp@gmail.com',
  url = 'https://github.com/fred3m/deblender',
  #download_url = 'https://github.com/fred3m/popper/tarball/0.1',
  keywords = ['astro', 'deblender', 'photometry', 'nmf'],
  #classifiers = [],
)
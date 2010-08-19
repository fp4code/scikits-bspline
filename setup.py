#! /usr/bin/env python
# Last Change: 2010-08-19

# Copyright (C) 2010 Fabrice Pardo <fp4code@gmail.com>

descr   = """scikits.spline package.

Several ways to choose the sign of a complex square root... 
"""

import os
import sys

DISTNAME            = 'scikits.spline'
DESCRIPTION         = 'Scikits spline package'
LONG_DESCRIPTION    = descr
MAINTAINER          = 'Fabrice Pardo',
MAINTAINER_EMAIL    = 'fp4code@gmail.com',
URL                 = 'http://github.com/fp4code/scikits.spline'
LICENSE             = 'MIT'
DOWNLOAD_URL        = 'http://github.com/fp4code/scikits.spline/tarball/master'
VERSION             = '0.0.1'

import setuptools
from numpy.distutils.core import setup

def configuration(parent_package='', top_path=None, package_name=DISTNAME):
    if os.path.exists('MANIFEST'): os.remove('MANIFEST')

    from numpy.distutils.misc_util import Configuration
    config = Configuration(package_name, parent_package, top_path,
                           version = VERSION,
                           maintainer  = MAINTAINER,
                           maintainer_email = MAINTAINER_EMAIL,
                           description = DESCRIPTION,
                           license = LICENSE,
                           url = URL,
                           download_url = DOWNLOAD_URL,
                           long_description = LONG_DESCRIPTION)

    return config

if __name__ == "__main__":
    setup(configuration = configuration,
        install_requires = 'numpy',
        namespace_packages = ['scikits'],
        packages = setuptools.find_packages(),
        include_package_data = True,
        #test_suite="tester", # for python setup.py test
        zip_safe = False, # the package can run out of an .egg file
        classifiers =
            [ 'Development Status :: 1 - Planning',
              'Environment :: Console',
              'Intended Audience :: Developers',
              'Intended Audience :: Science/Research',
              'License :: OSI Approved :: MIT License',
              'Topic :: Scientific/Engineering'])

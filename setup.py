#!/usr/bin/env python

from distutils.core import setup

setup(name='jacovid',
    version='0.0.1',
    description='Application of jacques models to COVID',
    author='Serena Wang, Evan L. Ray',
    author_email='elray@umass.edu',
    url='https://github.com/reichlab/jacques-covid',
    packages=['jacovid'],
    include_package_data=True,
    package_data={'': ['data/*.csv']},
)

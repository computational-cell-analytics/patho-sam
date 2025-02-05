#!/usr/bin/env python

import runpy
from distutils.core import setup


__version__ = runpy.run_path("patho_sam/__version__.py")["__version__"]


setup(
    name='patho_sam',
    description='Segment Anything for Histopathology.',
    version=__version__,
    author='Titus Griebel, Anwai Archit, Constantin Pape',
    author_email='anwai.archit@uni-goettingen.de',
    url='https://github.com/computational-cell-analytics/patho-sam',
    packages=['patho_sam'],
)

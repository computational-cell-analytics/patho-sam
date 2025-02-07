#!/usr/bin/env python

import runpy
from distutils.core import setup


__version__ = runpy.run_path("patho_sam/__version__.py")["__version__"]


setup(
    name='patho_sam',
    description='Segment Anything for Histopathology.',
    version=__version__,
    author='Titus Griebel, Anwai Archit, Constantin Pape',
    author_email='titus.griebel@stud.uni-goettingen.de, anwai.archit@uni-goettingen.de, constantin.pape@informatik.uni-goettingen.de',  # noqa
    url='https://github.com/computational-cell-analytics/patho-sam',
    packages=['patho_sam'],
    license="MIT",
    entry_points={
        "console_scripts": [
            "patho_sam.example_data = patho_sam.util:get_example_wsi_data",
            "patho_sam.automatic_segmentation = patho_sam.automatic_segmentation:main",
        ]
    }
)

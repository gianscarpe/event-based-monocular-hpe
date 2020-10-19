#!/usr/bin/env python
import re

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()
    version = re.search(
        r'^__version__\s*=\s*"(.*)"',
        open("experimenting/__init__.py").read(),
        re.MULTILINE,
    ).group(1)


setuptools.setup(
    name='Master_thesis',
    version=version,    
    description='My master thesis',
    long_description=long_description,
    author='Gianluca Scarpellini',
    author_email='gianluca@scarpellini.dev',
    url='https://github.com/gianscarpe/event-camera',
    install_requires=[
        'albumentations',
        'h5py',
        'scikit-image',
        'scikit-learn',
        'scikit-video',
        'scipy',
        'torch>1',
        'kornia',
        'hydra-core==1.0.0rc1',
        'omegaconf',
        'opencv-python',
        'pytorch-lightning==0.8.5',
        'torchvision',
        'tqdm',
    ],

    packages=setuptools.find_packages(exclude=("tests",)),
    test_suite="tests",
    long_description_content_type="text/markdown",    
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GPL License",
        "Operating System :: Linux",
    ],
)

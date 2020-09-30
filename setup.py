#!/usr/bin/env python


import setuptools

setuptools.setup(
    name='Master_thesis',
    version='0.2',
    description='My master thesis',
    author='Gianluca Scarpellini',
    author_email='gianluca@scarpellini.dev',
    url='https://github.com/gianscarpe/event-camera',
    packages=setuptools.find_packages(exclude=("tests",)),
    test_suite="tests",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GPL License",
        "Operating System :: Linux",
    ],
)

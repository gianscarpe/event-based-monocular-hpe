#!/usr/bin/env python


import setuptools

setuptools.setup(
    name='Master_thesis',
    version='0.2',
    description='My master thesis',
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
        'pose3d_utils @ git+https://github.com/anibali/pose3d-utils.git#egg=pose3d_utils',
        'segmentation_models_pytorch @ git+https://github.com/qubvel/segmentation_models.pytorch#egg=segmentation_models_pytorch'
    ],
    packages=setuptools.find_packages(exclude=("tests",)),
    test_suite="tests",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GPL License",
        "Operating System :: Linux",
    ],
)

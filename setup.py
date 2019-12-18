#!/usr/bin/env python3
#
# File: setup.py
#
from setuptools import setup, find_packages

def get_long_description():
    with open("README.md", "r") as readme_file:
        return readme_file.read()


setup(
    name='structmechmod',
    description='PyTorch models learning structured mechanical systems',
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    install_requires=[
        'click',
        'py-dateutil',
        'ipdb',
        'torch >= 1.2',
        'numpy >= 1.13',
        'matplotlib',
        'scipy',
        'tensorboard',
        'tqdm',
        'termcolor',
    ],
    url='https://github.com/sisl/mechamodlearn/',
    packages=find_packages(),
    classifiers=(
        'Programming Language :: Python :: 3',
        'Development Status :: 4 - Beta',),
    version='0.0.1',
    author='rejuvyesh',
    author_email='mail@rejuvyesh.com',
    license='MIT',)

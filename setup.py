#!/usr/bin/env python
# coding: utf-8

from setuptools import setup, find_packages

setup(
    name='dl',
    packages = find_packages(),
    version='0.1.3',
    url='https://github.com/5121eun/dl.git',
    install_requires=[
        'torch'
    ]
)

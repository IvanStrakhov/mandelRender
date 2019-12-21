#!/usr/bin/env python

from setuptools import setup, find_packages

with open('README.md') as fh:
    long_description = fh.read()

setup(
    name='mandelRender',
    version='0.0.1',
    author='Ivan Strakhov',
    author_email='ivan-music@mail.ru',
    description='mandelbrot set renderer',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='http://phys.msu.ru',
    license='MIT',
    packages=find_packages(),
    # If all your code are in a module, use py_modules instead of packages:
    # py_modules=['ser'],
    scripts=['bin/mandelbrot'],
    
    test_suite='test',
    install_requires=['numpy>=1.13', 'matplotlib>=2.0,<3.0','Pillow>=2.0.0','numba>=0.45.0','console_progressbar>=1.1.0'],
    classifiers=[
        'Intended Audience :: Science/Research',
        'Intended Audience :: Education',
        'License :: OSI Approved :: MIT License',
        'Topic :: Education',
        'Programming Language :: Python :: 3',
        # See full list on https://pypi.org/classifiers/
    ],
    keywords='sample science astrophysics',
)

"""Setup script for yastn."""

from setuptools import setup, find_packages

description = ('? - variational with purification')

long_description = open('README.md', encoding='utf-8').read()

__version__ = '1.0.0'

requirements = open('requirements.txt').readlines()

setup(
    name='varpur',
    version=__version__,
    author='Tim Klemm, Gabriela Wójtowicz',
    author_email='gabriela.wojtowicz@uni-ulm.de',
    license='Apache License 2.0',
    platform=['any'],
    python_requires=('>=3.10'),
    description=description,
    long_description=long_description,
    long_description_content_type='text/markdown',
    install_requires=requirements,
    packages=find_packages(exclude='tests')
)

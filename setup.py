# -*- coding: utf-8 -*-

from setuptools import setup, find_packages


with open('README.rst') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='birdMigration',
    version='0.1.0',
    description='AI toolbox to understand and predict bird migration based on radar data',
    long_description=readme,
    author='Fiona Lippert',
    author_email='fiona@lipperta.de',
    url='https://github.com/FionaLippert/birdMigration',
    license=license,
    packages=find_packages(exclude=('tests', 'docs'))
)

# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

setup(
    name='cfdr-clustering',
    author='Andrey Kazarinov',
    author_email='andrei.kazarinov@gmail.com',
    packages=find_packages(exclude=('tests', 'tests.*')),
    install_requires=open('./requirements.txt').read(),
    scripts=['bin/clustering']
)

# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

setup(
    name='hccf',
    description="Hierarchical Clustering of Categorical Features",
    license="MIT",
    version="0.1",
    url="https://github.com/kazarinov/hccf",
    author='Andrey Kazarinov',
    author_email='andrei.kazarinov@gmail.com',
    keywords=["hierarchical clustering", "dimensionality reduction"],

    packages=find_packages(exclude=('tests', 'tests.*')),
    install_requires=open('./requirements.txt').read(),
    entry_points={
        "console_scripts": [
            "hccf = hccf.main:main"
        ]
    },

)

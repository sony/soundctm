#!/usr/bin/env python3
from setuptools import find_packages, setup

long_description = open("README.md", "r").read()

setup(
    name="hear21passt",
    description="Passt pretrained model for HEAR 2021 NeurIPS Competition",
    author="Khaled Koutini",
    author_email="first.last@jku.at",
    url="https://github.com/kkoutini/passt_hear21",
    license="Apache-2.0",
    long_description=long_description,
    long_description_content_type="text/markdown",
    project_urls={
        "Bug Tracker": "https://github.com/kkoutini/passt_hear21/issues",
        "Source Code": "https://github.com/kkoutini/passt_hear21",
    },
    packages=find_packages(exclude=("tests",)),
    python_requires=">=3.7",
    install_requires=["timm>=0.4.12", "torchaudio>=0.7.0"],
)

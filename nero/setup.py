# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from setuptools import find_namespace_packages, setup

setup(
    name="hydra-configs-nero",
    version="0.1.1",
    packages=find_namespace_packages(include=["hydra_configs*"]),
    author=["Rosario Scalise"],
    author_email=["rosario@cs.uw.edu"],
    url="http://github.com/romesco/nero",
    include_package_data=True,
    install_requires=[
        "omegaconf",
    ],
)

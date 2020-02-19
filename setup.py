# example cobaya-compliant SO likelihood package;
# adapted from github.com/cobayasampler/example_external_likelihood

from setuptools import setup
import os

setup(
    name="solike",
    version="0.0",
    description="Prototype package for SO Likelihoods",
    zip_safe=False,
    packages=["solike", "solike.tests", "solike.clusters"],
    package_data={
        "solike": ["*.yaml", "*.bibtex"],
        "solike": ["data/simulated*/*.txt"],
        "solike": ["clusters/data/*"],
        "solike": ["clusters/data/selFn_equD56/*"],
    },
    # install_requires=['cobaya (>=2.0.5)'],
    test_suite="solike.tests",
)

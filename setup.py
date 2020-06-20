# example cobaya-compliant SO likelihood package;
# adapted from github.com/cobayasampler/example_external_likelihood

from setuptools import setup

setup(
    name="solike",
    version="0.0",
    description="Prototype package for SO Likelihoods",
    zip_safe=False,
    packages=["solike", "solike.tests", "solike.clusters"],
    package_data={
        "solike": [
            "*.yaml",
            "*.bibtex",
            "data/simulated*/*.txt",
            "clusters/data/*",
            "clusters/data/selFn_equD56/*",
        ]
    },
    install_requires=["cobaya (>=3.0)", "numpy (>=1.16.0)", "astropy"],
    test_suite="solike.tests",
)

# example cobaya-compliant SO likelihood package;
# adapted from github.com/cobayasampler/example_external_likelihood

from setuptools import setup

setup(
    name="solt",
    version="0.0",
    description="Prototype package for SO Likelihoods",
    zip_safe=False,
    packages=["solt", "solt.tests", "solt.clusters"],
    package_data={
        "solt": [
            "*.yaml",
            "*.bibtex",
            "data/simulated*/*.txt",
            "clusters/data/*",
            "clusters/data/selFn_equD56/*",
        ]
    },
    install_requires=["cobaya (>=3.0)", "astropy", "astLib", "scikit-learn"],
    test_suite="solt.tests",
)

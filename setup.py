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
    install_requires=[
        "astropy",
        "astLib",
        "scikit-learn",
        "fgspectra @ git+https://github.com/simonsobs/fgspectra@master#egg=fgspectra",
        "cobaya>=3.0",
        "sacc @ git+https://github.com/simonsobs/sacc@mflike_current#egg=sacc",
    ],
    test_suite="solt.tests",
)

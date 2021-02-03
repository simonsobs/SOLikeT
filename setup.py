# example cobaya-compliant SO likelihood package;
# adapted from github.com/cobayasampler/example_external_likelihood

from setuptools import setup

setup(
    name="soliket",
    version="0.0",
    description="SO Likelihoods & Theories",
    zip_safe=False,
    packages=["soliket", "soliket.tests", "soliket.clusters"],
    package_data={
        "soliket": [
            "*.yaml",
            "*.bibtex",
            "data/simulated*/*.txt",
            "clusters/data/*",
            "clusters/data/selFn_equD56/*",
        ]
    },
    install_requires=[
        "astropy",
        "scikit-learn",
        "fgspectra @ git+https://github.com/simonsobs/fgspectra@master#egg=fgspectra",
        "cobaya @ git+https://github.com/cobayasampler/cobaya",  # for now
        "sacc @ git+https://github.com/simonsobs/sacc@mflike_current#egg=sacc",
        "mflike @ git+https://github.com/simonsobs/lat_mflike"
    ],
    test_suite="soliket.tests",
)

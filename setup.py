# example cobaya-compliant SO likelihood package;
# adapted from github.com/cobayasampler/example_external_likelihood

from setuptools import setup

setup(
    name="soliket",
    version="0.0",
    description="SO Likelihoods & Theories",
    zip_safe=False,
    packages=["soliket", "soliket.tests", "soliket.clusters", "soliket.binned_clusters"],
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
        "cobaya",
        "sacc",
        #"pyccl",
        "fgspectra @ git+https://github.com/simonsobs/fgspectra@master#egg=fgspectra",
        "mflike @ git+https://github.com/simonsobs/lat_mflike"
    ],
    test_suite="soliket.tests",
)

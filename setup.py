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
            # "data/simulated*/*.txt",
            "clusters/data/*",
            "clusters/data/selFn_equD56/*",
            "lensing/data/*.txt",
        ]
    },
    install_requires=[
        "astropy",
        "scikit-learn",
        "cobaya",
        "sacc",
        "pyccl",
        "fgspectra @ git+https://github.com/simonsobs/fgspectra@act_sz_x_cib#egg=fgspectra", # noqa E501
        "mflike @ git+https://github.com/simonsobs/lat_mflike@master"
    ],
    extras_requires=[
        "cosmopower"
    ],
    test_suite="soliket.tests",
    include_package_data=True,
)

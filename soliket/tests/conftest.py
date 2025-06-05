import pytest


@pytest.fixture
def test_cosmology_params():
    params = {}
    params["As"] = 2.15086031154146e-9
    params["ns"] = 0.9625356
    params["ombh2"] = 0.02219218
    params["omch2"] = 0.1203058
    params["H0"] = 67.02393
    params["tau"] = 0.06574325
    params["nnu"] = 3.04
    params["mnu"] = 0.06
    return params


@pytest.fixture
def evaluate_one_info():
    info = {}
    info["likelihood"] = {"one": None}
    info["sampler"] = {"evaluate": None}
    info["debug"] = True
    return info


@pytest.fixture
def check_skip_pyccl():
    """
    Check if the pyCCL module can be imported, otherwise skip the tests.
    """
    pytest.importorskip(modname="pyccl", reason="Couldn't import 'pyccl' module")


@pytest.fixture
def check_skip_cosmopower():
    """
    Check if the CosmoPower module can be imported, otherwise skip the tests.
    """
    pytest.importorskip(
        modname="cosmopower", reason="Couldn't import 'cosmopower' module"
    )


@pytest.fixture
def install_planck_lite():
    """
    Install the Planck 2018 high-l multipoles likelihood.
    """
    from cobaya.install import install

    install(
        {"likelihood": {"planck_2018_highl_plik.TTTEEE_lite_native": None}},
        path=None,
        skip_global=False,
        force=False,
        debug=True,
        no_set_global=True,
    )

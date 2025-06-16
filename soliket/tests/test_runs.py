import pkgutil

import pytest
from cobaya.install import install
from cobaya.run import run
from cobaya.tools import resolve_packages_path
from cobaya.yaml import yaml_load

packages_path = resolve_packages_path()


@pytest.mark.parametrize(
    "lhood",
    [
        "lensing",
        "multi",
    ],
)
def test_installation(lhood):
    if lhood == "lensing":
        from soliket import LensingLikelihood

        is_installed = LensingLikelihood.is_installed(
            path=packages_path,
        )
        assert is_installed is True, (
            "LensingLikelihood is not installed! Please install it using "
            "'cobaya-install soliket.LensingLikelihood'"
        )

    elif lhood == "multi":
        _ = pytest.importorskip("mflike", reason="Couldn't import 'mflike' module")
        import mflike

        is_installed = mflike.TTTEEE.is_installed(
            path=packages_path,
        )
        assert is_installed is True, (
            "mflike.TTTEEE is not installed! Please install it using "
            "'cobaya-install mflike.TTTEEE'"
        )


@pytest.mark.parametrize(
    "lhood",
    [
        "lensing",
        "lensing_lite",
        "multi",
        # "galaxykappa",
        # "shearkappa"
        # "xcorr"
    ],
)
def test_evaluate(lhood):
    info = yaml_load(pkgutil.get_data("soliket", f"tests/test_{lhood}.yaml"))
    info["force"] = True
    info["sampler"] = {"evaluate": {}}

    if lhood == "multi":
        pytest.importorskip("mflike", reason="Couldn't import 'mflike' module")

    install(info, path=packages_path, skip_global=True, no_set_global=True)

    updated_info, sampler = run(info)


@pytest.mark.parametrize(
    "lhood",
    [
        "lensing",
        "lensing_lite",
        "multi",
        # "galaxykappa",
        # "shearkappa"
        # "xcorr"
    ],
)
def test_mcmc(lhood):
    info = yaml_load(pkgutil.get_data("soliket", f"tests/test_{lhood}.yaml"))
    info["force"] = True
    info["sampler"] = {"mcmc": {"max_samples": 10, "max_tries": 1000}}

    if lhood == "multi":
        pytest.importorskip("mflike", reason="Couldn't import 'mflike' module")

    install(info, path=packages_path, skip_global=True, no_set_global=True)

    updated_info, sampler = run(info)

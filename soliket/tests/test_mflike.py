import importlib

from cobaya.tools import resolve_packages_path

packages_path = resolve_packages_path()


def test_mflike_import():
    _ = importlib.import_module("mflike")
    _ = importlib.import_module("mflike").EE
    _ = importlib.import_module("mflike").TE
    _ = importlib.import_module("mflike").TT
    _ = importlib.import_module("mflike").TTTEEE
    _ = importlib.import_module("mflike").BandpowerForeground
    _ = importlib.import_module("mflike").Foreground


def test_mflike_install(request):
    from cobaya.install import install

    mflike_options = {
        "input_file": "LAT_simu_sacc_00044.fits",
        "cov_Bbl_file": "data_sacc_w_covar_and_Bbl.fits",
        "stop_at_error": True,
    }

    install(
        {"likelihood": {"mflike.TTTEEE": mflike_options}},
        path=packages_path,
        skip_global=False,
        force=False,
        debug=True,
        no_set_global=True,
    )

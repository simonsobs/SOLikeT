from cobaya.tools import resolve_packages_path

packages_path = resolve_packages_path()


def test_mflike_import():

    import mflike  # noqa: F401

    from mflike import BandpowerForeground, Foreground # noqa: F401

    from mflike import TTTEEE, TT, EE, TE # noqa: F401


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

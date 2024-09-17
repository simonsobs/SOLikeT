def test_soliket_import():
    import soliket  # noqa: F401


def test_soliket_version():
    import soliket

    assert soliket.__version__

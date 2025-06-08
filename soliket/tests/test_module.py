import importlib


def test_soliket_import():
    _ = importlib.import_module("soliket")


def test_soliket_version():
    import soliket

    assert soliket.__version__

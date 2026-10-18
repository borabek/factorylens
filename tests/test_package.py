import factorylens


def test_version_is_set():
    assert factorylens.__version__.count(".") == 2

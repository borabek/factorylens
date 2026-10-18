import pytest
from fakes import make_fake_root


@pytest.fixture
def root(tmp_path):
    return make_fake_root(tmp_path)

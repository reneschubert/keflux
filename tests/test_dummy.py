import pytest
from keflux.dummy_module import dummy_foo


def test_dummy():
    assert dummy_foo(4) == (4 + 4)

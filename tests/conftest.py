"""A test that hangs must fail loudly, not freeze the suite: this package is all threads."""

import faulthandler

import pytest


@pytest.fixture(autouse=True)
def _no_test_may_hang():
    faulthandler.dump_traceback_later(60, exit=True)  # prints every thread's stack, then exits
    yield
    faulthandler.cancel_dump_traceback_later()

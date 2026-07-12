from __future__ import annotations

import pytest


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "smoke: end-to-end example script smoke tests (needs optional deps)")

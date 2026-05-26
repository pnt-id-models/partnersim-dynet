"""Smoke test: package imports and version is exposed."""

import partnersim_dynet


def test_package_imports():
    assert partnersim_dynet.__version__ == "0.1.0"

"""Repo-level test configuration.

The root conftest.py (one directory up) still loads first and keeps its job of
redirecting DATABASE_URL to a temporary SQLite file; nothing here touches it.
"""

from __future__ import annotations


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "e2e: runs the real pipeline on a real clip with the trained models "
        "(skips automatically when those local assets are absent)",
    )

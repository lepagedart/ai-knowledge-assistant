"""Regression coverage for Flask runtime package-data declarations."""

from __future__ import annotations

import tomllib
from pathlib import Path


def test_flask_runtime_assets_are_narrowly_declared_as_package_data() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))

    package_data = pyproject["tool"]["setuptools"]["package-data"]

    assert package_data == {
        "ai_knowledge_assistant": [
            "templates/*.html",
            "static/*.css",
            "static/*.js",
        ]
    }

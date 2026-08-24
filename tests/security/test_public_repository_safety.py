"""Checks that the public-repository safety foundation remains intact."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_env_example_is_keyless() -> None:
    env_example = ROOT / ".env.example"
    assert env_example.is_file()
    documented_non_secret_defaults = {"OPENAI_ANSWER_MODEL=gpt-5.6-luna"}
    for line in env_example.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        assert line in documented_non_secret_defaults or re.fullmatch(
            r"[A-Z0-9_]+=", line
        )


def test_gitignore_covers_private_and_runtime_data() -> None:
    ignored = (ROOT / ".gitignore").read_text(encoding="utf-8")
    required_entries = {
        ".env",
        ".env.*",
        "!.env.example",
        ".venv/",
        "venv/",
        "uploads/",
        "storage/",
        "instance/",
        "tmp/",
        "temp/",
        "indexes/",
        "vector_indexes/",
        "artifacts/",
        "generated/",
        "*.sqlite",
        "*.sqlite3",
        "*.db",
        "*.faiss",
        "*.npy",
        "*.npz",
        "*.jsonl",
        "__pycache__/",
        ".pytest_cache/",
        ".ruff_cache/",
        ".coverage",
        "htmlcov/",
        "build/",
        "dist/",
    }
    assert required_entries.issubset(set(ignored.splitlines()))

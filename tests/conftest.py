"""Suite-wide safeguards for deterministic, offline tests."""

import socket

import pytest


@pytest.fixture(autouse=True)
def block_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail fast if a test accidentally attempts external network access."""

    def blocked(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AssertionError("Network access is not permitted in this test suite.")

    monkeypatch.setattr(socket, "create_connection", blocked)
    monkeypatch.setattr(socket.socket, "connect", blocked)

import socket
from unittest.mock import Mock

import requests
import pytest


@pytest.fixture(autouse=True)
def block_unmocked_network(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("Real network access is forbidden in tests")

    monkeypatch.setattr(requests.sessions.Session, "request", fail)


@pytest.fixture(autouse=True)
def fake_public_dns(monkeypatch):
    """Resolve every hostname to a public IP unless a test overrides it."""
    def public(host, port, *args, **kwargs):
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", port))]

    monkeypatch.setattr(socket, "getaddrinfo", public)


@pytest.fixture(autouse=True)
def redirect_provenance_log(monkeypatch, tmp_path):
    from common import provenance

    monkeypatch.setattr(provenance, "LOG_PATH", tmp_path / "requests.jsonl")
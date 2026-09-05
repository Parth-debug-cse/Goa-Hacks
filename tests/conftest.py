import requests
import pytest


@pytest.fixture(autouse=True)
def block_unmocked_network(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("Real network access is forbidden in tests")

    monkeypatch.setattr(requests.sessions.Session, "request", fail)

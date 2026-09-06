import json
import re

import pytest

from common import provenance


def _point_log_at(monkeypatch, tmp_path):
    log_path = tmp_path / "evidence" / "requests.jsonl"
    monkeypatch.setattr(provenance, "LOG_PATH", log_path)
    return log_path


def test_log_request_writes_jsonl_line_and_returns_id(monkeypatch, tmp_path):
    log_path = _point_log_at(monkeypatch, tmp_path)
    provenance_id = provenance.log_request(
        "serpapi_exact", "GET", "https://serpapi.com/search",
        {"engine": "google_lens", "q": "alice"}, 200, 12.3, 4096,
    )
    assert re.fullmatch(r"p_\d{4}", provenance_id)
    lines = log_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["provenance_id"] == provenance_id
    assert record["engine"] == "serpapi_exact"
    assert record["method"] == "GET"
    assert record["url"] == "https://serpapi.com/search"
    assert record["params"]["q"] == "alice"
    assert record["status"] == 200
    assert record["latency_ms"] == 12.3
    assert record["response_bytes"] == 4096
    assert record["error"] is None
    assert record["timestamp_utc"]


@pytest.mark.parametrize("secret_key", [
    "api_key", "key", "authorization", "x-api-key", "X-Api-Key", "Ocp-Apim-Subscription-Key",
])
def test_secret_keys_are_redacted_case_insensitively(monkeypatch, tmp_path, secret_key):
    log_path = _point_log_at(monkeypatch, tmp_path)
    provenance.log_request(
        "test", "GET", "https://example.com",
        {secret_key: "SECRETVALUE", "q": "ok"}, 200, 1.0, 10,
    )
    record = json.loads(log_path.read_text(encoding="utf-8").splitlines()[0])
    assert record["params"][secret_key] == "[REDACTED]"
    assert record["params"]["q"] == "ok"


def test_provenance_ids_are_monotonic(monkeypatch, tmp_path):
    _point_log_at(monkeypatch, tmp_path)
    first = provenance.log_request("a", "GET", "https://example.com/a", {}, 200, 1.0, 1)
    second = provenance.log_request("b", "GET", "https://example.com/b", {}, 200, 1.0, 1)
    assert second > first


def test_log_request_records_error(monkeypatch, tmp_path):
    log_path = _point_log_at(monkeypatch, tmp_path)
    provenance.log_request("x", "GET", "https://example.com", {}, None, 0.0, None, error="boom")
    record = json.loads(log_path.read_text(encoding="utf-8").splitlines()[0])
    assert record["error"] == "boom"
    assert record["status"] is None


def test_log_request_tolerates_non_serializable_values(monkeypatch, tmp_path):
    log_path = _point_log_at(monkeypatch, tmp_path)
    provenance.log_request("x", "POST", "https://example.com", {"obj": object()}, 200, 1.0, 10)
    record = json.loads(log_path.read_text(encoding="utf-8").splitlines()[0])
    assert record["provenance_id"]
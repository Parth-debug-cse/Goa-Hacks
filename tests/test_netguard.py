import socket
from unittest.mock import Mock, patch

import pytest

from common.netguard import UnsafeURLError, assert_public_url


def _patch_dns(monkeypatch, ips):
    monkeypatch.setattr(
        "socket.getaddrinfo",
        lambda host, port, *args, **kwargs: [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, port)) for ip in ips
        ],
    )


def test_accepts_public_http_url(monkeypatch):
    _patch_dns(monkeypatch, ["93.184.216.34"])
    assert_public_url("https://example.com/path")


@pytest.mark.parametrize("url", [
    "ftp://example.com/file",
    "example.com/path",
    "https://",
])
def test_rejects_non_http_or_missing_scheme(monkeypatch, url):
    _patch_dns(monkeypatch, ["93.184.216.34"])
    with pytest.raises(UnsafeURLError):
        assert_public_url(url)


def test_rejects_userinfo(monkeypatch):
    _patch_dns(monkeypatch, ["93.184.216.34"])
    with pytest.raises(UnsafeURLError):
        assert_public_url("https://user:pass@example.com/x")


@pytest.mark.parametrize("ip", ["127.0.0.1", "::1", "::ffff:127.0.0.1"])
def test_rejects_loopback(monkeypatch, ip):
    _patch_dns(monkeypatch, [ip])
    with pytest.raises(UnsafeURLError):
        assert_public_url("https://example.com/x")


def test_rejects_link_local(monkeypatch):
    _patch_dns(monkeypatch, ["169.254.169.254"])
    with pytest.raises(UnsafeURLError):
        assert_public_url("https://example.com/meta")


def test_rejects_rfc1918_private(monkeypatch):
    _patch_dns(monkeypatch, ["10.0.0.1"])
    with pytest.raises(UnsafeURLError):
        assert_public_url("https://example.com/internal")


def test_rejects_reserved(monkeypatch):
    _patch_dns(monkeypatch, ["240.0.0.1"])
    with pytest.raises(UnsafeURLError):
        assert_public_url("https://example.com/x")


def test_rejects_multicast(monkeypatch):
    _patch_dns(monkeypatch, ["224.0.0.1"])
    with pytest.raises(UnsafeURLError):
        assert_public_url("https://example.com/x")


def test_rejects_unspecified(monkeypatch):
    _patch_dns(monkeypatch, ["0.0.0.0"])
    with pytest.raises(UnsafeURLError):
        assert_public_url("https://example.com/x")


def test_rejects_when_any_resolved_address_is_unsafe(monkeypatch):
    _patch_dns(monkeypatch, ["93.184.216.34", "192.168.1.1"])
    with pytest.raises(UnsafeURLError):
        assert_public_url("https://example.com/x")


def test_rejects_unresolvable_hostname(monkeypatch):
    def fail(host, port, *args, **kwargs):
        raise socket.gaierror("no such host")

    monkeypatch.setattr("socket.getaddrinfo", fail)
    with pytest.raises(UnsafeURLError):
        assert_public_url("https://example.com/x")


def test_unsafe_candidate_is_rejected_and_verification_continues(monkeypatch):
    from stage2_search import CandidateURL
    from stage3_verify import process_verification

    def dns(host, port, *args, **kwargs):
        ip = "127.0.0.1" if "private" in host else "93.184.216.34"
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, port))]

    monkeypatch.setattr("socket.getaddrinfo", dns)

    private = CandidateURL("https://private.example.com/internal")
    public = CandidateURL("https://public.example.com/profile")

    with patch("stage3_verify._get_face_analyzer", return_value=Mock()), patch(
        "stage3_verify.fetch_candidate_images",
        side_effect=lambda url: [(url + "/img.jpg", b"data")],
    ), patch(
        "stage3_verify.verify_image",
        return_value=(True, {"arcface_cosine_similarity": 0.9, "adaface_cosine_similarity": 0.9}, "accepted"),
    ):
        result = process_verification(
            [private, public],
            {"quality_details": {}, "arcface_embedding": [1], "adaface_embedding": [1]},
        )

    assert result["match_found"] is True
    assert result["matched_page_url"] == public.url
    assert result["candidates_tried"] == 2
    assert result["candidates_rejected"] == [{"url": private.url, "reason": "unsafe_url"}]
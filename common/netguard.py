"""Basic SSRF guard for URLs returned by third-party search APIs.

:func:`assert_public_url` verifies a URL is fetchable before the pipeline ever
touches it: http/https only, no embedded credentials, and *every* address the
hostname resolves to must be a globally routable (public) IP. Loopback,
link-local, RFC1918-private, reserved, multicast, and unspecified addresses are
all treated as unsafe.
"""

from __future__ import annotations

import ipaddress
import socket
from urllib.parse import urlsplit
from typing import Union

class UnsafeURLError(Exception):
    """Raised when a URL is not safe for the pipeline to fetch."""


def _is_public_ip(ip: Union[ipaddress.IPv4Address, ipaddress.IPv6Address]) -> bool:
    # IPv4-mapped IPv6 literals (e.g. ::ffff:127.0.0.1) are classified by the
    # embedded IPv4 address, otherwise loopback traffic could slip through.
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        ip = ip.ipv4_mapped
    return not (
        ip.is_loopback
        or ip.is_link_local
        or ip.is_private
        or ip.is_reserved
        or ip.is_multicast
        or ip.is_unspecified
    )


def assert_public_url(url: str) -> None:
    """Raise :class:`UnsafeURLError` unless ``url`` is safe to fetch.

    The hostname is resolved with ``socket.getaddrinfo`` and *all* returned
    addresses are checked — a single non-public address rejects the URL.
    """
    parts = urlsplit(url or "")
    scheme = parts.scheme.lower()
    if scheme not in {"http", "https"}:
        raise UnsafeURLError(f"scheme {parts.scheme!r} not allowed (http/https only)")
    host = parts.hostname
    if not host:
        raise UnsafeURLError("URL has no hostname")
    if "@" in parts.netloc:
        raise UnsafeURLError("URL must not contain userinfo")
    if parts.port is not None and not (1 <= parts.port <= 65535):
        raise UnsafeURLError(f"invalid port {parts.port}")

    try:
        addresses = socket.getaddrinfo(host, None)
    except socket.gaierror as error:
        raise UnsafeURLError(f"could not resolve hostname {host}") from error
    if not addresses:
        raise UnsafeURLError(f"hostname {host} resolved to no addresses")

    for entry in addresses:
        raw_address = entry[4][0]
        try:
            ip = ipaddress.ip_address(raw_address)
        except ValueError as error:
            raise UnsafeURLError(
                f"non-IP address {raw_address!r} resolved for {host}"
            ) from error
        if not _is_public_ip(ip):
            raise UnsafeURLError(
                f"non-public address {raw_address} resolved for {host}"
            )
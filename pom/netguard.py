"""SSRF guard ensuring URLs resolve only to safe public IPv4/IPv6 addresses (§3)."""

from __future__ import annotations

import ipaddress
import socket
from urllib.parse import urlsplit


class UnsafeURLError(ValueError):
    """Raised when a candidate URL targets a private/local/reserved IP."""


def is_public_ip(address: str) -> bool:
    """Return True if the IP address is globally reachable and not private/loopback/reserved."""
    try:
        ip = ipaddress.ip_address(address)
    except ValueError:
        return False
    return not (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_reserved
        or ip.is_multicast
        or ip.is_unspecified
    )


def assert_public_url(url: str) -> None:
    """Validate that the given URL uses http(s), has no userinfo, and resolves to public IPs.
    
    Raises UnsafeURLError on any safety violation or DNS failure.
    """
    parts = urlsplit(url.strip())
    if parts.scheme.lower() not in {"http", "https"}:
        raise UnsafeURLError(f"Unsupported scheme: {parts.scheme}")
    
    if parts.username or parts.password:
        raise UnsafeURLError("URL contains embedded userinfo (username/password)")

    hostname = parts.hostname
    if not hostname:
        raise UnsafeURLError("URL has no hostname")

    port = parts.port or (443 if parts.scheme.lower() == "https" else 80)

    try:
        addr_info = socket.getaddrinfo(hostname, port, proto=socket.IPPROTO_TCP)
    except socket.gaierror as err:
        raise UnsafeURLError(f"DNS resolution failed for {hostname}: {err}") from err

    if not addr_info:
        raise UnsafeURLError(f"No IP addresses resolved for {hostname}")

    for entry in addr_info:
        sockaddr = entry[4]
        ip_str = sockaddr[0]
        if not is_public_ip(ip_str):
            raise UnsafeURLError(f"Hostname {hostname} resolves to non-public IP: {ip_str}")

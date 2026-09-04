"""Capture matched content and create hash-linked metadata."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import requests
from PIL import Image


def _embedding_sha256(face_embedding) -> str:
    """Compute SHA-256 over a face embedding byte representation.

    Args:
        face_embedding (Any): Numeric embedding array-like object.

    Returns:
        str: Hex digest of the embedding bytes.

    Raises:
        ValueError: If the embedding cannot be converted to bytes.
    """
    try:
        embedding_bytes = np.asarray(face_embedding, dtype=np.float32).tobytes()
    except Exception as error:
        raise ValueError(f"Invalid face embedding input: {error}") from error
    return hashlib.sha256(embedding_bytes).hexdigest()


def _capture_screenshot(url: str, screenshot_path: Path) -> bytes | None:
    """Attempt to render and screenshot a URL using Playwright Chromium.

    Args:
        url (str): URL to open in a headless browser.
        screenshot_path (Path): Output PNG path.

    Returns:
        bytes | None: Screenshot bytes when successful, else None.

    Raises:
        None. Failures are logged and represented via None fallback.
    """
    try:
        # playwright install chromium must be run once after pip install playwright
        from playwright.sync_api import sync_playwright
    except Exception as error:
        print(f"Screenshot fallback unavailable (Playwright import failed): {error}")
        return None

    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto(url, timeout=15_000)
            page.screenshot(path=str(screenshot_path), full_page=True)
            browser.close()
        screenshot_bytes = screenshot_path.read_bytes()
        with Image.open(screenshot_path) as screenshot_image:
            screenshot_image.verify()
        return screenshot_bytes
    except Exception as error:
        print(f"Screenshot fallback failed for {url}: {error}")
        return None


def capture_post(url, face_embedding, out_dir="captured"):
    """Capture remote post content and generate metadata tied to content hash.

    Args:
        url (str): Matched source URL.
        face_embedding (Any): Embedding array used to derive embedding hash.
        out_dir (str): Directory where capture artifacts are written.

    Returns:
        tuple[bytes, str, dict]: Content hash bytes, metadata JSON path, and metadata object.

    Raises:
        OSError: If metadata cannot be written to disk.
        ValueError: If face embedding hashing fails.
    """
    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).isoformat()
    embedding_hash = _embedding_sha256(face_embedding)

    response_body: bytes | None = None
    status_code: int | None = None
    failure_reason: str | None = None

    try:
        response = requests.get(url, timeout=15)
        status_code = response.status_code
        if response.status_code >= 400:
            failure_reason = f"HTTP status {response.status_code}"
        elif len(response.content) < 500:
            failure_reason = f"HTTP body too small ({len(response.content)} bytes)"
        else:
            response_body = response.content
    except requests.RequestException as error:
        failure_reason = f"Request failed: {error}"

    screenshot_path = output_dir / "screenshot.png"
    capture_mode = "HTTP"

    if response_body is None:
        screenshot_bytes = _capture_screenshot(url, screenshot_path)
        if screenshot_bytes is not None:
            response_body = screenshot_bytes
            capture_mode = "SCREENSHOT"
        else:
            fallback = f"{url}|{timestamp}|{embedding_hash}".encode("utf-8")
            response_body = fallback
            capture_mode = "UNREACHABLE"

    content_hash = hashlib.sha256(response_body).digest()

    metadata = {
        "source_url": url,
        "captured_at": timestamp,
        "face_embedding_sha256": embedding_hash,
        "content_hash_hex": content_hash.hex(),
        "capture_mode": capture_mode,
        "http_status_code": status_code,
        "failure_reason": failure_reason,
    }

    if screenshot_path.exists():
        metadata["screenshot_path"] = str(screenshot_path)

    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return content_hash, str(metadata_path), metadata

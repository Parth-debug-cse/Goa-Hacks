"""Image compression and optional EXIF extraction."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

from PIL import Image

EXIF_NAMES = {
    36867: "DateTimeOriginal",
    271: "Make",
    272: "Model",
    34853: "GPSInfo",
}


def compress_for_upload(path: str | Path, max_bytes: int = 500_000) -> bytes:
    """Return JPEG bytes no larger than ``max_bytes`` where possible."""
    with Image.open(path) as source:
        image = source.convert("RGB")
        quality = 90
        longest_edge = max(image.size)
        while True:
            candidate = image
            if longest_edge < max(image.size):
                scale = longest_edge / max(image.size)
                candidate = image.resize(
                    (max(1, int(image.width * scale)), max(1, int(image.height * scale))),
                    Image.Resampling.LANCZOS,
                )
            output = io.BytesIO()
            candidate.save(output, format="JPEG", quality=quality, optimize=True)
            data = output.getvalue()
            if len(data) <= max_bytes:
                return data
            if quality > 40:
                quality -= 5
            else:
                longest_edge = int(longest_edge * 0.85)
                quality = 85
                if longest_edge < 64:
                    raise ValueError(
                        f"Could not compress image below {max_bytes} bytes without "
                        "reducing it to an unusable size"
                    )


def extract_exif(path: str | Path) -> dict[str, Any]:
    """Extract only the agreed optional EXIF fields."""
    try:
        with Image.open(path) as image:
            exif = image.getexif()
            result: dict[str, Any] = {}
            for tag, name in EXIF_NAMES.items():
                value = exif.get(tag)
                if name == "GPSInfo" and value is not None:
                    try:
                        value = {
                            str(key): str(item)
                            for key, item in exif.get_ifd(tag).items()
                        }
                    except (AttributeError, KeyError, TypeError):
                        value = str(value)
                if value is not None:
                    result[name] = value if isinstance(value, dict) else str(value)
            return result
    except (OSError, ValueError):
        return {}

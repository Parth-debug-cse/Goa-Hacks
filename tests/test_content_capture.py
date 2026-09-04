from pathlib import Path
from unittest.mock import patch

from src.content_capture import capture_post


class _FakeResponse:
    status_code = 200
    content = b"hello"


def test_capture_post_writes_metadata_and_hash(tmp_path):
    out_dir = tmp_path / "captured"

    with patch("src.content_capture.requests.get", return_value=_FakeResponse()):
        content_hash, metadata_path, metadata = capture_post(
            "https://example.com/post", [0.1, 0.2, 0.3], out_dir=str(out_dir)
        )

    assert isinstance(content_hash, bytes)
    assert len(content_hash) == 32
    assert Path(metadata_path).exists()
    assert metadata["source_url"] == "https://example.com/post"

from types import SimpleNamespace

import pytest

from src import web_search


def test_find_matching_posts_prioritizes_social_domain(tmp_path, monkeypatch):
    image_path = tmp_path / "input.jpg"
    image_path.write_bytes(b"image-bytes")

    class FakeClient:
        def web_detection(self, image):
            del image
            return SimpleNamespace(
                error=SimpleNamespace(message=""),
                web_detection=SimpleNamespace(
                    pages_with_matching_images=[
                        SimpleNamespace(url="https://example.com/post", score=0.95),
                        SimpleNamespace(url="https://instagram.com/p/abc", score=0.10),
                    ]
                ),
            )

    fake_vision = SimpleNamespace(
        ImageAnnotatorClient=lambda: FakeClient(),
        Image=lambda content: SimpleNamespace(content=content),
    )

    monkeypatch.setattr(web_search, "vision", fake_vision)

    results = web_search.find_matching_posts(str(image_path))

    assert results[0]["url"] == "https://instagram.com/p/abc"
    assert results[0]["is_social"] is True

import time
from unittest.mock import Mock, patch

from PIL import Image

from common.image_utils import compress_for_upload
from stage2_search import (
    CandidateURL,
    _filter_candidates,
    build_search_query,
    merge_candidates,
    process_search,
    search_bing_text,
    search_serpapi,
)


def test_serpapi_exact_and_visual_are_parsed():
    upload = Mock()
    upload.json.return_value = {"image_id": "abc"}
    exact = Mock()
    exact.json.return_value = {"exact_matches": [{"title": "Profile", "link": "https://example.com/p", "thumbnail": "t"}]}
    visual = Mock()
    visual.json.return_value = {"visual_matches": [{"title": "Other", "link": "https://example.com/o"}]}
    session = Mock()
    session.post.return_value = upload
    session.get.side_effect = [exact, visual]
    with patch.dict("os.environ", {"SERPAPI_API_KEY": "secret"}), patch(
        "stage2_search.create_session", return_value=session
    ):
        results = search_serpapi(b"jpeg")
    assert [item.source_engine for item in results] == ["serpapi_exact", "serpapi_visual"]


def test_serpapi_error_degrades_to_empty():
    with patch.dict("os.environ", {"SERPAPI_API_KEY": "secret"}), patch(
        "stage2_search.create_session", side_effect=RuntimeError("offline")
    ):
        assert search_serpapi(b"jpeg") == []


def test_merge_deduplicates_tracking_urls_and_prefers_exact():
    result = merge_candidates(
        [CandidateURL("HTTPS://Example.com/profile/?utm_source=x", source_engine="serpapi_visual")],
        [CandidateURL("https://example.com/profile", source_engine="serpapi_exact", match_confidence_hint="exact")],
    )
    assert len(result) == 1
    assert result[0].source_engine == "serpapi_exact"


def test_bing_missing_key_is_skipped():
    from stage2_search import search_bing_visual
    with patch.dict("os.environ", {}, clear=True):
        assert search_bing_visual(b"jpeg") == []


def test_process_search_returns_at_timeout_without_waiting_for_slow_branch():
    def slow_branch(_image):
        time.sleep(0.25)
        return []

    start = time.monotonic()
    with patch("stage2_search.compress_for_upload", return_value=b"jpeg"), patch(
        "stage2_search.extract_exif", return_value={}
    ), patch("stage2_search.search_serpapi", return_value=[]), patch(
        "stage2_search._vision_candidates", return_value=([], "")
    ), patch("stage2_search.search_bing_visual", side_effect=slow_branch):
        candidates, warnings = process_search("photo.jpg", timeout_seconds=0.03)
    elapsed = time.monotonic() - start
    assert candidates == []
    assert "search_timeout_budget_exceeded" in warnings
    assert elapsed < 0.15


def test_query_relevance_is_used_after_exact_and_social_priority():
    candidates = _filter_candidates(
        [
            CandidateURL("https://example.com/unrelated", title="Other", source_engine="serpapi_visual"),
            CandidateURL("https://example.com/alice", title="Alice Goa", source_engine="serpapi_visual"),
        ],
        query="Alice Goa",
    )
    assert candidates[0].url.endswith("/alice")


def test_domain_filter_does_not_match_lookalike_hosts():
    candidates = _filter_candidates(
        [
            CandidateURL("https://notlinkedin.com/profile", title="Fake"),
            CandidateURL("https://linkedin.com/in/person", title="Real"),
        ]
    )
    assert [candidate.url for candidate in candidates] == [
        "https://linkedin.com/in/person",
        "https://notlinkedin.com/profile",
    ]


def test_compress_for_upload_respects_limit(tmp_path):
    source = tmp_path / "large.png"
    Image.effect_noise((2400, 2400), 128).save(source)
    assert len(compress_for_upload(source, max_bytes=500_000)) <= 500_000


def test_malformed_bing_response_is_ignored():
    from stage2_search import search_bing_visual
    response = Mock()
    response.json.return_value = {"tags": [{"actions": [{"actionType": "PagesIncluding", "data": {}}]}]}
    session = Mock()
    session.post.return_value = response
    with patch.dict("os.environ", {"AZURE_BING_VISUAL_SEARCH_KEY": "secret"}), patch(
        "stage2_search.create_session", return_value=session
    ):
        assert search_bing_visual(b"jpeg") == []


def test_google_vision_pages_and_query_are_parsed(monkeypatch):
    class Entity:
        score = 0.9
        description = "Alice Goa"

    class Page:
        url = "https://example.com/post"
        page_title = "Alice post"

    class Label:
        label = "portrait"

    class Web:
        web_entities = [Entity()]
        pages_with_matching_images = [Page()]
        best_guess_labels = [Label()]

    class Response:
        web_detection = Web()

    class VisionImage:
        def __init__(self, content):
            self.content = content

    class Client:
        def web_detection(self, image):
            return Response()

    import types
    vision = types.SimpleNamespace(
        Image=VisionImage,
        ImageAnnotatorClient=lambda: Client(),
    )
    monkeypatch.setitem(__import__("sys").modules, "google", types.SimpleNamespace(cloud=types.SimpleNamespace(vision=vision)))
    monkeypatch.setitem(__import__("sys").modules, "google.cloud", types.SimpleNamespace(vision=vision))
    from stage2_search import _vision_candidates
    results, query = _vision_candidates(b"image")
    assert results[0].url == "https://example.com/post"
    assert query == "Alice Goa portrait"


def test_bing_text_fallback_parses_host_pages():
    response = Mock(status_code=200)
    response.json.return_value = {"value": [{"name": "Alice", "hostPageUrl": "https://linkedin.com/in/alice"}]}
    session = Mock()
    session.get.return_value = response
    with patch.dict("os.environ", {"AZURE_BING_VISUAL_SEARCH_KEY": "secret"}), patch(
        "stage2_search.create_session", return_value=session
    ):
        results = search_bing_text("Alice Goa")
    assert results[0].source_engine == "bing_text"
    assert session.get.call_args.kwargs["params"]["q"] == "Alice Goa"


def test_exif_is_included_in_search_query():
    assert build_search_query("Alice", {"Make": "Canon", "Model": "R5"}) == "Alice Canon R5"

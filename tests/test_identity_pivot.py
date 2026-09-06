import socket
from unittest.mock import Mock, patch

import pytest

from identity_pivot import (
    _build_hop2_queries,
    extract_identity_signals,
    fetch_page_html,
    run_hop2_search,
)
from stage2_search import CandidateURL, process_search


def _serpapi_reply() -> Mock:
    response = Mock(status_code=200)
    response.json.return_value = {
        "organic_results": [{"title": "Alice Goa", "link": "https://linkedin.com/in/alice-goa"}]
    }
    return response


# --------------------------------------------------------------------------- #
# extract_identity_signals
# --------------------------------------------------------------------------- #
def test_title_and_og_meta_signals():
    html = (
        "<title> Alice Goa — Portfolio </title>"
        '<meta property="og:title" content="Alice Goa">'
        '<meta property="og:site_name" content="Portfolio">'
        '<meta name="description" content="photography">'
    )
    signals = extract_identity_signals("https://example.com/a", html)
    types = {signal["signal_type"] for signal in signals}
    assert {"title", "og:title", "og:site_name", "og:description"} <= types
    assert all(signal["seed_page"] == "https://example.com/a" for signal in signals)


def test_jsonld_person_with_alternate_name_and_sameas():
    html = (
        '<script type="application/ld+json">'
        '{"@context": "https://schema.org", "@type": "Person", "name": "Alice Goa", '
        '"alternateName": "A. Goa", "sameAs": ["https://twitter.com/alice", "https://github.com/alice"]}'
        "</script>"
    )
    signals = extract_identity_signals("https://example.com/a", html)
    by_type: dict[str, list[str]] = {}
    for signal in signals:
        by_type.setdefault(signal["signal_type"], []).append(signal["value"])
    assert "Alice Goa" in by_type.get("jsonld_name", [])
    assert "A. Goa" in by_type.get("jsonld_alternate_name", [])
    assert {"https://twitter.com/alice", "https://github.com/alice"} <= set(by_type.get("sameAs", []))


def test_jsonld_graph_profile_page_ignores_other_types():
    html = (
        '<script type="application/ld+json">'
        '{"@graph": [{"@type": "Person", "name": "Bob C", "sameAs": "https://x.com/bobc"}, '
        '{"@type": "Organization", "name": "Ignored Org"}]}'
        "</script>"
    )
    signals = extract_identity_signals("https://example.com/a", html)
    names = [signal["value"] for signal in signals if signal["signal_type"] == "jsonld_name"]
    same_as = [signal["value"] for signal in signals if signal["signal_type"] == "sameAs"]
    assert names == ["Bob C"]
    assert same_as == ["https://x.com/bobc"]


def test_author_and_rel_profile_links():
    html = (
        '<meta name="author" content="Alice Goa">'
        '<a rel="me" href="https://x.com/alice">x</a>'
        '<link rel="author" href="https://example.com/about">'
    )
    signals = extract_identity_signals("https://example.com/a", html)
    by_type: dict[str, list[str]] = {}
    for signal in signals:
        by_type.setdefault(signal["signal_type"], []).append(signal["value"])
    assert "Alice Goa" in by_type.get("meta_author", [])
    assert {"https://x.com/alice", "https://example.com/about"} <= set(by_type.get("rel_profile", []))


def test_handle_mentions_in_visible_text_skip_scripts():
    html = "<body>Follow @alice_goa and @bob. Also <script>var x = '@ignored'</script></body>"
    signals = extract_identity_signals("https://example.com/a", html)
    handles = {signal["value"] for signal in signals if signal["signal_type"] == "handle"}
    assert "alice_goa" in handles
    assert "bob" in handles
    assert "ignored" not in handles


@pytest.mark.parametrize("page_url, platform", [
    ("https://www.linkedin.com/in/alice-goa", "linkedin"),
    ("https://x.com/alice", "twitter"),
    ("https://twitter.com/alice", "twitter"),
    ("https://instagram.com/alice.ga/", "instagram"),
    ("https://github.com/alice", "github"),
    ("https://facebook.com/alice.goa", "facebook"),
])
def test_url_slug_signals(page_url, platform):
    signals = extract_identity_signals(page_url, "<html></html>")
    slugs = [signal["value"] for signal in signals if signal["signal_type"] == f"url_slug:{platform}"]
    assert slugs, page_url


def test_url_slug_signals_skip_chrome_segments():
    signals = extract_identity_signals("https://x.com/home", "<html></html>")
    assert not any(signal["signal_type"] == "url_slug:twitter" for signal in signals)


def test_title_name_segments():
    html = "<title>Alice Goa | Photo | Stock | Blog</title>"
    signals = extract_identity_signals("https://example.com/a", html)
    names = [signal["value"] for signal in signals if signal["signal_type"] == "title_name"]
    assert names == ["Alice Goa"]


def test_title_name_skips_only_verbatim_site_name():
    html = (
        "<title>Goa Pixels — Alice Goa</title>"
        '<meta property="og:site_name" content="Goa Pixels">'
    )
    signals = extract_identity_signals("https://example.com/a", html)
    names = [signal["value"] for signal in signals if signal["signal_type"] == "title_name"]
    assert names == ["Alice Goa"]


# --------------------------------------------------------------------------- #
# run_hop2_search
# --------------------------------------------------------------------------- #
def test_run_hop2_search_builds_queries_and_parses(monkeypatch, tmp_path):
    from common import provenance

    monkeypatch.setattr(provenance, "LOG_PATH", tmp_path / "requests.jsonl")
    session = Mock()
    session.get.side_effect = lambda url, **kwargs: _serpapi_reply()
    with patch("identity_pivot.create_session", return_value=session):
        results = run_hop2_search(
            [{"signal_type": "title_name", "value": "Alice Goa", "seed_page": "https://example.com/blog"}],
            serpapi_key="secret", max_queries=2,
        )
    assert len(results) == 2
    for result in results:
        assert result.source_engine == "serpapi_hop2"
        assert result.search_hop == 2
        assert result.provenance_id
        assert result.match_confidence_hint == "exact"
        assert result.discovered_via["seed_page"] == "https://example.com/blog"
        assert result.discovered_via["signal_type"] == "title_name"
        assert result.discovered_via["query"]
    assert {result.url for result in results} == {"https://linkedin.com/in/alice-goa"}


def test_run_hop2_search_caps_queries(monkeypatch, tmp_path):
    from common import provenance

    monkeypatch.setattr(provenance, "LOG_PATH", tmp_path / "requests.jsonl")
    calls: list[str] = []
    session = Mock()

    def side_effect(url, **kwargs):
        calls.append(kwargs["params"]["q"])
        return _serpapi_reply()

    session.get.side_effect = side_effect
    signals: list[dict] = [
        {"signal_type": "title_name", "value": "Name One", "seed_page": "https://example.com/a"},
    ]
    for handle in ("alice", "bob", "carol", "dave", "erin"):
        signals.append({"signal_type": "handle", "value": handle, "seed_page": "https://example.com/a"})
    with patch("identity_pivot.create_session", return_value=session):
        run_hop2_search(signals, serpapi_key="secret", max_queries=3)
    assert len(calls) == 3


def test_run_hop2_search_requires_key():
    assert run_hop2_search(
        [{"signal_type": "title_name", "value": "Alice Goa", "seed_page": "https://example.com/a"}],
        serpapi_key="",
    ) == []


def test_build_hop2_queries_uses_sameas_and_url_slugs():
    signals = [
        {"signal_type": "sameAs", "value": "https://twitter.com/alice", "seed_page": "https://example.com/a"},
        {"signal_type": "url_slug:linkedin", "value": "alice-goa", "seed_page": "https://example.com/b"},
    ]
    texts = [query for query, _ in _build_hop2_queries(signals)]
    assert '"@alice" site:x.com' in texts
    assert '"alice-goa" site:linkedin.com/in' in texts


def test_fetch_page_html_returns_html_and_provenance(monkeypatch, tmp_path):
    from common import provenance

    monkeypatch.setattr(provenance, "LOG_PATH", tmp_path / "requests.jsonl")
    page = Mock(status_code=200, headers={"Content-Type": "text/html; charset=utf-8"},
                text="<title>Alice Goa</title>")
    session = Mock()
    session.get.return_value = page
    with patch("identity_pivot.create_session", return_value=session):
        html, provenance_id = fetch_page_html("https://example.com/page")
    assert html == "<title>Alice Goa</title>"
    assert provenance_id


def test_fetch_page_html_rejects_unsafe_url(monkeypatch, tmp_path):
    from common import provenance

    monkeypatch.setattr(provenance, "LOG_PATH", tmp_path / "requests.jsonl")
    monkeypatch.setattr(
        "socket.getaddrinfo",
        lambda host, port, *args, **kwargs: [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", port))
        ],
    )
    html, provenance_id = fetch_page_html("https://private.example.com/x")
    assert html is None
    assert provenance_id is None


# --------------------------------------------------------------------------- #
# process_search wiring
# --------------------------------------------------------------------------- #
def test_process_search_merges_hop2_candidates(monkeypatch, tmp_path):
    from common import provenance

    monkeypatch.setattr(provenance, "LOG_PATH", tmp_path / "requests.jsonl")
    hop1_results = [
        CandidateURL("https://example.com/visual", title="Visual", source_engine="serpapi_visual",
                     match_confidence_hint="visual", provenance_id="p_0101"),
        CandidateURL("https://linkedin.com/in/alice", title="Alice", source_engine="serpapi_exact",
                     match_confidence_hint="exact", provenance_id="p_0102"),
    ]
    hop2_results = [
        CandidateURL("https://github.com/alice", title="alice", source_engine="serpapi_hop2",
                     match_confidence_hint="exact", provenance_id="p_0201", search_hop=2,
                     discovered_via={"seed_page": "https://example.com/visual",
                                     "signal_type": "title_name", "query": '"Alice Goa" site:github.com'}),
        CandidateURL("https://linkedin.com/in/alice", title="Alice", source_engine="serpapi_hop2",
                     match_confidence_hint="exact", provenance_id="p_0202", search_hop=2,
                     discovered_via={"seed_page": "https://example.com/visual",
                                     "signal_type": "title_name", "query": '"Alice Goa" site:linkedin.com/in'}),
    ]
    with patch.dict("os.environ", {"SERPAPI_API_KEY": "secret"}), patch(
        "stage2_search.compress_for_upload", return_value=b"jpeg"
    ), patch("stage2_search.extract_exif", return_value={}), patch(
        "stage2_search.search_serpapi", return_value=hop1_results[:1]
    ), patch("stage2_search._vision_candidates", return_value=([], "")), patch(
        "stage2_search.search_bing_visual", return_value=[]
    ), patch("identity_pivot.fetch_page_html", return_value=("<title>Alice Goa</title>", "p_0300")), patch(
        "identity_pivot.extract_identity_signals",
        return_value=[{"signal_type": "title_name", "value": "Alice Goa",
                       "seed_page": "https://example.com/visual"}],
    ), patch("identity_pivot.run_hop2_search", return_value=hop2_results):
        candidates, _warnings = process_search("photo.jpg")

    urls = [candidate.url for candidate in candidates]
    assert urls[0] == "https://linkedin.com/in/alice"
    assert urls.count("https://linkedin.com/in/alice") == 1
    assert "https://github.com/alice" in urls
    assert "https://example.com/visual" in urls


def test_process_search_requires_provenance_on_final_candidates(tmp_path, monkeypatch):
    from common import provenance

    monkeypatch.setattr(provenance, "LOG_PATH", tmp_path / "requests.jsonl")
    missing = CandidateURL("https://example.com/no-provenance", source_engine="google_vision")
    with patch("stage2_search.compress_for_upload", return_value=b"jpeg"), patch(
        "stage2_search.extract_exif", return_value={}
    ), patch("stage2_search.search_serpapi", return_value=[]), patch(
        "stage2_search._vision_candidates", return_value=([missing], "")
    ), patch("stage2_search.search_bing_visual", return_value=[]):
        with pytest.raises(AssertionError, match="provenance_id"):
            process_search("photo.jpg")
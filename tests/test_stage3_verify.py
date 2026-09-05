import io
import json
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image

from stage3_verify import (
    _pdl_enrich,
    cosine_similarity,
    extract_image_urls,
    fetch_candidate_images,
    verify_image,
)
from stage2_search import CandidateURL


def test_cosine_similarity_identical_and_orthogonal():
    assert cosine_similarity([1.0, 0.0], [1.0, 0.0]) == 1.0
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == 0.0


def test_relative_og_image_and_img_urls_are_resolved():
    html = '<meta property="og:image" content="/hero.jpg"><img src="photos/a.jpg">'
    assert extract_image_urls(html, "https://example.com/page") == [
        "https://example.com/hero.jpg", "https://example.com/photos/a.jpg"
    ]


def test_face_match_uses_both_embeddings():
    image = Image.new("RGB", (80, 80), "white")
    output = io.BytesIO()
    image.save(output, format="JPEG")
    face = Mock(bbox=[0, 0, 80, 80])
    analyzer = Mock()
    analyzer.get.return_value = [face]
    with patch("stage3_verify._stage1_min_face_size", return_value=40), patch(
        "stage3_verify._arcface_embedding", return_value=[1.0, 0.0]
    ), patch("stage3_verify._pick_primary_face", side_effect=lambda faces: (faces[0], len(faces))), patch(
        "stage3_verify._try_adaface", return_value=([1.0, 0.0], "ok")
    ):
        accepted, scores, reason = verify_image(
            output.getvalue(), {"arcface_embedding": [1.0, 0.0], "adaface_embedding": [1.0, 0.0]}, analyzer
        )
    assert accepted is True
    assert scores["arcface_cosine_similarity"] == 1.0
    assert reason == "accepted"


def test_face_match_accepts_numpy_reference_embeddings():
    image = Image.new("RGB", (80, 80), "white")
    output = io.BytesIO()
    image.save(output, format="JPEG")
    face = Mock(bbox=[0, 0, 80, 80])
    analyzer = Mock()
    analyzer.get.return_value = [face]
    with patch("stage3_verify._stage1_min_face_size", return_value=40), patch(
        "stage3_verify._arcface_embedding", return_value=np.array([1.0, 0.0])
    ), patch("stage3_verify._pick_primary_face", side_effect=lambda faces: (faces[0], len(faces))), patch(
        "stage3_verify._try_adaface", return_value=(np.array([1.0, 0.0]), "ok")
    ):
        accepted, _, reason = verify_image(
            output.getvalue(),
            {"arcface_embedding": np.array([1.0, 0.0]), "adaface_embedding": np.array([1.0, 0.0])},
            analyzer,
        )
    assert accepted is True
    assert reason == "accepted"


def test_face_match_reports_embedding_unavailable_when_adaface_missing():
    image = Image.new("RGB", (80, 80), "white")
    output = io.BytesIO()
    image.save(output, format="JPEG")
    face = Mock(bbox=[0, 0, 80, 80])
    analyzer = Mock()
    analyzer.get.return_value = [face]
    with patch("stage3_verify._stage1_min_face_size", return_value=40), patch(
        "stage3_verify._arcface_embedding", return_value=np.array([1.0, 0.0])
    ), patch("stage3_verify._pick_primary_face", side_effect=lambda faces: (faces[0], len(faces))), patch(
        "stage3_verify._try_adaface", return_value=(None, "adaface_failed")
    ):
        accepted, scores, reason = verify_image(
            output.getvalue(),
            {"arcface_embedding": np.array([1.0, 0.0]), "adaface_embedding": np.array([1.0, 0.0])},
            analyzer,
        )
    assert accepted is False
    assert scores is None
    assert reason == "embedding_unavailable"


def test_fetch_candidate_images_skips_non_html_pages():
    page = Mock(status_code=200, headers={"Content-Type": "image/jpeg"})
    session = Mock()
    session.get.return_value = page
    with patch("stage3_verify.create_session", return_value=session):
        assert fetch_candidate_images("https://example.com/photo") == []


def test_fetch_candidate_images_resolves_html_images_and_caps_download():
    page = Mock(
        status_code=200,
        headers={"Content-Type": "text/html; charset=utf-8"},
        text='<meta property="og:image" content="/photo.jpg">',
    )
    image = Mock(
        status_code=200,
        headers={},
    )
    image.iter_content.return_value = [b"jpeg", b"data"]
    session = Mock()
    session.get.side_effect = [page, image]
    with patch("stage3_verify.create_session", return_value=session):
        result = fetch_candidate_images("https://example.com/profile")
    assert result == [("https://example.com/photo.jpg", b"jpegdata")]
    assert session.get.call_args_list[1].args[0] == "https://example.com/photo.jpg"


def test_pdl_missing_key_is_not_attempted():
    warnings = []
    with patch.dict("os.environ", {}, clear=True):
        result = _pdl_enrich("https://linkedin.com/in/example", warnings)
    assert result == {"attempted": False, "matched": False}
    assert warnings == ["pdl_skipped: no api key"]


def test_fetch_candidate_images_handles_404():
    response = Mock(status_code=404, headers={"Content-Type": "text/html"}, text="")
    session = Mock()
    session.get.return_value = response
    with patch("stage3_verify.create_session", return_value=session):
        assert fetch_candidate_images("https://example.com/missing") == []


def test_process_verification_retries_until_third_candidate():
    candidates = [CandidateURL(f"https://example.com/{index}") for index in range(3)]
    outcomes = [(False, {"arcface_cosine_similarity": 0.1}, "below_threshold"),
                (False, {"arcface_cosine_similarity": 0.2}, "below_threshold"),
                (True, {"arcface_cosine_similarity": 0.9, "adaface_cosine_similarity": 0.9}, "accepted")]
    with patch("stage3_verify._get_face_analyzer", return_value=Mock()), patch(
        "stage3_verify.fetch_candidate_images", side_effect=lambda url: [(url + "/image.jpg", b"image")]
    ), patch("stage3_verify.verify_image", side_effect=outcomes):
        result = __import__("stage3_verify").process_verification(
            candidates,
            {"quality_details": {}, "arcface_embedding": [1], "adaface_embedding": [1]},
        )
    assert result["match_found"] is True
    assert result["matched_page_url"] == candidates[2].url
    assert result["candidates_tried"] == 3
    assert len(result["candidates_rejected"]) == 2


def test_process_verification_no_match_handoff_schema():
    candidates = [CandidateURL("https://example.com/no-match")]
    with patch("stage3_verify._get_face_analyzer", return_value=Mock()), patch(
        "stage3_verify.fetch_candidate_images", return_value=[]
    ):
        result = __import__("stage3_verify").process_verification(
            candidates,
            {"quality_details": {}, "arcface_embedding": [1], "adaface_embedding": [1]},
        )
    assert result["match_found"] is False
    assert {"run_timestamp_utc", "source_photo_note", "candidates_tried",
            "candidates_rejected", "warnings"} <= result.keys()
    assert "matched_page_url" not in result
    assert "face_match" not in result


def test_linkedin_match_calls_pdl_once_and_handles_404():
    candidate = CandidateURL("https://linkedin.com/in/alice")
    pdl_response = Mock(status_code=404)
    session = Mock()
    session.get.return_value = pdl_response
    with patch.dict("os.environ", {"PDL_API_KEY": "secret"}), patch(
        "stage3_verify._get_face_analyzer", return_value=Mock()
    ), patch("stage3_verify.fetch_candidate_images", return_value=[("https://cdn.example/image.jpg", b"image")]), patch(
        "stage3_verify.verify_image",
        return_value=(True, {"arcface_cosine_similarity": 0.9, "adaface_cosine_similarity": 0.9}, "accepted"),
    ), patch("stage3_verify.create_session", return_value=session):
        result = __import__("stage3_verify").process_verification(
            [candidate],
            {"quality_details": {}, "arcface_embedding": [1], "adaface_embedding": [1]},
        )
    assert result["pdl_enrichment"]["attempted"] is True
    session.get.assert_called_once()


def test_non_social_match_does_not_call_pdl_and_match_handoff_round_trips():
    candidate = CandidateURL("https://example.com/post")
    with patch.dict("os.environ", {}, clear=True), patch(
        "stage3_verify._get_face_analyzer", return_value=Mock()
    ), patch("stage3_verify.fetch_candidate_images", return_value=[("https://cdn.example/image.jpg", b"image")]), patch(
        "stage3_verify.verify_image",
        return_value=(True, {"arcface_cosine_similarity": 0.9, "adaface_cosine_similarity": 0.9}, "accepted"),
    ), patch("stage3_verify._pdl_enrich") as pdl:
        result = __import__("stage3_verify").process_verification(
            [candidate],
            {"quality_details": {}, "arcface_embedding": [1], "adaface_embedding": [1]},
        )
    pdl.assert_not_called()
    loaded = json.loads(json.dumps(result, sort_keys=True))
    assert loaded["match_found"] is True
    assert {"matched_page_url", "matched_image_url", "face_match",
            "candidates_tried", "candidates_rejected", "warnings"} <= loaded.keys()

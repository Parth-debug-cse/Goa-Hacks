"""Stage 3: fetch candidate pages, verify faces, and optionally enrich PDL."""

from __future__ import annotations

import io
import logging
import os
import time
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urljoin, urlsplit

import numpy as np
from PIL import Image

from common.http_utils import create_session, response_meta
from common.netguard import UnsafeURLError, assert_public_url
from common.provenance import log_request
from stage2_search import CandidateURL
LOGGER = logging.getLogger(__name__)
ARCFACE_MATCH_THRESHOLD = 0.36
ADAFACE_MATCH_THRESHOLD = 0.30


def _stage1_min_face_size() -> int:
    from stage1_face import MIN_FACE_SIZE_PX
    return MIN_FACE_SIZE_PX


def _get_face_analyzer():
    from stage1_face import _get_face_analyzer as implementation
    return implementation()


def _pick_primary_face(faces):
    from stage1_face import _pick_primary_face as implementation
    return implementation(faces)


def _arcface_embedding(face):
    from stage1_face import _arcface_embedding as implementation
    return implementation(face)


def _try_adaface(crop, root=None):
    from stage1_face import _try_adaface as implementation
    return implementation(crop, root)


def cosine_similarity(left: list[float], right: list[float]) -> float:
    a, b = np.asarray(left, dtype=np.float32), np.asarray(right, dtype=np.float32)
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denominator) if denominator else 0.0


def _embedding_available(value: Any) -> bool:
    if value is None:
        return False
    try:
        return np.asarray(value).size > 0
    except (TypeError, ValueError):
        return False


def extract_image_urls(html: str, page_url: str) -> list[str]:
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html, "html.parser")
    urls: list[str] = []
    for selector in ('meta[property="og:image"]', 'meta[name="twitter:image"]'):
        for tag in soup.select(selector):
            content = tag.get("content")
            if content:
                urls.append(urljoin(page_url, content))
    for tag in soup.find_all("img"):
        src = tag.get("src")
        if not src:
            continue
        width, height = tag.get("width"), tag.get("height")
        if (width and width.isdigit() and int(width) < 100) or (height and height.isdigit() and int(height) < 100):
            continue
        if any(token in src.lower() for token in ("logo", "icon", "sprite", "avatar-default")):
            continue
        urls.append(urljoin(page_url, src))
    return list(dict.fromkeys(urls))


def fetch_candidate_images(page_url: str, limit: int = 3) -> list[tuple[str, bytes]]:
    try:
        assert_public_url(page_url)
    except UnsafeURLError as error:
        LOGGER.warning("Skipping unsafe candidate page URL %s: %s", page_url, error)
        return []
    try:
        session = create_session()
        started = time.monotonic()
        page = session.get(page_url, timeout=10)
        status, size = response_meta(page)
        log_request("page_fetch", "GET", page_url, {},
                    status, (time.monotonic() - started) * 1000, size)
        content_type = page.headers.get("Content-Type", "")
        if (status is not None and status >= 400) or not content_type.lower().startswith("text/html"):
            LOGGER.info("Skipping %s: status/content type %s/%s", page_url, status, content_type)
            return []
        urls = extract_image_urls(page.text, page_url)[:limit]
        images: list[tuple[str, bytes]] = []
        for image_url in urls:
            try:
                assert_public_url(image_url)
            except UnsafeURLError as error:
                LOGGER.warning("Skipping unsafe candidate image URL %s: %s", image_url, error)
                continue
            started = time.monotonic()
            response = session.get(image_url, timeout=10, stream=True)
            status, size = response_meta(response)
            log_request("image_fetch", "GET", image_url, {},
                        status, (time.monotonic() - started) * 1000, size)
            if status is not None and status >= 400:
                continue
            length = response.headers.get("Content-Length")
            try:
                if length and int(length) > 15 * 1024 * 1024:
                    continue
            except (TypeError, ValueError):
                LOGGER.info("Skipping %s: invalid Content-Length %r", image_url, length)
                continue
            data = bytearray()
            for chunk in response.iter_content(chunk_size=64 * 1024):
                if not chunk:
                    continue
                data.extend(chunk)
                if len(data) > 15 * 1024 * 1024:
                    data.clear()
                    break
            if data:
                images.append((image_url, bytes(data)))
        return images
    except Exception as error:
        LOGGER.warning("Page fetch failed for %s: %s", page_url, error)
        return []


def verify_image(image_bytes: bytes, reference: dict[str, Any], analyzer: Any) -> tuple[bool, dict[str, float] | None, str]:
    try:
        image = np.asarray(Image.open(io.BytesIO(image_bytes)).convert("RGB"))[:, :, ::-1]
        faces = analyzer.get(image)
        if not faces:
            return False, None, "no_face_found"
        face, _ = _pick_primary_face(faces)
        bbox = getattr(face, "bbox", None)
        if bbox is None or (bbox[2] - bbox[0]) < _stage1_min_face_size() or (bbox[3] - bbox[1]) < _stage1_min_face_size():
            return False, None, "no_face_found"
        arc = _arcface_embedding(face)
        x1, y1, x2, y2 = [int(value) for value in bbox]
        crop = image[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
        ada, _ = _try_adaface(crop, None)
        reference_arc = reference.get("arcface_embedding")
        reference_ada = reference.get("adaface_embedding")
        if not _embedding_available(reference_arc) or not _embedding_available(reference_ada):
            return False, None, "embedding_unavailable"
        if not _embedding_available(arc) or not _embedding_available(ada):
            return False, None, "embedding_unavailable"
        scores = {"arcface_cosine_similarity": cosine_similarity(reference_arc, arc),
                  "adaface_cosine_similarity": cosine_similarity(reference_ada, ada)}
        accepted = scores["arcface_cosine_similarity"] >= ARCFACE_MATCH_THRESHOLD and scores["adaface_cosine_similarity"] >= ADAFACE_MATCH_THRESHOLD
        return accepted, scores, "accepted" if accepted else "below_threshold"
    except Exception as error:
        LOGGER.warning("Candidate face verification failed: %s", error)
        return False, None, "verification_error"


def _pdl_enrich(url: str, warnings: list[str]) -> dict[str, Any]:
    key = os.environ.get("PDL_API_KEY")
    if not key:
        warnings.append("pdl_skipped: no api key")
        return {"attempted": False, "matched": False}
    result: dict[str, Any] = {"attempted": True, "matched": False}
    params = {"profile": url, "min_likelihood": 4}
    started = time.monotonic()
    try:
        response = create_session().get(
            "https://api.peopledatalabs.com/v5/person/enrich",
            headers={"X-Api-Key": key},
            params=params,
            timeout=10,
        )
        status, size = response_meta(response)
        log_request("pdl", "GET", "https://api.peopledatalabs.com/v5/person/enrich",
                    {**params, "X-Api-Key": key},
                    status, (time.monotonic() - started) * 1000, size)
        if status is not None and status == 404:
            return result
        if status is not None and status >= 400:
            warnings.append(f"pdl_http_error: {status}")
            return result
        payload = response.json()
        if not isinstance(payload, dict):
            warnings.append("pdl_invalid_response")
            return result
        data = payload.get("data") or {}
        if not isinstance(data, dict):
            return result
        result.update({"matched": bool(data), "likelihood": payload.get("likelihood"),
                       "full_name": data.get("full_name"), "linkedin_url": data.get("linkedin_url"),
                       "job_title": data.get("job_title"), "job_company_name": data.get("job_company_name"),
                       "location_name": data.get("location_name"), "raw_pdl_data": data})
        return result
    except Exception as error:
        log_request("pdl", "GET", "https://api.peopledatalabs.com/v5/person/enrich",
                    {**params, "X-Api-Key": key},
                    None, (time.monotonic() - started) * 1000, None, str(error))
        warnings.append(f"pdl_failed: {error}")
        return result


def process_verification(candidates: list[CandidateURL], reference: dict[str, Any]) -> dict[str, Any]:
    warnings: list[str] = []
    rejected: list[dict[str, Any]] = []
    if not candidates:
        return {
            "run_timestamp_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
            "match_found": False,
            "source_photo_note": reference.get("processing_note") or reference.get("quality_details"),
            "candidates_tried": 0,
            "candidates_rejected": [],
            "warnings": ["no_candidates_to_verify"],
        }
    try:
        analyzer = _get_face_analyzer()
    except Exception as error:
        warnings.append(f"face_verification_unavailable: {error}")
        return {
            "run_timestamp_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
            "match_found": False, "source_photo_note": reference.get("processing_note") or reference.get("quality_details"),
            "candidates_tried": 0, "candidates_rejected": [], "warnings": warnings,
        }
    for index, candidate in enumerate(candidates, start=1):
        try:
            assert_public_url(candidate.url)
        except UnsafeURLError as error:
            LOGGER.warning("Rejecting unsafe candidate %s: %s", candidate.url, error)
            rejected.append({"url": candidate.url, "reason": "unsafe_url"})
            continue
        for image_url, image_bytes in fetch_candidate_images(candidate.url):
            accepted, scores, reason = verify_image(image_bytes, reference, analyzer)
            if accepted and scores:
                host = urlsplit(candidate.url).netloc.lower()
                enrichment = None
                if (
                    (host == "linkedin.com" or host.endswith(".linkedin.com"))
                    and "/in/" in urlsplit(candidate.url).path
                ) or host == "x.com" or host.endswith(".x.com") or host == "twitter.com" or host.endswith(".twitter.com"):
                    enrichment = _pdl_enrich(candidate.url, warnings)
                result = {
                    "run_timestamp_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
                    "match_found": True, "source_photo_note": reference.get("processing_note") or reference.get("quality_details"),
                    "matched_page_url": candidate.url, "matched_image_url": image_url,
                    "source_engine": candidate.source_engine, "face_match": {
                        **scores, "arcface_threshold_used": ARCFACE_MATCH_THRESHOLD,
                        "adaface_threshold_used": ADAFACE_MATCH_THRESHOLD, "decision_rule": "and_ensemble",
                    }, "candidates_tried": index, "candidates_rejected": rejected, "warnings": warnings,
                }
                if enrichment is not None:
                    result["pdl_enrichment"] = enrichment
                return result
            rejected_item: dict[str, Any] = {"url": candidate.url, "reason": reason}
            if scores:
                rejected_item.update(scores)
            rejected.append(rejected_item)
        if not any(item.get("url") == candidate.url for item in rejected):
            rejected.append({"url": candidate.url, "reason": "no_matching_image"})
    return {
        "run_timestamp_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "match_found": False, "source_photo_note": reference.get("processing_note") or reference.get("quality_details"),
        "candidates_tried": len(candidates), "candidates_rejected": rejected, "warnings": warnings,
    }

"""Google Vision reverse-image lookup helpers."""

from __future__ import annotations

from typing import Any

try:
    from google.cloud import vision
except ImportError:
    vision = None  # type: ignore[assignment]

from src.search_serpapi import reverse_image_search

SOCIAL_DOMAINS = [
    "instagram.com",
    "x.com",
    "twitter.com",
    "facebook.com",
    "linkedin.com",
    "youtube.com",
    "tiktok.com",
]


def _is_social_url(url: str) -> bool:
    """Check whether a URL belongs to a known social media domain.

    Args:
        url (str): URL to classify.

    Returns:
        bool: True when URL contains one of the configured social domains.

    Raises:
        None.
    """
    lowered = url.lower()
    return any(domain in lowered for domain in SOCIAL_DOMAINS)


def find_matching_posts(image_path: str) -> list[dict[str, Any]]:
    """Query Google Vision web detection and return normalized page matches.

    Args:
        image_path (str): Local path to the image to search.

    Returns:
        list[dict[str, Any]]: Sorted results shaped as {url, score, is_social}.

    Raises:
        ImportError: If google-cloud-vision is not available.
        FileNotFoundError: If the image path does not exist.
        RuntimeError: If Google Vision response contains an API error message.
    """
    if vision is None:
        raise ImportError("google-cloud-vision is required for find_matching_posts")

    client = vision.ImageAnnotatorClient()
    with open(image_path, "rb") as image_file:
        content = image_file.read()

    image = vision.Image(content=content)
    response = client.web_detection(image=image)

    if getattr(response, "error", None) and getattr(response.error, "message", ""):
        raise RuntimeError(f"Google Vision API error: {response.error.message}")

    web_detection = getattr(response, "web_detection", None)
    pages = getattr(web_detection, "pages_with_matching_images", []) if web_detection else []

    results: list[dict[str, Any]] = []
    for page in pages:
        url = getattr(page, "url", "")
        if not url:
            continue
        score = float(getattr(page, "score", 0.0) or 0.0)
        results.append({"url": url, "score": score, "is_social": _is_social_url(url)})

    results.sort(key=lambda item: (not item["is_social"], -item["score"]))
    return results


def best_social_match(image_path: str, min_results: int = 1) -> dict[str, Any]:
    """Return the top-ranked result using Vision and SerpApi fallback when needed.

    Args:
        image_path (str): Local path to the image used for reverse-image search.
        min_results (int, optional): Minimum desired number of results. Defaults to 1.

    Returns:
        dict[str, Any]: Top result object with keys {url, score, is_social}.

    Raises:
        ValueError: If both Vision and fallback return no usable matches.
        RuntimeError: If Google Vision call fails.
    """
    results = find_matching_posts(image_path)

    if len(results) < min_results:
        try:
            # TODO(human): verify SerpApi upload-vs-URL support and wire a hosted image URL here.
            fallback_results = reverse_image_search(image_path)
            results.extend(fallback_results)
        except NotImplementedError as error:
            print(f"SerpApi fallback unavailable: {error}")

    results.sort(key=lambda item: (not item["is_social"], -item["score"]))

    if not results:
        raise ValueError(
            "No matching posts found. Try a photo you know is already public online."
        )

    return results[0]

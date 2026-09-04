"""SerpApi reverse-image fallback search."""

from __future__ import annotations

import os

import requests


def reverse_image_search(image_path: str, image_url: str | None = None) -> list[dict]:
    """Search SerpApi reverse-image results and normalize output shape.

    Args:
        image_path (str): Local image path (reserved for future hosted-upload flow).
        image_url (str | None): Publicly reachable image URL for SerpApi lookup.

    Returns:
        list[dict]: List of result dictionaries with keys {url, score, is_social}.

    Raises:
        NotImplementedError: If image_url is not provided.
        KeyError: If SERPAPI_KEY is missing from environment.
        RuntimeError: If the SerpApi request fails.
    """
    if image_url is None:
        # TODO(human): verify SerpApi upload-vs-URL support against current docs before enabling local file path usage.
        raise NotImplementedError(
            "SerpApi fallback requires a publicly hosted image_url; local-file upload is not "
            "confirmed supported — TODO(human): verify against current SerpApi docs before "
            "using this path."
        )

    _ = image_path
    api_key = os.environ["SERPAPI_KEY"]

    try:
        response = requests.get(
            "https://serpapi.com/search",
            params={
                "engine": "google_reverse_image",
                "image_url": image_url,
                "api_key": api_key,
            },
            timeout=30,
        )
        response.raise_for_status()
    except requests.RequestException as error:
        raise RuntimeError(f"SerpApi reverse image search failed: {error}") from error

    payload = response.json()
    image_results = payload.get("image_results", [])

    from src.web_search import SOCIAL_DOMAINS

    results: list[dict] = []
    for item in image_results:
        url = item.get("link")
        if not isinstance(url, str) or not url:
            continue
        is_social = any(domain in url.lower() for domain in SOCIAL_DOMAINS)
        results.append({"url": url, "score": 0.5, "is_social": is_social})

    results.sort(key=lambda value: (not value["is_social"], -value["score"]))
    return results

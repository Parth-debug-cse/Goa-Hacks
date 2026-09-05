# Face -> Web Match -> Blockchain
## Engineering Handoff: Stage 1-3 Pre-Anchor Pipeline

**Repository:** `Parth-debug-cse/Goa-Hacks`
**Scope delivered here:** Stage 1 contract integration, Stage 2 web/social search, Stage 3 verification through PDL enrichment
**Scope not delivered here:** Polygon Amoy anchoring, IPFS/Pinata, post-anchor deduplication, final blockchain JSON writer
**Runtime target:** Python 3.11 on macOS, using `uv` and a project-local `.venv`

## 1. Executive status

The repository now contains a working, consent-gated pre-anchor pipeline:

```text
consented photo
  -> Stage 1 quality gate + ArcFace + AdaFace
  -> Stage 2 SerpApi + Google Vision + optional Bing visual/text search
  -> URL normalization, filtering, ranking, and deduplication
  -> Stage 3 page/image fetching
  -> candidate face detection and dual-embedding verification
  -> optional LinkedIn/X/Twitter PDL enrichment
  -> deterministic handoff JSON for the blockchain teammate
```

The local mocked test suite currently passes with **25 tests**. Tests block unmocked `requests` network calls.

The following are intentionally not claimed as complete until run with team credentials and consenting control photos:

- live SerpApi, Google Vision, Bing, and PDL checkpoint runs;
- real positive/negative threshold calibration;
- complete end-to-end latency measurement with the team’s production photo;
- Polygon Amoy, IPFS/Pinata, post-anchor deduplication, and final on-chain output.

## 2. Architecture mapping

### Stage 1: Face identification

| Diagram box | Implementation | Status |
|---|---|---|
| Input photo | `stage1_face.process_image(image_path, adaface_root_cli)` | Existing and integrated |
| SCRFD detector | InsightFace `FaceAnalysis(name="buffalo_l")` | Implemented |
| Largest/primary face | `stage1_face._pick_primary_face()` | Implemented; returns `(primary, total_faces)` |
| Quality gate | `stage1_face._quality_gate()` | Implemented |
| Blur check | Laplacian variance, default minimum `50.0` | Implemented |
| Pose check | Roll, yaw, pitch from landmarks | Implemented |
| Occlusion check | Landmark bounds and crop texture standard deviation | Implemented |
| ArcFace | `buffalo_l/w600k_r50.onnx`, 512-dimensional output | Implemented |
| AdaFace | `ir_101`, WebFace12M R100 checkpoint, 512-dimensional output | Implemented |
| Verified face embedding | Stage 1 result dictionary passed into Stage 3 | Implemented |

The actual Stage 1 output keys confirmed from source are:

```text
source_image
quality_passed
reason                 # failure paths
quality_details
faces_detected
processing_note        # only when multiple faces are detected
arcface_embedding
adaface_embedding
adaface_status
timestamp
```

Both embedding helpers return ordinary Python lists:

```python
_arcface_embedding(...) -> embedding.tolist()
_try_adaface(...) -> list from _adaface_embedding(...) or None
```

Stage 3 still defensively accepts list-like and NumPy-array embeddings.

### Critical biometric-provider constraint

The implementation does not use GPT-4o, Gemini, Claude, or another general-purpose LLM to compare faces. Face comparison is performed only by ArcFace and AdaFace embeddings with cosine similarity.

## 3. Stage 1 details

### Model roles

`buffalo_l` is not the same model as AdaFace. It supplies:

- SCRFD face detection;
- InsightFace landmarks;
- ArcFace recognition model `w600k_r50.onnx`.

AdaFace is loaded separately from:

```text
third_party/AdaFace/pretrained/adaface_ir101_webface12m.ckpt
```

The production decision uses both embeddings:

```text
ArcFace score >= ARCFACE_MATCH_THRESHOLD
AND
AdaFace score >= ADAFACE_MATCH_THRESHOLD
```

Current starting values:

```python
ARCFACE_MATCH_THRESHOLD = 0.36
ADAFACE_MATCH_THRESHOLD = 0.30
```

These are configurable constants, not a completed calibration. They must be measured against the team’s own positive and negative controls before the judged demo.

### Stage 1 quality behavior

- No face: rejected.
- Face smaller than `MIN_FACE_SIZE_PX = 40`: rejected/ignored.
- Multiple faces: largest face is selected and `faces_detected` plus `processing_note` are recorded.
- Quality failure: embeddings remain `None`.
- AdaFace unavailable: ArcFace may still be produced, while `adaface_embedding` is `None` and `adaface_status` explains why.

## 4. Stage 2: Web/social search

### Files

| File | Responsibility |
|---|---|
| `stage2_search.py` | Provider calls, parsing, query construction, filtering, ranking, deduplication |
| `common/http_utils.py` | Shared requests session, User-Agent, retry policy, timeout |
| `common/image_utils.py` | SerpApi compression and EXIF extraction |

### Path A: SerpApi / Google Lens

`search_serpapi(image_bytes)`:

1. Requires `SERPAPI_API_KEY`.
2. Uploads local image bytes to `https://serpapi.com/image`.
3. Uses returned `image_id` immediately.
4. Calls Google Lens twice:
   - `type=exact_matches`;
   - `type=visual_matches`.
5. Reads only documented fields through `.get()`.
6. Converts results to `CandidateURL`.

The image is compressed to a maximum of 500,000 bytes before upload.

Candidate source labels:

```text
serpapi_exact
serpapi_visual
```

### Path B: Google Vision

`_vision_candidates(image_bytes)`:

1. Lazily imports `google.cloud.vision`.
2. Calls `ImageAnnotatorClient().web_detection(...)`.
3. Reads:
   - `pages_with_matching_images`;
   - `web_entities`;
   - `best_guess_labels`.
4. Treats matching pages as direct high-confidence candidates.
5. Builds a clean attribute query from the top entities and labels.

Candidate source label:

```text
google_vision
```

### Attribute and EXIF query strategy

`build_search_query(web_query, exif)` combines:

- Vision entity/label terms;
- `DateTimeOriginal`;
- camera `Make`;
- camera `Model`;
- GPS IFD values when present.

Generic terms such as `photo`, `image`, `person`, `stock`, and `unknown` are removed.

EXIF is optional. Missing or unreadable EXIF does not fail the search.

### Bing Visual Search

`search_bing_visual(image_bytes)`:

- Requires `AZURE_BING_VISUAL_SEARCH_KEY`.
- Defaults to:
  ```text
  https://api.bing.microsoft.com/v7.0/images/visualsearch
  ```
- Sends raw bytes in multipart field `image`.
- Parses `PagesIncluding` and `VisualSearch` actions.
- Uses `hostPageUrl` as the candidate page URL.

Candidate source labels:

```text
bing_pages_including
bing_visual_similar
```

If the key is absent, this branch logs/skips without taking down the pipeline.

### Bing text fallback

`search_bing_text(query)`:

- Uses the Vision/EXIF-derived query.
- Defaults to:
  ```text
  https://api.bing.microsoft.com/v7.0/images/search
  ```
- Sends `q`, `count`, and `safeSearch`.
- Parses `hostPageUrl` or `webSearchUrl`.

Candidate source label:

```text
bing_text
```

The fallback is skipped when the overall Stage 2 budget is exhausted.

### URL cleaning and ranking

There is no LLM dependency in this repository. The implemented deterministic cleaner:

- blocks known stock/e-commerce domains;
- recognizes social domains by hostname, not substring;
- removes tracking query parameters;
- lowercases scheme and host;
- removes trailing slash differences;
- ranks exact hints before visual hints;
- prioritizes social domains;
- uses engine priority and query relevance as later tie-breakers.

This is intentionally honest: the pipeline must not claim LLM ranking when no LLM client is configured.

### Deduplication

`merge_candidates()` deduplicates by `normalize_url()`.

The live `process_search()` path calls:

```python
_filter_candidates(merge_candidates(*groups), query)
```

Therefore duplicate URLs from SerpApi, Vision, and Bing collapse before Stage 3.

### Stage 2 timeout behavior

Three independent branches are submitted concurrently:

- SerpApi;
- Google Vision;
- Bing visual.

The caller stops collecting after `timeout_seconds` and performs:

```python
future.cancel()
pool.shutdown(wait=False, cancel_futures=True)
```

Important runtime limitation: a worker already inside a network request cannot be forcibly killed by `ThreadPoolExecutor`. It can continue in the background, but its result is not consumed after the deadline. Individual requests have their own timeout and retries.

## 5. Stage 3: Verification and enrichment

### Files

| File | Responsibility |
|---|---|
| `stage3_verify.py` | Page fetch, image extraction, face verification, PDL, handoff object |
| `run_pipeline.py` | Consent-gated Stage 1 -> Stage 2 -> Stage 3 orchestration |

### Page fetching

For each candidate:

1. GET the candidate page.
2. Require `Content-Type` beginning with `text/html`.
3. Skip HTTP errors.
4. Extract images in priority order:
   - `og:image`;
   - `twitter:image`;
   - filtered `<img src>`.
5. Resolve relative URLs with `urljoin`.
6. Skip obvious logos/icons/sprites/default avatars.
7. Download at most three image candidates.
8. Stream image bytes in chunks.
9. Enforce a 15 MB cap during download.

### Face verification

For each candidate image:

1. Decode image bytes with Pillow.
2. Convert to the format expected by InsightFace.
3. Reuse Stage 1’s analyzer, primary-face picker, ArcFace helper, AdaFace helper, and minimum face-size threshold.
4. Reject images without a sufficiently large face.
5. Compute both embeddings.
6. Compute cosine similarity independently.
7. Require both thresholds to pass.
8. Stop on the first accepted image/page.
9. Otherwise record rejection reason and continue.

Rejection reasons include:

```text
no_face_found
embedding_unavailable
below_threshold
verification_error
no_matching_image
```

No match is fabricated when all candidates fail.

### PDL enrichment

PDL is attempted only after face verification accepts a candidate and the host is:

- `linkedin.com` or a subdomain with `/in/` in the path;
- `x.com` or a subdomain;
- `twitter.com` or a subdomain.

Endpoint:

```text
https://api.peopledatalabs.com/v5/person/enrich
```

Inputs:

- `X-Api-Key`;
- `profile`;
- `min_likelihood=4`.

Behavior:

- Missing key: `attempted: false`.
- HTTP 404: no enrichment, no pipeline failure.
- Other HTTP error: warning, no pipeline failure.
- One call maximum per accepted match.
- Only selected display fields are copied; complete returned `data` is preserved in `raw_pdl_data`.

## 6. Handoff JSON contract

The blockchain teammate receives exactly one object per run.

### Match case

```json
{
  "run_timestamp_utc": "2026-09-05T12:34:56Z",
  "match_found": true,
  "source_photo_note": {},
  "matched_page_url": "https://www.linkedin.com/in/example",
  "matched_image_url": "https://media.example/photo.jpg",
  "source_engine": "serpapi_exact",
  "face_match": {
    "arcface_cosine_similarity": 0.41,
    "adaface_cosine_similarity": 0.35,
    "arcface_threshold_used": 0.36,
    "adaface_threshold_used": 0.3,
    "decision_rule": "and_ensemble"
  },
  "candidates_tried": 4,
  "candidates_rejected": [],
  "warnings": [],
  "pdl_enrichment": {
    "attempted": true,
    "matched": true,
    "likelihood": 7,
    "full_name": "Example Person",
    "linkedin_url": "https://linkedin.com/in/example",
    "job_title": "Example title",
    "job_company_name": "Example company",
    "location_name": "Example location",
    "raw_pdl_data": {}
  }
}
```

`pdl_enrichment` is included only when the accepted URL is an eligible LinkedIn/X/Twitter profile.

### No-match case

```json
{
  "run_timestamp_utc": "2026-09-05T12:34:56Z",
  "match_found": false,
  "source_photo_note": {},
  "candidates_tried": 4,
  "candidates_rejected": [],
  "warnings": []
}
```

No-match output does not include:

```text
matched_page_url
matched_image_url
face_match
pdl_enrichment
```

The JSON is serialized with sorted keys by `run_pipeline.py`, making the handoff deterministic for downstream hashing.

## 7. CLI and exit codes

Run from the repository root:

```zsh
source .venv/bin/activate

python run_pipeline.py \
  captured/consenting/reference.jpeg \
  --adaface-root ./third_party/AdaFace \
  --consent-confirmed
```

Exit codes:

| Code | Meaning |
|---:|---|
| `0` | Verified match found |
| `1` | Stage 1 quality or embedding failure |
| `2` | Search/verification completed with no match |

The `--consent-confirmed` flag is required to make the intended scope explicit. The pipeline is not designed for arbitrary third-party face-database processing.

## 8. macOS setup

This project uses `uv`; Homebrew is not required.

```zsh
cd /Users/parthsrivastava/Developer/Goa-Hacks

uv python install 3.11
uv venv --python 3.11 .venv
source .venv/bin/activate

uv pip install --python .venv/bin/python \
  -r requirements-dev.txt
```

Required local model files:

```text
third_party/AdaFace/net.py
third_party/AdaFace/face_alignment/mtcnn.py
third_party/AdaFace/pretrained/adaface_ir101_webface12m.ckpt
```

InsightFace automatically downloads:

```text
$HOME/.insightface/models/buffalo_l
```

The first Stage 1 execution performs this download. Later runs reuse the cached directory.

## 9. Environment variables

Create a local ignored file such as `.env.local`:

```zsh
export GOOGLE_APPLICATION_CREDENTIALS="$HOME/.config/face-chain-verify/google-vision.json"
export SERPAPI_API_KEY="..."
export AZURE_BING_VISUAL_SEARCH_KEY="..."
export AZURE_BING_VISUAL_SEARCH_ENDPOINT="https://api.bing.microsoft.com/v7.0/images/visualsearch"
export AZURE_BING_IMAGE_SEARCH_ENDPOINT="https://api.bing.microsoft.com/v7.0/images/search"
export PDL_API_KEY="..."
```

Load it:

```zsh
source .env.local
```

Never commit:

- API keys;
- Google service-account JSON;
- consented photos;
- embeddings;
- downloaded model weights;
- raw provider responses.

## 10. Test coverage

Run:

```zsh
python -m pytest -q
```

Current coverage includes:

- SerpApi exact/visual parsing;
- SerpApi provider failure;
- Bing missing-key skip;
- Bing malformed response;
- Bing text fallback parsing;
- Google Vision page/entity parsing;
- EXIF query inclusion;
- URL normalization and deduplication;
- timeout return behavior with a slow branch;
- query relevance ranking;
- hostname lookalike filtering;
- 500 KB compression;
- non-HTML page skip;
- HTTP 404 handling;
- relative image URL resolution;
- streamed download behavior;
- ArcFace/AdaFace similarity acceptance;
- NumPy embedding inputs;
- missing AdaFace output;
- three-candidate reject/retry flow;
- non-social PDL exclusion;
- LinkedIn PDL call and 404 handling;
- match handoff JSON round-trip;
- no-match handoff JSON shape;
- test-wide blocking of unmocked requests network calls.

## 11. Live verification checklist

These must be run by a teammate with credentials and consenting photos.

### CP1: SerpApi

Run the pipeline with `SERPAPI_API_KEY` and confirm at least one sensible candidate for a known indexed team photo.

### CP2: Vision and Bing

Confirm Vision returns pages/entities. Confirm Bing either returns candidates or logs a clean skip.

### CP3: Merge/dedup

Inspect candidates and confirm duplicate normalized URLs collapse and social/exact priorities are sensible.

### CP4: HTTP fetch

Confirm at least one reachable candidate exposes a usable image. Confirm deliberately broken URLs do not crash the run.

### CP5: Threshold calibration

Use 2-3 same-person and 2-3 different-person consenting photos. Record both similarity scores for each pair. Set thresholds with margin; do not rely on defaults without measurement.

### CP6: Reject/retry

Confirm two rejected candidates are followed by acceptance of a third controlled candidate.

### CP7: PDL

Confirm exactly one call for an accepted LinkedIn/X/Twitter profile, no call for non-eligible domains, and graceful 404 handling.

### CP8: Handoff

Validate both match and no-match JSON objects with `json.loads(json.dumps(...))` and inspect required/omitted keys.

### CP9: Offline test isolation

Run the full pytest suite with no real network access. The autouse test fixture should reject accidental `requests` calls.

### CP10: Demo timing

```zsh
time python run_pipeline.py \
  captured/consenting/reference.jpeg \
  --adaface-root ./third_party/AdaFace \
  --consent-confirmed
```

Record provider warnings, candidates tried, match result, and total runtime.

## 12. Remaining integration work

The downstream teammate still needs to implement or connect:

1. Canonical handoff file/API boundary.
2. Deterministic content hashing of the sorted JSON bytes.
3. IPFS/Pinata upload and CID capture.
4. Polygon Amoy transaction submission.
5. Post-anchor deduplication.
6. Final on-chain JSON output and explorer link.

The pre-anchor payload from `run_pipeline.py` is the boundary between this work and that module.

## 13. Known operational caveats

- Thread cancellation cannot kill a provider request already executing in a worker. The caller returns after the Stage 2 budget, but an in-flight worker can finish later.
- Google Vision requires ADC through `GOOGLE_APPLICATION_CREDENTIALS`.
- Bing provisioning may be unavailable depending on the Azure account/resource type.
- Thresholds are not production-calibrated until CP5 is completed.
- The deterministic ranking path is not an LLM ranking path and must be described accurately to reviewers.
- Search results and profile pages may be inaccessible due to robots, login walls, rate limits, or provider-specific policies.
- The system must remain limited to owned/consented photos and must not be converted into a batch arbitrary-person face database.

## 14. Safe team handoff summary

At handoff, provide the next teammate:

- repository checkout;
- this `HANDOFF.md`;
- the exact Python/`uv` setup;
- environment variable names, never secret values;
- the Stage 1 output contract;
- the Stage 2 candidate model;
- the Stage 3 handoff JSON contract;
- current threshold values and CP5 calibration results once available;
- the list of live checkpoints completed and their evidence.

Do not hand off secrets, raw embeddings, private photos, or unredacted provider payloads in chat or Git.

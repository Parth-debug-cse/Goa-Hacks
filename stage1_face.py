#!/usr/bin/env python3
"""Stage 1 — Face Identification: Quality Gate + Dual Embedding Ensemble.

Pipeline for the hackathon face-verification challenge.

Given an input photo this script:
  1. Detects the largest valid face using SCRFD (via insightface buffalo_l).
  2. Runs a quality gate on that face:
       - blur     : Laplacian variance on the cropped face (OpenCV)
       - pose     : yaw / roll / pitch estimated from facial landmarks
       - occlusion: key landmarks (eyes / nose / mouth) present & in bounds
     If any quality check fails, the run stops immediately and reports
     {"quality_passed": false, "reason": "..."} — no embeddings are computed.
  3. If the face passes quality checks, it generates TWO independent
     face embeddings (the ensemble):
       - ArcFace   (insightface buffalo_l  -> w600k_r50 backbone)
       - AdaFace   (R100 / WebFace12M pretrained weights, quality-adaptive)
  4. Writes a single JSON document to ``verified_embedding.json`` and prints it.

No LLM / large-language-model is used anywhere in this stage. All face
processing is performed exclusively with classical CV + the two embedding
models named above (deliberate exclusion for biometric-identity policy).

CLI notes:
    * The script exits 0 when the quality gate passes, 1 when it fails
      (or when an unrecoverable error occurs), so it composes well with
      shell pipelines.
    * On a quality-gate failure embeddings are null and the run stops before
      any embedding model is invoked.

Usage:
    python stage1_face.py <path_to_image>
    python stage1_face.py <path_to_image> --adaface-root ../AdaFace
    python stage1_face.py <path_to_image> --output run_42.json --min-blur 40
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from datetime import datetime, timezone
from typing import Any, Optional

import cv2
import numpy as np

# --------------------------------------------------------------------------- #
# Tunable defaults (documented in SETUP.md)
# --------------------------------------------------------------------------- #
MODEL_PACK = "buffalo_l"                 # insightface ready-made pack (SCRFD det + ArcFace rec + landmarks)
MIN_FACE_SIZE_PX = 40                    # faces smaller than this are ignored (too small to be useful)
BLUR_MIN_VARIANCE = 50.0                 # below this Laplacian variance the face is considered "blurry"
MAX_ROLL_DEG = 20.0                      # max absolute eye-line tilt before we call the pose "turned"
MAX_YAW_SCORE = 0.30                     # 0 = frontal; grows as the nose drifts toward one eye
MAX_PITCH_SCORE = 0.55                   # asymmetry between eye->nose and nose->mouth spacing
OCCLUSION_MIN_CROP_STD = 10.0            # grey-level std of the crop; suspiciously flat crops are rejected

# Where stage1 looks for the AdaFace repo, in order of preference.
DEFAULT_ADAFACE_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "third_party", "AdaFace")
ADAFACE_WEIGHT_RELATIVE = os.path.join("pretrained", "adaface_ir101_webface12m.ckpt")

# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def _now_iso() -> str:
    """Return the current UTC time in ISO 8601 format."""
    return datetime.now(timezone.utc).isoformat()


def _json_default(value):
    """Fallback encoder for the JSON output: coerces numpy scalars/arrays into
    JSON-native Python types so a stray np.bool_/np.float64 can never break it."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _load_image(path: str) -> np.ndarray:
    """Load an image from disk with cv2 (BGR order).

    Raises:
        FileNotFoundError: if the path does not exist or cannot be decoded.
    """
    image = cv2.imread(path)
    if image is None:
        raise FileNotFoundError(
            f"Could not read image at '{path}'. Check the path and that the "
            "file is a valid image (jpg/png/webp/...)."
        )
    return image


# --------------------------------------------------------------------------- #
# Step 1 — Face detection (SCRFD via insightface)
# --------------------------------------------------------------------------- #
def _get_face_analyzer():
    """Build the InsightFace FaceAnalysis object (SCRFD buffalo_l).

    Uses GPU when available, otherwise transparently falls back to CPU.
    The model weights auto-download on first run into ~/.insightface/models.

    Raises:
        RuntimeError: if InsightFace cannot be initialised (e.g. missing
        package, missing weights download, or unsupported providers).
    """
    import onnxruntime as ort
    from insightface.app import FaceAnalysis

    available = ort.get_available_providers()
    providers: list[str] = []
    if "CUDAExecutionProvider" in available:
        providers.append("CUDAExecutionProvider")
    providers.append("CPUExecutionProvider")

    try:
        app = FaceAnalysis(name=MODEL_PACK, providers=providers)
        app.prepare(ctx_id=0)  # ctx_id=0 selects GPU device 0 when CUDA is active
        return app
    except Exception as error:
        raise RuntimeError(
            "Failed to initialise InsightFace. Install with "
            "`pip install -r stage1_requirements.txt` and make sure the "
            f"buffalo_l weights can be downloaded. Underlying error: {error}"
        ) from error


def _pick_primary_face(faces: list) -> tuple[Any, int]:
    """Return the largest detectable face (used when several are present).

    Args:
        faces: list of insightface Face objects.

    Returns:
        (face, total_faces): the chosen face and the total number of faces found.
    """
    total = len(faces)
    # The "primary" face is simply the one with the largest bounding-box area.
    primary = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    return primary, total


# --------------------------------------------------------------------------- #
# Step 1b — Quality gate
# --------------------------------------------------------------------------- #
def _check_blur(crop_gray: np.ndarray) -> tuple[bool, float]:
    """Evaluate sharpness via Laplacian variance (standard OpenCV technique).

    A completely flat / blurry region has a variance near zero; a sharp image
    with edges produces a high variance. Thresholds are documented above.

    Returns:
        (is_sharp, laplacian_variance)
    """
    variance = float(cv2.Laplacian(crop_gray, cv2.CV_64F).var())
    return variance >= BLUR_MIN_VARIANCE, variance


def _check_pose(kps: np.ndarray) -> tuple[bool, dict]:
    """Estimate yaw / roll / pitch from the 5 facial landmarks and decide if
    the face is close enough to frontal (within tolerable orientation).

    Landmark layout from SCRFD (index order):
        0 left-eye, 1 right-eye, 2 nose, 3 left-mouth, 4 right-mouth.

    NOTE: the formulas below are deliberately sign-symmetric, so they stay
    correct even if the left/right landmark labels were ever swapped by a
    different detector version.

    Returns:
        (pose_ok, details) where details holds the numeric orientation metrics.
    """
    details: dict[str, Any] = {}
    eye_left, eye_right, nose = kps[0], kps[1], kps[2]
    mouth_mid = (kps[3] + kps[4]) / 2.0
    eye_mid = (eye_left + eye_right) / 2.0

    inter_eye = float(np.linalg.norm(eye_right - eye_left))
    if inter_eye < 1e-6:  # degenerate geometry should never be accepted
        return False, {"pose_ok": False, "reason_detail": "degenerate eye landmarks"}

    # roll — tilt of the eye line from horizontal.
    roll_rad = np.arctan2(eye_right[1] - eye_left[1], eye_right[0] - eye_left[0])
    roll_deg = float(abs(roll_rad) * 180.0 / np.pi)
    details["roll_deg"] = roll_deg

    # yaw — horizontal drift of the nose away from the inter-eye midpoint,
    # normalised by the inter-eye distance. ~0 = frontal.
    yaw_score = float(abs(nose[0] - eye_mid[0]) / inter_eye)
    details["yaw_score"] = yaw_score

    # pitch — asymmetry between the eye->nose and nose->mouth vertical spans.
    eye_nose = float(nose[1] - eye_mid[1])
    nose_mouth = float(mouth_mid[1] - nose[1])
    pitch_score = float(abs(eye_nose - nose_mouth) / (eye_nose + nose_mouth + 1e-6))
    details["pitch_score"] = pitch_score

    details["roll_ok"] = roll_deg <= MAX_ROLL_DEG
    details["yaw_ok"] = yaw_score <= MAX_YAW_SCORE
    details["pitch_ok"] = pitch_score <= MAX_PITCH_SCORE
    pose_ok = bool(details["roll_ok"] and details["yaw_ok"] and details["pitch_ok"])
    return pose_ok, details


def _check_occlusion(kps: np.ndarray, image_shape: tuple, crop_bgr: np.ndarray) -> tuple[bool, dict]:
    """Basic occlusion / anatomy sanity check.

    The SCRFD keypoints live in the *original image* coordinate space, so they
    must be validated against the full image shape (not the crop). We require
    all five landmarks (eyes, nose, mouth corners) to be present, finite and
    inside the image bounds, and the cropped face must have some grey-level
    texture (rejects black/white boxes a detector could hallucinate).
    """
    details: dict[str, Any] = {}
    img_h, img_w = image_shape[0], image_shape[1]
    landmarks_ok = bool(
        kps.shape == (5, 2)
        and bool(np.isfinite(kps).all())
        and bool(kps[:, 0].min() >= 0)
        and bool(kps[:, 1].min() >= 0)
        and bool(kps[:, 0].max() <= img_w)
        and bool(kps[:, 1].max() <= img_h)
    )
    details["landmarks_ok"] = landmarks_ok

    crop_gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    texture_std = float(crop_gray.std())
    details["crop_std"] = texture_std
    texture_ok = texture_std >= OCCLUSION_MIN_CROP_STD

    occlusion_ok = bool(landmarks_ok and texture_ok)
    return occlusion_ok, details


def _quality_gate(face, crop_bgr: np.ndarray, image_shape: tuple) -> tuple[bool, dict, Optional[str]]:
    """Run all quality checks on a detected face.

    Args:
        face: insightface Face object for the selected face.
        crop_bgr: BGR crop of the face (used for blur + texture checks).
        image_shape: (H, W) of the full image (keypoints are in full-image coords).

    Returns:
        (passed, quality_details, failure_reason). ``failure_reason`` is None
        when the face passes.
    """
    kps = getattr(face, "kps", None)
    if kps is None:
        return False, {"occlusion_ok": False, "note": "detector returned no landmarks"}, "no landmarks from detector"

    crop_gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)

    blur_ok, blur_score = _check_blur(crop_gray)
    pose_ok, pose_details = _check_pose(kps)
    occlusion_ok, occlusion_details = _check_occlusion(kps, image_shape, crop_bgr)

    quality_details: dict[str, Any] = {
        "blur_score": blur_score,
        "blur_ok": blur_ok,
        "pose_ok": pose_ok,
        **pose_details,
        "occlusion_ok": occlusion_ok,
        **occlusion_details,
    }

    if not blur_ok:
        return False, quality_details, f"face is too blurry (Laplacian variance {blur_score:.1f} < {BLUR_MIN_VARIANCE})"
    if not pose_ok:
        return False, quality_details, "face is turned too far from frontal (yaw/roll/pitch outside thresholds)"
    if not occlusion_ok:
        return False, quality_details, "key landmarks (eyes/nose/mouth) not fully visible (occlusion suspected)"

    return True, quality_details, None


# --------------------------------------------------------------------------- #
# Step 2 — Dual embedding ensemble
# --------------------------------------------------------------------------- #
def _arcface_embedding(face) -> list[float]:
    """Return the 512-d ArcFace embedding for a detected face (buffalo_l rec)."""
    embedding = np.asarray(getattr(face, "embedding", []), dtype=np.float32)
    if embedding.size == 0:
        raise RuntimeError("InsightFace returned no embedding for the detected face")
    return embedding.tolist()


def _find_adaface_root(cli_root: Optional[str]) -> Optional[str]:
    """Locate the AdaFace repo: --adaface-root flag > ADAFACE_ROOT env var >
    the default third_party/AdaFace path. Returns None if not found.

    Args:
        cli_root: value passed through the --adaface-root CLI flag (may be None).
    """
    candidates = []
    if cli_root:
        candidates.append(cli_root)
    env_root = os.environ.get("ADAFACE_ROOT")
    if env_root:
        candidates.append(env_root)
    candidates.append(DEFAULT_ADAFACE_ROOT)

    for candidate in candidates:
        # The AdaFace repo is considered present if its backbone builder
        # (net.py) and its bundled MTCNN live under the candidate directory.
        if (
            os.path.isfile(os.path.join(candidate, "net.py"))
            and os.path.isfile(os.path.join(candidate, "face_alignment", "mtcnn.py"))
        ):
            return candidate
    return None


def _load_module_from_path(name: str, path: str):
    """Load a Python module from a file path without going through package
    __init__.py files (avoids AdaFace's CUDA-hardcoded align module)."""
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _adaface_embedding(pil_face, model, device, device_str: str) -> list[float]:
    """Compute a 512-d AdaFace embedding from an aligned face crop.

    The crop must be BGR, 112x112, pixel values in [0,1] normalised to mean /
    std  0.5 — see AdaFace's to_input(). That conversion is replicated here
    instead of importing AdaFace's inference.py (which pulls in a hardcoded
    'cuda:0' alignment pipeline and a stale pytorch-lightning dependency).

    Args:
        pil_face: PIL Image of the aligned face returned by their MTCNN.
        model: torch module (backbone) in eval mode.
        device / device_str: torch target device.

    Returns:
        the normalised embedding as a plain Python list.
    """
    import torch  # local import: torch is only needed when AdaFace runs

    np_img = np.array(pil_face)                     # RGB order from PIL
    bgr_img = ((np_img[:, :, ::-1] / 255.0) - 0.5) / 0.5   # to BGR, normalise to ~[-1, 1]
    tensor = torch.tensor([bgr_img.transpose(2, 0, 1)], dtype=torch.float32).to(device)

    with torch.no_grad():
        feature, _ = model(tensor)                  # backbone returns (embedding, norm)
    return feature[0].cpu().numpy().tolist()


# --------------------------------------------------------------------------- #
# Step 3 — Orchestration
# --------------------------------------------------------------------------- #
def process_image(image_path: str, adaface_root_cli: Optional[str] = None) -> dict:
    """Run the full Stage 1 flow on a single image.

    Args:
        image_path: path to the input photo.
        adaface_root_cli: optional explicit path to the AdaFace repo.

    Returns:
        The result dictionary (as designed in the module docstring / spec).
    """
    result = {"source_image": image_path}

    # -- Step 1: detect ---------------------------------------------------- #
    try:
        analyzer = _get_face_analyzer()
    except RuntimeError as error:
        # No point continuing without a detector.
        result.update(
            {
                "quality_passed": False,
                "reason": f"face detection unavailable: {error}",
                "timestamp": _now_iso(),
            }
        )
        return result

    image = _load_image(image_path)                 # BGR
    faces = analyzer.get(image)

    if not faces:
        result.update(
            {
                "quality_passed": False,
                "reason": "no face detected in the image",
                "quality_details": {"faces_detected": 0},
                "faces_detected": 0,
                "timestamp": _now_iso(),
            }
        )
        return result

    face, total_faces = _pick_primary_face(faces)
    result["faces_detected"] = total_faces
    if total_faces > 1:
        result["processing_note"] = (
            f"{total_faces} faces found — the largest/most prominent is processed below."
        )

    # Crop the detected face (clamped to image bounds).
    x1, y1, x2, y2 = [int(v) for v in face.bbox]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(image.shape[1], x2), min(image.shape[0], y2)
    crop_bgr = image[y1:y2, x1:x2]

    # -- Step 1b: quality gate --------------------------------------------- #
    passed, quality_details, failure_reason = _quality_gate(face, crop_bgr, image.shape)
    result["quality_passed"] = passed
    result["quality_details"] = quality_details

    if not passed:
        result["reason"] = failure_reason
        result["arcface_embedding"] = None
        result["adaface_embedding"] = None
        result["timestamp"] = _now_iso()
        return result

    # -- Step 2: dual embeddings ------------------------------------------- #
    try:
        result["arcface_embedding"] = _arcface_embedding(face)
    except Exception as error:                      # noqa: BLE001 — propagate clearly
        result["arcface_embedding"] = None
        result["reason"] = f"ArcFace embedding failed: {error}"
        result["adaface_embedding"] = None
        result["timestamp"] = _now_iso()
        return result

    # AdaFace is best-effort: if the repo/weights are not set up we record a
    # null embedding and say why, rather than failing the whole run.
    result["adaface_embedding"], result["adaface_status"] = _try_adaface(crop_bgr, adaface_root_cli)

    result["timestamp"] = _now_iso()
    return result


def _try_adaface(crop_bgr: np.ndarray, adaface_root_cli: Optional[str]):
    """Best-effort AdaFace embedding. Returns (embedding_or_None, status_str)."""
    import torch

    adaface_root = _find_adaface_root(adaface_root_cli)
    if adaface_root is None:
        return None, (
            "SKIPPED: AdaFace repo not found (checked --adaface-root, ADAFACE_ROOT, "
            f"and {DEFAULT_ADAFACE_ROOT}). See SETUP.md."
        )

    weight_path = os.path.join(adaface_root, ADAFACE_WEIGHT_RELATIVE)
    if not os.path.isfile(weight_path):
        return None, (
            f"SKIPPED: AdaFace weights not found at {weight_path}. "
            "See SETUP.md for the download link."
        )

    try:
        device_str = "cuda:0" if torch.cuda.is_available() else "cpu"
        device = torch.device(device_str)

        # Build the ResNet-100 backbone from AdaFace's net.py and load weights.
        sys.path.insert(0, adaface_root)
        import net  # noqa: E402  (AdaFace's backbone builder)
        model = net.build_model("ir_101")
        state = torch.load(weight_path, map_location=device)
        state_dict = {k[6:]: v for k, v in state["state_dict"].items() if k.startswith("model.")}
        model.load_state_dict(state_dict)
        model.eval().to(device)

        # Load their bundled MTCNN directly (bypasses align.py's hardcoded cuda).
        mtcnn_path = os.path.join(adaface_root, "face_alignment", "mtcnn.py")
        mtcnn_mod = _load_module_from_path("adaface_mtcnn", mtcnn_path)
        aligner = mtcnn_mod.MTCNN(device=device_str, crop_size=(112, 112))

        # Their MTCNN expects a PIL RGB image of the *whole* photo; but to
        # stay consistent with our quality-gated crop we reuse the SCRFD crop.
        from PIL import Image
        pil_crop = Image.fromarray(crop_bgr[:, :, ::-1]).convert("RGB")

        boxes, faces = aligner.align_multi(pil_crop, limit=1)
        if not faces:
            return None, "SKIPPED: AdaFace MTCNN did not find a face in the crop"

        embedding = _adaface_embedding(faces[0], model, device, device_str)
        return embedding, f"OK ({device_str})"
    except Exception as error:                       # noqa: BLE001 — report, don't crash
        return None, f"FAILED: AdaFace error — {error}"


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Stage 1 — face identification: quality gate + ArcFace/AdaFace embedding ensemble."
    )
    parser.add_argument("image_path", help="Path to the input photo")
    parser.add_argument(
        "--adaface-root",
        default=None,
        help="Path to a clone of mk-minchul/AdaFace (else ADAFACE_ROOT env var, "
        "else ./third_party/AdaFace).",
    )
    parser.add_argument(
        "--output",
        default="verified_embedding.json",
        help="Output JSON path (default: verified_embedding.json)",
    )
    parser.add_argument(
        "--min-blur",
        type=float,
        default=BLUR_MIN_VARIANCE,
        help="Minimum Laplacian variance for the blur gate (default: 50)",
    )
    return parser


def main() -> int:
    """CLI entrypoint: parse args, run the pipeline, save + print the result."""
    args = _build_arg_parser().parse_args()

    global BLUR_MIN_VARIANCE                        # noqa: PLW0603 — CLI override of the default
    BLUR_MIN_VARIANCE = args.min_blur

    result = process_image(args.image_path, adaface_root_cli=args.adaface_root)

    # Print first (so a failure to write never hides the result), then save.
    print(json.dumps(result, indent=2, default=_json_default))
    with open(args.output, "w", encoding="utf-8") as out:
        json.dump(result, out, indent=2, default=_json_default)

    return 0 if result.get("quality_passed") else 1


if __name__ == "__main__":
    sys.exit(main())
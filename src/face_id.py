"""Face detection and embedding helpers."""

from __future__ import annotations

import numpy as np
import cv2
from insightface.app import FaceAnalysis

MIN_FACE_SIZE_PX = 40


def _prepare_face_analyzer() -> FaceAnalysis:
    """Create and initialize an InsightFace analyzer.

    Returns:
        FaceAnalysis: Prepared face analyzer using GPU with CPU fallback.

    Raises:
        RuntimeError: If both GPU+CPU and CPU-only initialization fail.
    """
    app = FaceAnalysis(name="buffalo_l", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    try:
        app.prepare(ctx_id=0)
        return app
    except Exception as gpu_error:
        print(f"Warning: GPU acceleration unavailable, retrying on CPU only. Error: {gpu_error}")
        cpu_app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
        try:
            cpu_app.prepare(ctx_id=0)
            return cpu_app
        except Exception as cpu_error:
            raise RuntimeError(f"Failed to initialize InsightFace providers: {cpu_error}") from cpu_error


def identify_face(image_path: str):
    """Detect the largest valid face and return embedding, crop, and bounding box.

    Args:
        image_path (str): Path to an input image on disk.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]:
            ArcFace embedding vector, cropped face BGR image, and bounding box array.

    Raises:
        FileNotFoundError: If the image path cannot be loaded.
        RuntimeError: If InsightFace cannot be initialized.
        ValueError: If no valid face larger than minimum threshold is detected.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not read image at path: {image_path}")

    app = _prepare_face_analyzer()
    detected_faces = app.get(image)

    valid_faces = []
    for face in detected_faces:
        x1, y1, x2, y2 = [int(v) for v in face.bbox]
        width = x2 - x1
        height = y2 - y1
        if width >= MIN_FACE_SIZE_PX and height >= MIN_FACE_SIZE_PX:
            valid_faces.append(face)

    if not valid_faces:
        raise ValueError(
            f"No face found with minimum size {MIN_FACE_SIZE_PX}px in image: {image_path}"
        )

    face = max(valid_faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    x1, y1, x2, y2 = [int(v) for v in face.bbox]

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(image.shape[1], x2)
    y2 = min(image.shape[0], y2)

    cropped_face = image[y1:y2, x1:x2]
    embedding = np.asarray(face.embedding, dtype=np.float32)
    bbox = np.asarray([x1, y1, x2, y2], dtype=np.int32)

    return embedding, cropped_face, bbox

# SETUP.md — Stage 1: Face Quality Gate + Dual Embedding Ensemble

This document walks through installing every dependency needed to run
`stage1_face.py` and produce a `verified_embedding.json`.

---

## 1. What the script does (30-second overview)

```
photo → SCRFD face detection (insightface buffalo_l)
      → quality gate (blur / pose / occlusion)
      → ArcFace embedding        (buffer_l rec, 512-d)
      → AdaFace embedding        (ResNet-100 / WebFace12M, 512-d)
      → verified_embedding.json
```

No LLM is used anywhere. Face processing is classical CV + the two named
models only (deliberate exclusion for biometric-identity policy reasons).

---

## 2. Python version

* **Recommended:** Python **3.10 or 3.11** (no source-compilation of
  insightface needed).
* **Tested-on/ok:** Python **3.12** — works, but pip compiles insightface's
  Cython extensions from source on first install (requires a C compiler, e.g.
  `gcc`; on Ubuntu: `sudo apt install build-essential`).

---

## 3. Install Python dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r stage1_requirements.txt
```

If you hit a wheel/toolchain error on Python 3.12, install `build-essential`
(or `xcode-select --install` on macOS) and retry.

### GPU vs CPU

The pinned requirements use CPU-only wheels so the script runs anywhere. On a
**GPU box** (e.g. RTX 6000 Ada) for the fastest detection:

```bash
pip uninstall -y onnxruntime
pip install onnxruntime-gpu==1.18.1

# and a CUDA-enabled torch matching your CUDA version, e.g.
pip install torch==2.2.2 --index-url https://download.pytorch.org/whl/cu121
```

InsightFace (`face_id`-style provider selection in `stage1_face.py`) picks the
CUDA execution provider automatically and falls back to CPU if CUDA is not
available, so this is purely an optimisation.

---

## 4. InsightFace model weights (buffalo_l)

insightface auto-downloads the `buffalo_l` package (~350 MB) on the **first**
run of the script into `~/.insightface/models/buffalo_l`. No manual step is
needed — the download happens during Step 1 detection.

If the site it downloads from is blocked in your network, pre-seed it from a
mirror into that folder before running.

---

## 5. AdaFace (second embedding model)

AdaFace has **no pip package**, so we use the official repo directly.

### 5a. Clone the repository

```bash
mkdir -p third_party
git clone https://github.com/mk-minchul/AdaFace.git third_party/AdaFace
```

The script locates the repo by checking, in order:

1. the `--adaface-root <PATH>` CLI flag,
2. the `ADAFACE_ROOT` environment variable,
3. `./third_party/AdaFace` (the default laid out above).

Only `net.py` (backbone builder) and `face_alignment/` (their bundled MTCNN)
are used. AdaFace's own training-oriented `requirements.txt` (mxnet, menpo,
bcolz, pytorch-lightning …) is **not** installed — it is unnecessary for
inference.

### 5b. Download the pretrained weights

We use the **ResNet-100 / WebFace12M** checkpoint (`adaface_ir101_webface12m.ckpt`,
~140 MB), which yields the strongest AdaFace accuracy in the paper.

Official Google-Drive link (from the AdaFace README, R100 | WebFace12M row):

```
https://drive.google.com/file/d/1dswnavflETcnAuplZj1IOKKP0eM8ITgT/view?usp=sharing
```

Place the file exactly at:

```
third_party/AdaFace/pretrained/adaface_ir101_webface12m.ckpt
```

> Large Google Drive files are easier to pull with `gdown`:
> ```bash
> pip install gdown
> gdown 1dswnavflETcnAuplZj1IOKKP0eM8ITgT -O third_party/AdaFace/pretrained/adaface_ir101_webface12m.ckpt
> ```

If both the repo and the checkpoint are missing, the script **does not fail**:
it records `"adaface_embedding": null` with an explanatory `adaface_status`
and continues with the ArcFace embedding. (The quality gate and ArcFace path
never depend on AdaFace being present.)

---

## 6. Run the pipeline

```bash
python stage1_face.py path/to/your/photo.jpg
```

Output — terminal print **and** `verified_embedding.json` (same content):

```json
{
  "source_image": "path/to/your/photo.jpg",
  "quality_passed": true,
  "quality_details": {
    "blur_score": 218.4,
    "blur_ok": true,
    "pose_ok": true,
    "roll_deg": 2.1,
    "yaw_score": 0.04,
    "pitch_score": 0.12,
    "occlusion_ok": true
  },
  "arcface_embedding": [ ... 512 floats ... ],
  "adaface_embedding": [ ... 512 floats ... ],
  "adaface_status": "OK (cpu)",
  "timestamp": "2026-09-05T12:34:56.789+00:00"
}
```

### Useful flags

| Flag | Purpose |
|---|---|
| `--adaface-root <PATH>` | Point at a non-default AdaFace clone |
| `--output <FILE>` | Different output file name (default `verified_embedding.json`) |
| `--min-blur <FLOAT>` | Lower/raise the blur-gate threshold (default 50) |

### Exit codes (pipeline-friendly)

* `0` — quality gate passed, embeddings produced
* `1` — quality gate rejected the face, *or* an unrecoverable error occurred
  (details are always printed + written to the JSON file)

---

## 7. Quality gate thresholds (where to tune)

| Check | Metric | Default threshold |
|---|---|---|
| blur | Laplacian variance of the grey-level face crop | `>= 50.0` |
| roll | eye-line tilt from horizontal | `<= 20.0 deg` |
| yaw | nose-horiz-offset / inter-eye distance | `<= 0.30` |
| pitch | eye↔nose vs nose↔mouth spacing asymmetry | `<= 0.55` |
| occlusion | all 5 landmarks in-bounds + crop texture std | `>= 10.0` |

Understanding: blur, pose and occlusion are **heuristics**, not learned models.
They reject the obvious unusable cases (dark/blurry/turned faces) but a
"borderline" face will pass. This is intentional — the ensemble embeddings
(ArcFace + AdaFace, the latter explicitly trained for degraded faces) do the
heavy lifting on the hard cases.

---

## 8. Behaviour on edge cases

* **No face detected** → `quality_passed: false`, reason `no face detected`,
  exit 1.
* **Multiple faces** → the largest/most prominent face is processed; the JSON
  includes `faces_detected` and a `processing_note`.
* **AdaFace unavailable** → `adaface_embedding: null` + reason in
  `adaface_status`; the run still succeeds with ArcFace only.
* **Quality gate rejects** → embeddings are `null` and the script stops before
  touching either embedding model.

---

## 9. Known limitations (be ready to state these to judges)

1. **Blur/pose/occlusion checks are rule-based heuristics.** They are tuned to
   be permissive rather than over-aggressive; genuinely marginal faces may
   pass. They exist to catch the obvious bad inputs, not to be a perfect
   quality oracle.
2. **AdaFace weights come from Google Drive** — a manual download. If that
   link is unavailable/blocked the AdaFace half of the ensemble is skipped
   gracefully (recorded in `adaface_status`).
3. **InsightFace's Auto-download of buffalo_l** requires internet on first
   run; the single largest one-time cost (~350 MB).
4. **`innocent` vs `culprit` matching is NOT implemented here.** This stage
   only *produces* embeddings. Comparing them to other photos (the actual
   identity match) is a later stage and out of scope for this module.
5. AdaFace's bundled MTCNN re-detects the face inside the SCRFD crop; if it
   fails to find one, `adaface_embedding` is null even though ArcFace
   succeeded. This is surfaced in `adaface_status`, never silent.

---

## 10. Folder layout produced by this stage

```
verified_embedding.json   # canonical output (also printed to stdout)
third_party/AdaFace/      # optional clone; ignored by git
~/.insightface/models/buffalo_l/   # auto-downloaded model weights
```

Everything in `third_party/` and any model weights are `.gitignore`d — only
source and docs are committed to the repo.
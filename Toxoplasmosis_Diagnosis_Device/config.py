# =============================================================================
# PoCT - Toxoplasmosis Diagnosis Device (Dec, 2026)
# Dodit Suprianto
# Email: Dodit.suprianto@polinema.ac.id
# Affiliation:
#   - Department of Electrical Engineering, Politeknik Negeri Malang, Malang, Indonesia
#   - AIDS, Toxoplasma, Opportunistic Disease and Malaria (ATOM) Research Group,
#     Faculty of Medicine, Universitas Brawijaya, Malang, Indonesia
# =============================================================================
"""
config.py - central configuration of the diagnosis application
==============================================================
Edit the values below, or override the paths with environment variables (no code change needed):

    DOTBLOT_YOLO_WEIGHTS   YOLOv8-seg weights file          (default: model/best.pt)
    DOTBLOT_CLASSIFIER     classifier bundle (.joblib)      (default: model/dotblot_q4_mlp.joblib)
    DOTBLOT_OUTPUT_DIR     folder for all captured files    (default: output/)
    DOTBLOT_CAMERA         camera index to use              (default: try CAMERA_INDICES)
    DOTBLOT_DEBUG          1 = print timing / detection / canvas diagnostics to the console
"""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
YOLO_WEIGHTS = Path(os.environ.get("DOTBLOT_YOLO_WEIGHTS", BASE_DIR / "model" / "best.pt"))
CLASSIFIER_PATH = Path(os.environ.get("DOTBLOT_CLASSIFIER", BASE_DIR / "model" / "dotblot_q4_mlp.joblib"))
# Extra arguments of ultralytics `model.predict`, e.g. {"imgsz": 1920, "conf": 0.25}. Empty = Ultralytics defaults.
YOLO_PREDICT_KWARGS = {}

# ---------------------------------------------------------------------------
# Output folders (created on first use; excluded from git by .gitignore)
# ---------------------------------------------------------------------------
OUTPUT_DIR = Path(os.environ.get("DOTBLOT_OUTPUT_DIR", BASE_DIR / "output"))
DIAGNOSIS_DIR = OUTPUT_DIR / "diagnosis"        # Diagnosis tab: captured crops + diagnosis_log.csv
LABELLING_DIR = OUTPUT_DIR / "labelling"        # Labelling tab: {column}_{row}_{label}_{lux}.png
RAW_CAPTURE_DIR = OUTPUT_DIR / "raw_capture"    # Raw-capture tab: raw photographs for segmentation training

# ---------------------------------------------------------------------------
# Camera
# ---------------------------------------------------------------------------
_cam = os.environ.get("DOTBLOT_CAMERA")
CAMERA_INDEX = int(_cam) if _cam not in (None, "") else None   # None = try CAMERA_INDICES in order
CAMERA_INDICES = (0, 1, 2)
CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FPS = 3840, 2160, 30       # 4K request; the driver falls back if unsupported
CAMERA_BUFFER_SIZE = 1                                          # small buffer = low latency
PREVIEW_INTERVAL_MS = 66                                        # timer period, ~15 frames per second

# ---------------------------------------------------------------------------
# Spot cropping (must match the way the training crops were produced)
# ---------------------------------------------------------------------------
CROP_PADDING = 1.05          # circular crop radius = 1.05 x detected spot radius
SCALE_TOLERANCE = 0.15       # rescale the crop to the training spot size only if it deviates by more than 15 %

# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------
DEBUG = os.environ.get("DOTBLOT_DEBUG", "0") not in ("", "0", "false", "False")

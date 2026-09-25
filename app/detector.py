"""YOLO caries detector.

Wraps an Ultralytics YOLO model behind a small, stable interface so the model
can be swapped (custom caries weights vs. generic fallback) without touching the
API layer. Also handles OpenCV preprocessing (contrast enhancement) that helps
with the low-contrast, small-lesion challenge described in the CariesYOLO paper.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .config import MODELS_DIR, settings

logger = logging.getLogger("cariospectra.detector")


class CariesDetector:
    """Loads one YOLO model and runs caries detection on an image."""

    def __init__(self, model_ref: str, name: str, model_type: str) -> None:
        # Imported lazily so the module imports even before ultralytics/torch
        # are installed (useful for linting / partial environments).
        from ultralytics import YOLO

        logger.info("Loading model '%s' (%s) from %s", name, model_type, model_ref)
        self.name = name
        self.model_type = model_type
        self.model = YOLO(model_ref)
        self.model_name = Path(model_ref).name

        # Class id -> name map from the loaded model.
        self.names: Dict[int, str] = dict(self.model.names)
        self.caries_ids: set[int] = self._compute_caries_ids()

    def _compute_caries_ids(self) -> set[int]:
        """Decide which class ids represent caries.

        - Single-class model -> that class is caries.
        - Multi-class model  -> classes whose name matches a caries keyword.
        - No match (e.g. the generic COCO fallback) -> treat every class as a
          finding so the pipeline still returns/draws boxes.
        """
        if len(self.names) <= 1:
            return set(self.names)
        keywords = settings.caries_classes
        ids = {
            i for i, name in self.names.items()
            if any(k in name.lower() for k in keywords)
        }
        return ids or set(self.names)

    @property
    def class_names(self) -> List[str]:
        return [self.names[k] for k in sorted(self.names)]

    # ------------------------------------------------------------------ #
    # Preprocessing
    # ------------------------------------------------------------------ #
    def _preprocess(self, image_bgr: np.ndarray) -> np.ndarray:
        """Optional contrast enhancement (CLAHE on the luminance channel).

        Dental X-rays and intraoral photos are often low-contrast; CLAHE makes
        early lesions more separable without distorting color too much.
        """
        if not settings.enhance_contrast:
            return image_bgr
        lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

    @staticmethod
    def decode_image(data: bytes) -> np.ndarray:
        """Decode raw image bytes into a BGR numpy array."""
        arr = np.frombuffer(data, dtype=np.uint8)
        image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Could not decode image — unsupported or corrupt file.")
        return image

    # ------------------------------------------------------------------ #
    # Inference
    # ------------------------------------------------------------------ #
    def detect(
        self, image_bgr: np.ndarray, conf: float | None = None
    ) -> Tuple[List[dict], int, int, float]:
        """Run detection.

        Returns (detections, width, height, inference_ms).
        Each detection: {label, confidence, box{x1,y1,x2,y2}, box_norm{...}}.
        `conf` overrides the default confidence threshold (e.g. stricter for X-rays).
        """
        height, width = image_bgr.shape[:2]
        processed = self._preprocess(image_bgr)

        start = time.perf_counter()
        results = self.model.predict(
            source=processed,
            conf=settings.conf_threshold if conf is None else conf,
            iou=settings.iou_threshold,
            imgsz=settings.infer_imgsz,
            verbose=False,
        )
        inference_ms = (time.perf_counter() - start) * 1000.0

        detections: List[dict] = []
        if results:
            result = results[0]
            for box in result.boxes:
                cls_id = int(box.cls[0])
                # Keep only caries-relevant classes (drops e.g. a plain "tooth"
                # box in a multi-class custom model).
                if cls_id not in self.caries_ids:
                    continue
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                label = self.names.get(cls_id, str(cls_id))
                detections.append(
                    {
                        "label": label,
                        "confidence": round(conf, 4),
                        "is_caries": True,
                        "box": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                        "box_norm": {
                            "x1": x1 / width,
                            "y1": y1 / height,
                            "x2": x2 / width,
                            "y2": y2 / height,
                        },
                    }
                )

        # Sort strongest-first so the UI can highlight the top finding.
        detections.sort(key=lambda d: d["confidence"], reverse=True)
        return detections, width, height, inference_ms


# --------------------------------------------------------------------------- #
# Model registry — multiple models coexist and can be switched at runtime.
# --------------------------------------------------------------------------- #
# Any *.pt file dropped in backend/models/ becomes a selectable "caries" model
# (its filename stem is the model name). The generic YOLO fallback is always
# offered too, so the app works even with no custom weights.

_GENERIC_NAME = "generic"

_cache: Dict[str, CariesDetector] = {}
_default_name: Optional[str] = None


def discover_models() -> List[dict]:
    """List selectable models (metadata only — does not load them)."""
    infos: List[dict] = []
    for pt in sorted(MODELS_DIR.glob("*.pt")):
        infos.append(
            {"name": pt.stem, "filename": pt.name, "type": "caries", "ref": str(pt)}
        )
    # Always include the generic fallback as a selectable option.
    infos.append(
        {
            "name": _GENERIC_NAME,
            "filename": settings.fallback_model,
            "type": "generic-fallback",
            "ref": settings.fallback_model,
        }
    )
    return infos


def default_model_name() -> str:
    """Preferred model: a file literally named 'caries', else any custom model,
    else the generic fallback."""
    global _default_name
    if _default_name is not None:
        return _default_name
    infos = discover_models()
    caries = [i for i in infos if i["type"] == "caries"]
    pick = next((i for i in caries if i["name"] == "caries"), None) or (
        caries[0] if caries else infos[-1]
    )
    _default_name = pick["name"]
    return _default_name


def set_default_model(name: str) -> str:
    """Set the server-wide default (active) model. Raises KeyError if unknown."""
    if name not in {i["name"] for i in discover_models()}:
        raise KeyError(name)
    global _default_name
    _default_name = name
    return name


def get_detector(name: Optional[str] = None) -> CariesDetector:
    """Return a (cached) detector for `name`, or the default model.

    Raises KeyError if the requested model does not exist.
    """
    infos = {i["name"]: i for i in discover_models()}
    if name is None:
        name = default_model_name()
    if name not in infos:
        raise KeyError(name)
    if name not in _cache:
        info = infos[name]
        _cache[name] = CariesDetector(info["ref"], info["name"], info["type"])
    return _cache[name]

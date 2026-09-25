"""Lightweight heuristic to tell dental X-rays apart from clinical photos.

X-rays are effectively grayscale (very low inter-channel difference and
saturation). Intraoral / phone photos have real color. This is intentionally
simple — good enough to route to the right YOLO weights without a second ML model.
"""
from __future__ import annotations

from typing import Literal

import cv2
import numpy as np

ImageKind = Literal["xray", "photo"]

# Tuned for typical dental radiographs vs phone/intraoral photos.
_COLORFULNESS_MAX = 10.0
_SATURATION_MAX = 28.0


def classify_image_kind(image_bgr: np.ndarray) -> ImageKind:
    """Return ``"xray"`` or ``"photo"`` for an OpenCV BGR image."""
    b, g, r = cv2.split(image_bgr)
    r64 = r.astype(np.float64)
    g64 = g.astype(np.float64)
    b64 = b.astype(np.float64)
    colorfulness = (
        np.mean(np.abs(r64 - g64))
        + np.mean(np.abs(r64 - b64))
        + np.mean(np.abs(g64 - b64))
    ) / 3.0

    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    saturation = float(np.mean(hsv[:, :, 1]))

    if colorfulness < _COLORFULNESS_MAX and saturation < _SATURATION_MAX:
        return "xray"
    return "photo"


def preferred_model_for_kind(kind: ImageKind, available: set[str]) -> str:
    """Map image kind → best installed model name.

    Preference:
      xray  → caries → any non-generic → generic
      photo → caries_photo → caries → any non-generic → generic
    """
    if kind == "xray":
        order = ("caries",)
    else:
        order = ("caries_photo", "caries")

    for name in order:
        if name in available:
            return name

    custom = sorted(n for n in available if n != "generic")
    if custom:
        return custom[0]
    return "generic"

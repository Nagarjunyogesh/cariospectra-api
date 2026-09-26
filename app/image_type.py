"""Lightweight heuristics to tell dental X-rays, intraoral photos, and
everything else (screens, rooms, UI screenshots) apart.

X-rays are effectively grayscale. Intraoral photos have pink/red tissue plus
bright tooth enamel. Colorful app icons on a laptop screen are neither — and
YOLO will happily box those icons as "caries" if we let it run.
"""
from __future__ import annotations

from typing import Literal

import cv2
import numpy as np

ImageKind = Literal["xray", "photo", "other"]

# Tuned for typical dental radiographs vs phone/intraoral photos.
_COLORFULNESS_MAX = 10.0
_SATURATION_MAX = 28.0


def classify_image_kind(image_bgr: np.ndarray) -> Literal["xray", "photo"]:
    """Return ``"xray"`` or ``"photo"`` from colour statistics only."""
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
        # A mostly-white webpage can look "grayscale on average" even when it
        # is covered in colorful app icons — that is not an X-ray.
        if _looks_like_ui_screenshot(image_bgr):
            return "photo"
        return "xray"
    return "photo"


def is_dental_image(image_bgr: np.ndarray, kind: Literal["xray", "photo"]) -> bool:
    """True only when the frame looks like a radiograph or an intraoral photo."""
    if image_bgr is None or image_bgr.size == 0:
        return False
    if kind == "xray":
        return _looks_like_radiograph(image_bgr)
    return _looks_like_intraoral(image_bgr)


def preferred_model_for_kind(kind: str, available: set[str]) -> str:
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


def _looks_like_radiograph(image_bgr: np.ndarray) -> bool:
    """Dental X-rays have both dark background and brighter enamel, not a flat grey."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    std = float(np.std(gray))
    p10, p90 = np.percentile(gray, (10, 90))
    return std >= 18.0 and float(p10) <= 90 and float(p90) >= 130


def _looks_like_intraoral(image_bgr: np.ndarray) -> bool:
    if _looks_like_ui_screenshot(image_bgr):
        return False
    return _has_oral_tissue(image_bgr)


def _looks_like_ui_screenshot(image_bgr: np.ndarray) -> bool:
    """Many small saturated blobs ≈ app icons / bookmarks on a phone or laptop screen."""
    h, w = image_bgr.shape[:2]
    n = float(h * w)
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    s, v = hsv[:, :, 1], hsv[:, :, 2]
    high_sat = ((s >= 90) & (v >= 70)).astype(np.uint8)
    _num, _labels, stats, _ = cv2.connectedComponentsWithStats(high_sat, connectivity=8)
    iconish = 0
    for i in range(1, stats.shape[0]):
        area = float(stats[i, cv2.CC_STAT_AREA])
        if not (0.0004 * n <= area <= 0.025 * n):
            continue
        bw = int(stats[i, cv2.CC_STAT_WIDTH])
        bh = int(stats[i, cv2.CC_STAT_HEIGHT])
        if bw < 4 or bh < 4:
            continue
        aspect = max(bw, bh) / min(bw, bh)
        if aspect < 2.8:
            iconish += 1
    return iconish >= 4


def _has_oral_tissue(image_bgr: np.ndarray) -> bool:
    """Intraoral frames have pink/red gingiva plus bright, low-saturation enamel."""
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    n = float(h.size)

    gum = ((h <= 15) | (h >= 165)) & (s >= 35) & (s <= 165) & (v >= 45) & (v <= 240)
    tooth = (s <= 110) & (v >= 145) & ~((h >= 85) & (h <= 135) & (s > 25))

    gum_frac = float(np.mean(gum))
    tooth_frac = float(np.mean(tooth))

    gum_u8 = gum.astype(np.uint8)
    _num, _labels, stats, _ = cv2.connectedComponentsWithStats(gum_u8, connectivity=8)
    largest_gum = 0.0
    for i in range(1, stats.shape[0]):
        largest_gum = max(largest_gum, float(stats[i, cv2.CC_STAT_AREA]) / n)

    if largest_gum >= 0.025 and tooth_frac >= 0.04:
        return True
    if gum_frac >= 0.06 and tooth_frac >= 0.03:
        return True
    # Extreme close-up of a single tooth: lots of enamel, a sliver of gum.
    if tooth_frac >= 0.28 and largest_gum >= 0.008:
        return True
    return False

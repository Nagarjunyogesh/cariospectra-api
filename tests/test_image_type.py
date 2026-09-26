"""Synthetic checks for dental vs non-dental routing."""
from __future__ import annotations

import numpy as np

from app.image_type import classify_image_kind, is_dental_image


def _ui_screenshot(h=480, w=800) -> np.ndarray:
    """White page + 8 saturated circular 'app icons' (the Google-shortcut case)."""
    img = np.full((h, w, 3), 245, dtype=np.uint8)
    colors = [
        (60, 60, 220),
        (40, 40, 200),
        (40, 200, 40),
        (20, 180, 240),
        (200, 80, 40),
        (180, 40, 180),
        (30, 160, 220),
        (50, 50, 50),
    ]
    for i, bgr in enumerate(colors):
        cx = 80 + i * 90
        cy = 220
        yy, xx = np.ogrid[:h, :w]
        mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= 28**2
        img[mask] = bgr
    return img


def _intraoral(h=480, w=640) -> np.ndarray:
    """Cream teeth on top, pink gingiva along the bottom."""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[: int(h * 0.62)] = (210, 225, 235)  # BGR enamel-ish
    img[int(h * 0.55) :] = (90, 90, 190)  # BGR pink/red gingiva
    return img


def test_ui_screenshot_is_photo_but_not_dental():
    img = _ui_screenshot()
    kind = classify_image_kind(img)
    assert is_dental_image(img, kind) is False


def test_intraoral_is_dental_photo():
    img = _intraoral()
    kind = classify_image_kind(img)
    assert kind == "photo"
    assert is_dental_image(img, kind) is True


def test_flat_gray_is_not_a_radiograph():
    img = np.full((400, 400, 3), 128, dtype=np.uint8)
    kind = classify_image_kind(img)
    assert kind == "xray"
    assert is_dental_image(img, kind) is False

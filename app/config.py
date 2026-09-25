"""Runtime configuration, read from environment variables.

Dependency-free (plain os.getenv). Export vars in the shell, or rely on defaults.
Weights are discovered from ``models/*.pt`` — drop files there to add models.
"""
from __future__ import annotations

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"

# Auto-load backend/.env (if python-dotenv is installed) before reading settings.
# override=True so a stale LLM_MODEL in the shell cannot pin a retired Groq id.
try:
    from dotenv import load_dotenv

    load_dotenv(BASE_DIR / ".env", override=True)
except ImportError:
    pass

# Groq shut these down on 2026-08-16. Keep old .env values working.
_GROQ_RETIRED_MODELS = {
    "llama-3.3-70b-versatile": "openai/gpt-oss-120b",
    "llama-3.1-8b-instant": "openai/gpt-oss-20b",
    "llama-3.1-70b-versatile": "openai/gpt-oss-120b",
}


def _resolve_llm_model(name: str) -> str:
    raw = (name or "").strip() or "openai/gpt-oss-120b"
    return _GROQ_RETIRED_MODELS.get(raw, raw)


def _get_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


class Settings:
    """Central settings object."""

    # Generic YOLO used when no custom weights exist (auto-downloaded by ultralytics).
    fallback_model: str = os.getenv("FALLBACK_MODEL", "yolov8n.pt")

    # Industry-standard YOLO default. The CORRECT value for a given model is its
    # F1-optimal confidence (peak of Ultralytics' F1_curve.png from validation);
    # set that per model once you train your own weights.
    conf_threshold: float = _get_float("CONF_THRESHOLD", 0.25)
    # Optional per-image-type override (set to the X-ray model's own F1-optimal
    # after training). Defaults to the standard threshold = no special-casing.
    xray_conf_threshold: float = _get_float("XRAY_CONF_THRESHOLD", 0.25)
    iou_threshold: float = _get_float("IOU_THRESHOLD", 0.45)
    infer_imgsz: int = int(os.getenv("INFER_IMGSZ", "640"))
    enhance_contrast: bool = os.getenv("ENHANCE_CONTRAST", "true").lower() == "true"
    # Free hosts (Render 512 MB) cannot hold the YOLOv8x X-ray model in RAM.
    # When true, skip caries.pt and serve photo weights only.
    light_memory: bool = os.getenv("LIGHT_MEMORY", "false").lower() == "true"

    # Multi-class models: class-name substrings that count as caries findings.
    # NOTE: "lesion" is intentionally excluded — the pretrained X-ray model has a
    # separate "Periapical lesion" class that is NOT caries and caused false hits.
    caries_classes: list[str] = [
        c.strip().lower()
        for c in os.getenv("CARIES_CLASSES", "caries,carie,cavity,decay").split(",")
        if c.strip()
    ]

    disclaimer: str = (
        "CarioSpectra provides a preliminary screening result only and is not a "
        "medical diagnosis. Please consult a qualified dentist for confirmation "
        "and treatment."
    )

    # Chatbot LLM (any OpenAI-compatible endpoint).
    #   Groq (default):  base https://api.groq.com/openai/v1, model openai/gpt-oss-120b
    #   Ollama (local):  base http://localhost:11434/v1,       model llama3.2:3b
    #   OpenRouter:      base https://openrouter.ai/api/v1,     model <free model>
    @property
    def llm_base_url(self) -> str:
        return os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1")

    @property
    def llm_api_key(self) -> str:
        return os.getenv("LLM_API_KEY", "")

    @property
    def llm_model(self) -> str:
        return _resolve_llm_model(os.getenv("LLM_MODEL", "openai/gpt-oss-120b"))


settings = Settings()

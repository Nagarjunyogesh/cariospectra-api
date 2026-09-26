"""Pydantic response models for the detection API."""
from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class Box(BaseModel):
    """Bounding box in absolute pixel coordinates (top-left, bottom-right)."""

    x1: float
    y1: float
    x2: float
    y2: float


class BoxNorm(BaseModel):
    """Bounding box normalized to 0..1 relative to image size.

    The mobile app multiplies these by its *displayed* image dimensions so the
    overlay lines up regardless of scaling.
    """

    x1: float
    y1: float
    x2: float
    y2: float


class Detection(BaseModel):
    label: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    is_caries: bool = True
    box: Box
    box_norm: BoxNorm


class DetectResponse(BaseModel):
    model: str
    model_type: Literal["caries", "generic-fallback"]
    model_name: str
    image_width: int
    image_height: int
    verdict: Literal["Caries Detected", "No Caries", "Not a dental image"]
    count: int
    detections: List[Detection]
    inference_ms: float
    disclaimer: str
    # Present when the client asked for model="auto" (live-scan routing).
    image_type: Literal["xray", "photo", "other"] | None = None
    auto_selected: bool = False


class HealthResponse(BaseModel):
    status: str
    active_model: str
    model_type: Literal["caries", "generic-fallback"]
    model_name: str
    classes: List[str]


class ModelInfo(BaseModel):
    name: str
    type: Literal["caries", "generic-fallback"]
    filename: str
    active: bool


class ModelsResponse(BaseModel):
    active: str
    models: List[ModelInfo]


class SetModelRequest(BaseModel):
    name: str


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    patient: Optional[dict] = None
    scans: Optional[List[dict]] = None


class ChatResponse(BaseModel):
    reply: str

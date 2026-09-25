"""CarioSpectra FastAPI application.

Endpoints:
  GET  /                 -> service banner
  GET  /health           -> model status + class names
  GET  /models           -> list installed weights
  POST /models/active    -> set server default model
  POST /detect           -> multipart image upload -> detections JSON
                           (pass model=auto to classify X-ray vs photo)
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from . import __version__
from .chat import chat_completion
from .config import settings
from .detector import (
    CariesDetector,
    default_model_name,
    discover_models,
    get_detector,
    set_default_model,
)
from .image_type import classify_image_kind, preferred_model_for_kind
from .schemas import (
    ChatRequest,
    ChatResponse,
    DetectResponse,
    HealthResponse,
    ModelInfo,
    ModelsResponse,
    SetModelRequest,
)

AUTO_MODEL = "auto"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("cariospectra")

MAX_UPLOAD_BYTES = 15 * 1024 * 1024  # 15 MB


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Warm the model at startup so the first request isn't slow.
    logger.info("Loading detection model...")
    get_detector()
    logger.info("Model ready.")
    yield


app = FastAPI(
    title="CarioSpectra API",
    description="Deep-learning dental caries detection service.",
    version=__version__,
    lifespan=lifespan,
)

# Allow the mobile app (any origin during development) to call the API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {
        "name": "CarioSpectra API",
        "version": __version__,
        "docs": "/docs",
        "disclaimer": settings.disclaimer,
    }


@app.get("/health", response_model=HealthResponse)
def health():
    detector = get_detector()
    return HealthResponse(
        status="ok",
        active_model=detector.name,
        model_type=detector.model_type,
        model_name=detector.model_name,
        classes=detector.class_names,
    )


def _models_response() -> ModelsResponse:
    active = default_model_name()
    return ModelsResponse(
        active=active,
        models=[
            ModelInfo(
                name=i["name"],
                type=i["type"],
                filename=i["filename"],
                active=(i["name"] == active),
            )
            for i in discover_models()
        ],
    )


@app.get("/models", response_model=ModelsResponse)
def list_models():
    """List every selectable model (custom weights in models/ + generic fallback)."""
    return _models_response()


@app.post("/models/active", response_model=ModelsResponse)
def set_active_model(req: SetModelRequest):
    """Set the server-wide default model, and warm it so the next call is fast."""
    try:
        set_default_model(req.name)
        get_detector(req.name)  # load + cache now
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Unknown model '{req.name}'.")
    return _models_response()


@app.post("/detect", response_model=DetectResponse)
async def detect(image: UploadFile = File(...), model: Optional[str] = Form(None)):
    if image.content_type and not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file is not an image.")

    data = await image.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty file.")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Image too large (max 15 MB).")

    try:
        image_bgr = CariesDetector.decode_image(data)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    image_type = None
    auto_selected = False
    resolved = model

    # Live scan (and any client) can pass model=auto: classify X-ray vs photo
    # and pick caries / caries_photo without the user choosing.
    if model == AUTO_MODEL:
        available = {i["name"] for i in discover_models()}
        image_type = classify_image_kind(image_bgr)
        resolved = preferred_model_for_kind(image_type, available)
        auto_selected = True
        logger.info("Auto model: image_type=%s → %s", image_type, resolved)

    try:
        detector = get_detector(resolved)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Unknown model '{model}'.")

    # X-rays over-flag on the pretrained model → require higher confidence.
    conf = settings.xray_conf_threshold if image_type == "xray" else None
    detections, width, height, inference_ms = detector.detect(image_bgr, conf=conf)
    verdict = "Caries Detected" if detections else "No Caries"

    return DetectResponse(
        model=detector.name,
        model_type=detector.model_type,
        model_name=detector.model_name,
        image_width=width,
        image_height=height,
        verdict=verdict,
        count=len(detections),
        detections=detections,
        inference_ms=round(inference_ms, 1),
        disclaimer=settings.disclaimer,
        image_type=image_type,
        auto_selected=auto_selected,
    )


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """Answer a patient question using their profile + screening history as context."""
    if not req.messages:
        raise HTTPException(status_code=400, detail="No messages provided.")
    messages = [{"role": m.role, "content": m.content} for m in req.messages]
    try:
        reply = await chat_completion(messages, req.patient, req.scans)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return ChatResponse(reply=reply)

# CarioSpectra backend — container image for Hugging Face Spaces (Docker SDK).
# HF free CPU Spaces: 2 vCPU / 16 GB RAM — plenty for CPU YOLO inference.
FROM python:3.11-slim

# System libs OpenCV needs (even the headless build wants libglib).
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

# HF Spaces run as UID 1000 — create a matching user with a writable HOME
# (Ultralytics/torch write caches under $HOME).
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1
WORKDIR /home/user/app

# Install CPU-only PyTorch first so ultralytics doesn't pull the huge CUDA build.
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements.txt

# App code.
COPY --chown=user . .

# Bake the two caries models into the image at build time (keeps the repo light).
RUN python scripts/get_model.py --source url \
      --url "https://huggingface.co/nsitnov/8024-yolov8-model/resolve/main/8024.pt" --name caries \
 && python scripts/get_model.py --source url \
      --url "https://github.com/AndreyGermanov/yolov8_caries_detector/raw/main/best.pt" --name caries_photo

# Hugging Face Spaces use 7860. Render/Railway/Fly set $PORT.
# LIGHT_MEMORY skips the 144 MB YOLOv8x X-ray weights (too large for 512 MB).
ENV PORT=7860 \
    LIGHT_MEMORY=true \
    INFER_IMGSZ=320 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1
EXPOSE 7860
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-7860}"]

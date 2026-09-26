#!/usr/bin/env bash
# Start CarioSpectra on a Lightning AI Studio.
# Studios forbid extra venvs — use the default conda env (cloudspace).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

export LIGHT_MEMORY="${LIGHT_MEMORY:-false}"
export INFER_IMGSZ="${INFER_IMGSZ:-640}"
export PORT="${PORT:-8000}"

if ! python -c "import ultralytics, fastapi, uvicorn" 2>/dev/null; then
  pip install --upgrade pip
  pip install -r requirements.txt
fi

mkdir -p models
if [[ ! -f models/caries.pt ]]; then
  python scripts/get_model.py --source url \
    --url "https://huggingface.co/nsitnov/8024-yolov8-model/resolve/main/8024.pt" \
    --name caries
fi
if [[ ! -f models/caries_photo.pt ]]; then
  python scripts/get_model.py --source url \
    --url "https://github.com/AndreyGermanov/yolov8_caries_detector/raw/main/best.pt" \
    --name caries_photo
fi

exec python -m uvicorn app.main:app --host 0.0.0.0 --port "$PORT"

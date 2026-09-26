#!/usr/bin/env bash
# Run once in a Lightning Studio terminal after the repo is on disk.
# Writes .lightning_studio/on_start.sh so the API comes back after each reboot.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"

if [[ -d /teamspace/studios/this_studio ]]; then
  STUDIO_HOME=/teamspace/studios/this_studio
else
  STUDIO_HOME="$HOME"
fi

mkdir -p "$STUDIO_HOME/.lightning_studio"
cat > "$STUDIO_HOME/.lightning_studio/on_start.sh" <<EOF
#!/bin/bash
# Auto-start CarioSpectra after each Studio boot (including 4h free restarts).
# Do not create a venv — Lightning allows only the default conda env.
cd "$ROOT"
export LIGHT_MEMORY=false INFER_IMGSZ=640 PORT=8000
nohup python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 >> /tmp/cariospectra.log 2>&1 &
EOF
chmod +x "$STUDIO_HOME/.lightning_studio/on_start.sh"

if [[ ! -f "$ROOT/.env" ]]; then
  echo "Create $ROOT/.env with your Groq key (do not commit it):"
  echo "  LLM_API_KEY=gsk_..."
  echo "  LLM_BASE_URL=https://api.groq.com/openai/v1"
  echo "  LLM_MODEL=openai/gpt-oss-120b"
  echo "  LIGHT_MEMORY=false"
  echo "  INFER_IMGSZ=640"
fi

echo "Wrote $STUDIO_HOME/.lightning_studio/on_start.sh"
echo "Start the API with:  bash $ROOT/scripts/lightning-start.sh"

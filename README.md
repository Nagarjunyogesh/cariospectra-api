# CarioSpectra Backend

FastAPI service that runs a YOLO dental-caries detection model and returns
bounding boxes + confidence scores as JSON.

## Setup

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

- Interactive docs: http://localhost:8000/docs
- Health check:     http://localhost:8000/health

`--host 0.0.0.0` matters: it lets your phone reach the server over the LAN. Use
your computer's LAN IP (e.g. `192.168.1.20`) in the mobile app config.

## Getting a caries model

Drop YOLO `.pt` weights into `backend/models/`. The service discovers every
`*.pt` file there (stem = model name: `caries.pt` → `caries`).

- **If you have trained weights:** copy them to `backend/models/caries.pt`
  (and optionally `caries_photo.pt` for clinical photos).
- **Pretrained from Roboflow Universe** (free account + API key):
  ```bash
  pip install roboflow
  export ROBOFLOW_API_KEY=xxxxxxxx
  python scripts/get_model.py --source roboflow \
      --workspace <ws> --project <proj> --version <n>
  ```
  Search https://universe.roboflow.com for "dental caries" / "tooth decay"
  detection projects and read the workspace/project/version from the model's
  deploy snippet.
- **Direct URL:** `python scripts/get_model.py --source url --url https://.../caries.pt`

**Fallback:** if no caries weights are found, the service loads a generic YOLO
model (`yolov8n.pt`, auto-downloaded) so the full app still works end-to-end.
Responses are flagged `"model_type": "generic-fallback"` and the detected labels
are generic COCO classes — swap in real caries weights for meaningful results.

## Auto routing (X-ray vs photo)

The mobile app always sends `model=auto`. The backend classifies the image
(grayscale ≈ X-ray → `caries`; color ≈ photo → `caries_photo`) and runs the
matching weights. You can still force a model via the API for debugging:

```bash
# list models
curl http://localhost:8000/models

# auto (same as the app)
curl -X POST http://localhost:8000/detect -F "image=@tooth.jpg" -F "model=auto"

# force a specific model
curl -X POST http://localhost:8000/detect -F "image=@tooth.jpg" -F "model=caries_photo"

# change the server-wide default (used when model is omitted)
curl -X POST http://localhost:8000/models/active \
     -H "Content-Type: application/json" -d '{"name":"caries"}'
```

## API

### `POST /detect`
Multipart form upload, field `image` (+ optional `model` to pick which model).

```bash
curl -X POST http://localhost:8000/detect -F "image=@tooth.jpg"
```

Response:
```jsonc
{
  "model_type": "caries",
  "model_name": "caries.pt",
  "image_width": 1280,
  "image_height": 960,
  "verdict": "Caries Detected",
  "count": 2,
  "detections": [
    {
      "label": "caries",
      "confidence": 0.91,
      "box":      { "x1": 402, "y1": 210, "x2": 470, "y2": 288 },
      "box_norm": { "x1": 0.314, "y1": 0.219, "x2": 0.367, "y2": 0.300 }
    }
  ],
  "inference_ms": 88.4,
  "disclaimer": "CarioSpectra provides a preliminary screening result only ..."
}
```

The mobile app draws boxes from `box_norm` (multiply by the displayed image size).

### `POST /chat`
Dental-health assistant. The app sends the conversation plus the patient's
profile + recent scans; the backend adds a guarded system prompt and calls the
configured LLM.

```jsonc
// request
{ "messages": [{"role":"user","content":"What do my results mean?"}],
  "patient": { "full_name": "...", "dob": "2000-05-01", "sex": "..." },
  "scans":   [{ "created_at":"...", "verdict":"Caries Detected", "count":2 }] }
// response
{ "reply": "..." }
```

**Setup (Groq, free):** get a key at https://console.groq.com/keys, then run the
backend with it set:
```bash
export LLM_API_KEY=gsk_your_key         # Windows: set LLM_API_KEY=gsk_your_key
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
Defaults to `openai/gpt-oss-120b`. To use local Ollama instead, set
`LLM_BASE_URL=http://localhost:11434/v1`, `LLM_MODEL=llama3.2:3b`, and leave
`LLM_API_KEY` empty. Until a key (or Ollama) is configured, `/chat` returns a
clear 503.

## Configuration

Copy `.env.example` and export vars if needed. See that file for options:
`FALLBACK_MODEL`, `CONF_THRESHOLD`, `IOU_THRESHOLD`, `INFER_IMGSZ`,
`ENHANCE_CONTRAST`, `CARIES_CLASSES`.
Weights live in `models/*.pt` (discovered automatically).

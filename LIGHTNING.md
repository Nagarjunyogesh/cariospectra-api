# CarioSpectra API on Lightning AI (no credit card)

Lightning gives one free CPU Studio with enough RAM for both YOLO models.
No card. The Studio **restarts every 4 hours**; the start script below
brings the API back automatically.

You create the account and Studio in the browser. That cannot be done from
this repo.

## 1. Create a Studio

1. Sign up at [https://lightning.ai](https://lightning.ai) (GitHub or email).
2. **New Studio** → keep the default **CPU** machine.
3. Open the Studio terminal.

## 2. Clone and start the API

Paste this in the Studio terminal. Replace `gsk_...` with the Groq key from
your laptop `backend/.env` — do not commit that file.

```bash
cd /teamspace/studios/this_studio
git clone https://github.com/Nagarjunyogesh/cariospectra-api.git
cd cariospectra-api

cat > .env <<'EOF'
LIGHT_MEMORY=false
INFER_IMGSZ=640
LLM_BASE_URL=https://api.groq.com/openai/v1
LLM_MODEL=openai/gpt-oss-120b
LLM_API_KEY=gsk_PASTE_YOUR_KEY
EOF

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python scripts/get_model.py --source url \
  --url "https://huggingface.co/nsitnov/8024-yolov8-model/resolve/main/8024.pt" --name caries
python scripts/get_model.py --source url \
  --url "https://github.com/AndreyGermanov/yolov8_caries_detector/raw/main/best.pt" --name caries_photo

mkdir -p ~/.lightning_studio
cat > ~/.lightning_studio/on_start.sh <<'EOF'
#!/bin/bash
cd /teamspace/studios/this_studio/cariospectra-api
source .venv/bin/activate
export LIGHT_MEMORY=false INFER_IMGSZ=640 PORT=8000
nohup uvicorn app.main:app --host 0.0.0.0 --port 8000 >> /tmp/cariospectra.log 2>&1 &
EOF
chmod +x ~/.lightning_studio/on_start.sh

export LIGHT_MEMORY=false INFER_IMGSZ=640
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

If this laptop already has `scripts/lightning-start.sh` (after you upload
`backend/` or push it), you can use that instead of the long block:

```bash
bash scripts/lightning-studio-bootstrap.sh
bash scripts/lightning-start.sh
```

First run installs CPU PyTorch and downloads both `.pt` files (10–20 minutes).
Later starts only boot uvicorn.

Leave that terminal running. Health check in a second terminal:

```bash
curl http://127.0.0.1:8000/health
```

You want JSON with `"ok": true`.

## 3. Get a public HTTPS URL

1. In the Studio, open the **Port viewer** plugin (or **Deploy on public ports**).
2. Expose / share **port 8000**.
3. Copy the public URL (it will look like `https://….lightning.ai`).
4. Open `https://THAT_URL/health` in a browser.

Optional: turn on **Auto start** so the Studio sleeps when idle and wakes when
the app calls it. The first scan after sleep is slow (Studio boot + model load).

## 4. Point Expo at Lightning

In `mobile/.env`:

```
EXPO_PUBLIC_API_BASE_URL=https://YOUR-LIGHTNING-URL
```

No trailing slash. Then:

```bash
cd mobile && npx expo start -c
```

## After the 4-hour restart

The Studio filesystem (venv, models, `.env`) stays on disk.
`~/.lightning_studio/on_start.sh` starts uvicorn again.

If `/health` fails after a reboot, re-share port 8000 in Port viewer and
confirm `/tmp/cariospectra.log` on the Studio.

## What this Studio runs

| Setting        | Value   | Why                                      |
|----------------|---------|------------------------------------------|
| `LIGHT_MEMORY` | `false` | Photo + X-ray weights both available     |
| `INFER_IMGSZ`  | `640`   | Full YOLO size (Render had to use `160`) |
| Port           | `8000`  | FastAPI default                          |

Only one model is in RAM at a time, so the default CPU Studio (~7.5 GB) is
enough. If a scan still OOMs, switch that Studio to a larger CPU machine
(`cpu-4`, ~15 GB) in the Studio hardware menu.

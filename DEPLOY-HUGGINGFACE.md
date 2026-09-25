# Deploying the CarioSpectra backend

Result: a public **https** URL the phone app can call from any network — no
laptop, no same-Wi‑Fi requirement.

## Hugging Face Spaces (needs PRO)

Docker Spaces on the free `cpu-basic` hardware now require a
[Hugging Face PRO](https://huggingface.co/pro) plan. Creating
`NagarjunYogesh/cariospectra-api` as Docker returns HTTP 402.

If you have PRO, follow the original steps below (SDK **Docker**, `app_port: 7860`).

## Render (free alternative)

1. Push this `backend/` folder to GitHub (or use the `cariospectra-api` repo).
2. Sign in at https://dashboard.render.com
3. **New → Blueprint** and select that repo (`render.yaml`), **or**
   **New → Web Service → Docker** pointing at this folder.
4. Set secret **`LLM_API_KEY`** (your Groq key) in the service Environment tab.
5. First build downloads PyTorch + both YOLO weights — several minutes.
6. Open `https://<service>.onrender.com/health`
7. Point the app at it:
   ```
   EXPO_PUBLIC_API_BASE_URL=https://<service>.onrender.com
   ```

Free Render services sleep after inactivity; the first request after that is slow.

---

## 1. Create the Space
1. Sign up (free) at https://huggingface.co
2. Go to https://huggingface.co/new-space
   - **Owner:** you · **Space name:** `cariospectra-api`
   - **SDK:** **Docker** → *Blank* template
   - **Visibility:** Public (simplest) — or Private (then the app must send an HF token)
3. Create it. This makes a git repo for the Space.

## 2. Set the app port + metadata
In the Space's auto-created **`README.md`**, make the frontmatter look like this
(the important line is `app_port: 7860`):
```yaml
---
title: CarioSpectra API
emoji: 🦷
sdk: docker
app_port: 7860
pinned: false
---
```

## 3. Push the backend code
The Space repo root must contain the `Dockerfile`, `requirements.txt`, `app/`,
and `scripts/`. From your machine:

```bash
git clone https://huggingface.co/spaces/<your-username>/cariospectra-api
cd cariospectra-api

# copy the backend files into the Space repo (from this backend/ folder)
cp -r /path/to/ml-app/backend/{Dockerfile,.dockerignore,requirements.txt,app,scripts} .

git add .
git commit -m "Deploy CarioSpectra backend"
git push
```
- When git asks for a password, use a **Hugging Face access token** (create one at
  https://huggingface.co/settings/tokens with **write** scope) — not your login password.
- `.env` and the model `.pt` files are **not** pushed (git-ignored). The Dockerfile
  downloads the two caries models automatically during the build.

> No git? Use the Space's **Files → Add file → Upload** UI and drag in the same
> files (keep the `app/` and `scripts/` folder structure).

## 4. Add your secrets (Space → Settings → Variables and secrets)
- **Secret** — `LLM_API_KEY` = your Groq key (for the chatbot)
- **Variables** (optional, to match your local tuning):
  - `XRAY_CONF_THRESHOLD` = `0.3`
  - `CARIES_CLASSES` = `caries,carie,cavity,decay`

## 5. Wait for the build
The Space builds the Docker image (installs PyTorch, downloads the models) — a few
minutes. Watch the **Logs** tab. When it says the app is running:

- Test in a browser: `https://<your-username>-cariospectra-api.hf.space/health`
  → should return `{"status":"ok", ...}`

## 6. Point the app at it
Set the endpoint (no trailing slash):

- **Expo Go / local dev:** in `mobile/.env`
  ```
  EXPO_PUBLIC_API_BASE_URL=https://<your-username>-cariospectra-api.hf.space
  ```
- **APK / EAS build:** the same key under `build.*.env` in `eas.json`

Restart Expo (`npx expo start -c`) and scan — it now talks to the hosted backend
over https, from any network.

---

## Notes
- **Sleeping:** free CPU Spaces pause after ~48 h of inactivity; the first request
  after that wakes it (slow), or hit "Restart" on the Space page.
- **Speed:** CPU inference is ~1–4 s per image (the 137 MB X-ray model is the slower one).
- **Security:** never commit `.env` — secrets live only in the Space settings.
- Everything (auto model routing, `/chat`, reports) works unchanged; only the URL moves.

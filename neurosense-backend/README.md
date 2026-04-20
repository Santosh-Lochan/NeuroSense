# NeuroSense — Integration Guide

## Project Structure

```
neurosense-backend/         ← Python FastAPI server
  main.py
  requirements.txt
  models/                   ← Put your .pth, .joblib files here
    neurosense_v4_bimodal_attn.pth
    neurosense_v5_svm.joblib
    neurosense_v5_scaler.joblib

neurosense-frontend/        ← React web app
  package.json
  public/
    index.html
  src/
    App.jsx
    index.css
    index.js
```

---

## STEP 1 — Prerequisites

Install these on your machine if not already present:

- **Node.js** (v18+): https://nodejs.org
- **Python** (3.10+): https://python.org
- **ffmpeg**: Required for audio conversion
  - macOS:   `brew install ffmpeg`
  - Ubuntu:  `sudo apt install ffmpeg`
  - Windows: https://ffmpeg.org/download.html (add to PATH)

---

## STEP 2 — Set Up the Backend

```bash
cd neurosense-backend

# 1. Create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Place your model files
mkdir -p models
# Copy these 3 files from your Google Drive into models/:
#   neurosense_v4_bimodal_attn.pth
#   neurosense_v5_svm.joblib
#   neurosense_v5_scaler.joblib

# 4. Set your Anthropic API key
export ANTHROPIC_API_KEY="sk-ant-..."   # Windows: set ANTHROPIC_API_KEY=sk-ant-...

# 5. Start the server
uvicorn main:app --reload --port 8000
```

✅ You should see: `NeuroSense backend is live`
Test it: open http://localhost:8000/health in your browser.

---

## STEP 3 — Set Up the Frontend

```bash
cd neurosense-frontend

# 1. Install dependencies
npm install

# 2. Start the dev server
npm start
```

This opens http://localhost:3000 in your browser automatically.

---

## STEP 4 — Run a Full Session

1. Make sure the **backend is running** on port 8000
2. Open the **frontend** on port 3000
3. Click **Begin Session** — grant mic + camera permissions
4. **Hold the button** to speak your answer, **release** to send
5. Lumi (Claude) will respond and guide you through 9 DAIC-WOZ questions
6. After the interview completes, the app sends audio + video to the backend
7. The result screen shows the prediction + a **Download Transcript** button

---

## STEP 5 — The Downloaded Transcript

The `.txt` file follows the DAIC-WOZ format exactly:

```
start_time	stop_time	speaker	value
36.588	39.868	Ellie	hello i'm lumi thanks for joining today
62.320	63.178	Participant	good
63.790	64.738	Ellie	that's good how have you been sleeping
...
```

- `start_time` / `stop_time` — seconds elapsed since session start
- `speaker` — `Ellie` (Lumi) or `Participant` (user)
- `value` — lowercased transcript text (matches DAIC-WOZ convention)

---

## Environment Variables

| Variable            | Required | Description                        |
|---------------------|----------|------------------------------------|
| ANTHROPIC_API_KEY   | ✅ Yes   | Your Anthropic API key             |

---

## Common Issues

| Problem | Fix |
|---|---|
| "Camera/mic permission denied" | Use HTTPS or localhost only |
| ffmpeg not found | Install ffmpeg and ensure it's in your PATH |
| Backend CORS error | Make sure backend is running on port 8000 |
| Whisper slow on first run | It downloads the model (~150MB) once |
| SVM not found warning | Copy .joblib files into models/ |

---

## For Deployment (Later)

- **Backend**: Deploy to [Render](https://render.com) or [Railway](https://railway.app) — both support Python + GPU
- **Frontend**: Deploy to [Vercel](https://vercel.com) — change `BACKEND` constant in `App.jsx` to your deployed URL
- Set `ANTHROPIC_API_KEY` as an environment variable in your hosting dashboard

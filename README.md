# NeuroSense

**A multimodal AI-powered depression screening prototype built on the DAIC-WOZ clinical interview protocol.**

NeuroSense conducts a structured clinical interview through a conversational AI companion named **Lumi**, records audio and video, extracts multimodal features, and runs them through a trained hybrid deep learning + SVM pipeline to produce a depression likelihood score.

> ⚠️ **Research Prototype** — NeuroSense is an academic research tool and does not constitute a medical diagnosis. It should not be used as a substitute for professional mental health evaluation.

---

## Demo

![NeuroSense Welcome Screen](https://i.imgur.com/placeholder.png)

> *Replace the above with an actual screenshot once deployed.*

---

## How It Works

```
User speaks to Lumi (LLaMA via Groq)
         ↓
Browser records full-session audio + video
         ↓
Whisper  →  transcript  →  BERT  →  text_feat   [768-dim]
Audio    →  WavLM                →  wavlm_feat  [768-dim]
         ↓
BiModalGatedAttention (frozen V4 PyTorch)
         →  64-dim fused attention vector
         ↓
StandardScaler  →  SVM (V5)
         →  Binary prediction + % likelihood score
         ↓
DAIC-WOZ formatted transcript (.txt) available for download
```

### AI Roles

| Component | Role |
|---|---|
| **LLaMA 3.1 8B (Groq)** | Conducts the DAIC-WOZ interview as Lumi |
| **Whisper (OpenAI)** | Transcribes user speech to text |
| **BERT** | Encodes transcript into semantic embeddings |
| **WavLM (Microsoft)** | Encodes raw audio into acoustic embeddings |
| **BiModalGatedAttention** | Fuses text + audio with learned attention weights |
| **SVM** | Final binary classification (PHQ ≥ 10 threshold) |

---

## Project Structure

```
neurosense/
├── neurosense-backend/         # Python FastAPI server
│   ├── main.py                 # All endpoints + feature extraction + inference
│   ├── requirements.txt
│   └── models/                 # Model weights (not tracked by git)
│       ├── neurosense_v4_bimodal_attn.pth
│       ├── neurosense_v5_svm.joblib
│       └── neurosense_v5_scaler.joblib
│
└── neurosense-frontend/        # React web app
    ├── package.json
    ├── public/
    │   └── index.html
    └── src/
        ├── App.jsx             # Main interview app + all screens
        ├── App.css
        ├── Orb.jsx             # Animated Lumi mascot
        ├── Orb.css
        ├── index.js
        └── index.css
```

---

## Prerequisites

| Tool | Version | Notes |
|---|---|---|
| Python | 3.10+ | |
| Node.js | 18+ | |
| ffmpeg | Any | Must be in system PATH |
| Groq API Key | — | Free tier at console.groq.com |

---

## Setup & Running Locally

### 1. Clone the repo

```bash
git clone https://github.com/YOUR_USERNAME/neurosense.git
cd neurosense
```

### 2. Add model weights

Download your trained model files and place them in `neurosense-backend/models/`:

```
neurosense-backend/models/
├── neurosense_v4_bimodal_attn.pth
├── neurosense_v5_svm.joblib
└── neurosense_v5_scaler.joblib
```

> These files are excluded from git via `.gitignore` because they are large binary files. Store them in Google Drive or another file host and document the download link for collaborators.

### 3. Start the backend

```bash
cd neurosense-backend

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt

export GROQ_API_KEY="gsk_your_key_here"   # Windows: set GROQ_API_KEY=...

uvicorn main:app --reload --port 8000
```

Confirm it's running: http://localhost:8000/health

### 4. Start the frontend

```bash
cd neurosense-frontend
npm install
npm start
```

App opens at: http://localhost:3000

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Server status check |
| `POST` | `/transcribe` | Audio blob → Whisper transcript |
| `POST` | `/chat` | User message → Lumi (LLaMA) response |
| `POST` | `/analyze` | Full session audio + video → prediction + score |

---

## Transcript Format

Downloaded transcripts follow the **DAIC-WOZ format** exactly:

```
start_time	stop_time	speaker	value
36.588	39.868	Ellie	hello i'm lumi thanks for joining today
62.320	63.178	Participant	good
63.790	67.240	Ellie	that's good how have you been sleeping lately
```

- `Ellie` = Lumi's turns (matches original DAIC-WOZ interviewer naming)
- `Participant` = user turns
- Timestamps are seconds elapsed from session start
- All text is lowercased to match dataset convention

---

## Model Performance (V5)

Evaluated on 142 patients (DAIC-WOZ train + dev split), 5-fold cross-validation:

| Metric | Score |
|---|---|
| Accuracy | 69.0% |
| Precision | 53.0% |
| Recall | 61.1% |
| F1-Score | 52.8% |

> The model uses a binary threshold of PHQ-10 ≥ 10. The displayed percentage score is derived from the SVM's decision function via sigmoid transformation — it reflects model confidence, not a clinical PHQ score.

---

## Architecture

### Frontend
- **React 18** — interview UI, orb mascot, session management
- **Web Speech API** — text-to-speech for Lumi's responses
- **MediaRecorder API** — captures full-session audio and video
- **SVG gauge** — animated circular score display

### Backend
- **FastAPI** — async Python server
- **AsyncGroq** — non-blocking LLM calls for Lumi
- **asyncio.to_thread** — all CPU-heavy inference offloaded from event loop
- **ffmpeg** — audio/video format conversion

---

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `GROQ_API_KEY` | ✅ | Your Groq API key (console.groq.com) |

---

## Roadmap

- [ ] Replace Whisper CPU inference with Groq's Whisper API for faster transcription
- [ ] Add PHQ-9 self-report as a parallel input stream
- [ ] Deploy backend to Render / Railway with GPU support
- [ ] Deploy frontend to Vercel
- [ ] Increase training dataset size beyond 142 patients

---

## Acknowledgements

- [DAIC-WOZ Dataset](https://dcapswoz.ict.usc.edu/) — USC Institute for Creative Technologies
- [WavLM](https://arxiv.org/abs/2110.13900) — Microsoft Research
- [Whisper](https://openai.com/research/whisper) — OpenAI
- [Groq](https://groq.com) — LLaMA inference

---

## License

MIT License. See `LICENSE` for details.

> This project was built for academic research purposes. The authors make no clinical claims about the system's diagnostic accuracy.

"""
NeuroSense Backend — FastAPI  (v3 — bugs fixed + full logging)
Endpoints:
  GET  /health
  POST /transcribe   — audio blob → whisper transcript
  POST /chat         — message → Lumi (Groq / LLaMA) reply
  POST /analyze      — full session audio + video + transcript → prediction

Bug fixes in this version:
  - /tmp replaced with tempfile.gettempdir() — works on Windows
  - /chat wrapped in try/except with clear AuthenticationError message
  - Proper Python logging throughout (replaces bare print statements)
  - Every endpoint logs entry, key steps, timing, and errors
"""

import os, uuid, shutil, subprocess, asyncio, logging, time, tempfile
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import torch
import torch.nn as nn
import numpy as np
import joblib

from fastapi import FastAPI, File, UploadFile, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

import whisper
from transformers import (
    AutoTokenizer, AutoModel,
    WavLMModel, AutoFeatureExtractor,
)
import librosa
import opensmile
import cv2
from torchvision import models, transforms
from groq import AsyncGroq, AuthenticationError, RateLimitError, APIError

# ─── LOGGING SETUP ────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("neurosense")

# ─── CONFIG ───────────────────────────────────────────────────────────────────
MODELS_DIR    = Path("models")
PT_MODEL_PATH = MODELS_DIR / "neurosense_v4_bimodal_attn.pth"
SVM_PATH      = MODELS_DIR / "neurosense_v5_svm.joblib"
SCALER_PATH   = MODELS_DIR / "neurosense_v5_scaler.joblib"
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TMPDIR        = tempfile.gettempdir()   # /tmp on Linux/Mac, C:\Users\...\AppData\Local\Temp on Windows

log.info(f"Device: {DEVICE}")
log.info(f"Temp directory: {TMPDIR}")

cpu_pool = ThreadPoolExecutor(max_workers=2)

# ─── FASTAPI APP ──────────────────────────────────────────────────────────────
app = FastAPI(title="NeuroSense API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── REQUEST LOGGING MIDDLEWARE ───────────────────────────────────────────────
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()
    log.info(f"→ {request.method} {request.url.path}")
    response = await call_next(request)
    elapsed  = (time.time() - start) * 1000
    log.info(f"← {request.method} {request.url.path}  [{response.status_code}]  {elapsed:.0f}ms")
    return response

# ─── LOAD MODELS AT STARTUP ───────────────────────────────────────────────────
log.info("=" * 55)
log.info("NeuroSense backend starting — loading models...")
log.info("=" * 55)

log.info("[1/6] Loading Whisper (base)...")
t0 = time.time()
whisper_model = whisper.load_model("base")
log.info(f"      Whisper loaded in {time.time()-t0:.1f}s")

log.info("[2/6] Loading BERT (bert-base-uncased)...")
t0 = time.time()
bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
bert_model     = AutoModel.from_pretrained("bert-base-uncased").eval()
log.info(f"      BERT loaded in {time.time()-t0:.1f}s")

log.info("[3/6] Loading WavLM (microsoft/wavlm-base)...")
t0 = time.time()
wavlm_extractor = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-base")
wavlm_model     = WavLMModel.from_pretrained("microsoft/wavlm-base").eval()
log.info(f"      WavLM loaded in {time.time()-t0:.1f}s")

log.info("[4/6] Loading OpenSMILE (eGeMAPSv02)...")
t0 = time.time()
smile_extractor = opensmile.Smile(
    feature_set=opensmile.FeatureSet.eGeMAPSv02,
    feature_level=opensmile.FeatureLevel.Functionals,
)
log.info(f"      OpenSMILE loaded in {time.time()-t0:.1f}s")

log.info("[5/6] Loading ResNet50 (vision)...")
t0 = time.time()
resnet = models.resnet50(pretrained=True)
resnet.fc = nn.Identity()
resnet.eval()
vision_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
log.info(f"      ResNet50 loaded in {time.time()-t0:.1f}s")

log.info("[6/6] Loading NeuroSense PyTorch model + SVM...")

class BiModalGatedAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.text_enc = nn.Sequential(
            nn.LayerNorm(768), nn.Linear(768, 64),
            nn.LayerNorm(64),  nn.LeakyReLU(), nn.Dropout(0.4)
        )
        self.wav_enc = nn.Sequential(
            nn.LayerNorm(768), nn.Linear(768, 64),
            nn.LayerNorm(64),  nn.LeakyReLU(), nn.Dropout(0.4)
        )
        self.attention_scorer = nn.Linear(64, 1)
        self.classifier = nn.Sequential(
            nn.Linear(64, 32), nn.LeakyReLU(), nn.Dropout(0.4), nn.Linear(32, 1)
        )

    def forward(self, text, wavlm, smile=None, vision=None):
        t = self.text_enc(text)
        w = self.wav_enc(wavlm)
        stacked = torch.stack([t, w], dim=1)
        attn    = torch.softmax(self.attention_scorer(stacked), dim=1)
        return torch.sum(stacked * attn, dim=1)

ns_model = BiModalGatedAttention().to(DEVICE)
if PT_MODEL_PATH.exists():
    ns_model.load_state_dict(torch.load(PT_MODEL_PATH, map_location=DEVICE))
    log.info("      ✅ PyTorch weights loaded.")
else:
    log.warning("      ⚠️  PyTorch weights NOT found — random init (results meaningless)")
ns_model.eval()

svm_clf    = joblib.load(SVM_PATH)    if SVM_PATH.exists()    else None
svm_scaler = joblib.load(SCALER_PATH) if SCALER_PATH.exists() else None
if svm_clf:
    log.info("      ✅ SVM + Scaler loaded.")
else:
    log.warning("      ⚠️  SVM/Scaler NOT found — /analyze will return dummy output")

log.info("=" * 55)
log.info("All models ready. Server starting.")
log.info("=" * 55)

# ─── GROQ CLIENT ──────────────────────────────────────────────────────────────
_groq_key = os.environ.get("GROQ_API_KEY", "")
if not _groq_key:
    log.warning("⚠️  GROQ_API_KEY is not set — /chat will fail with 401")
else:
    log.info("✅ GROQ_API_KEY found.")

groq_client = AsyncGroq(api_key=_groq_key)

LUMI_SYSTEM = """
You are Lumi, a warm, empathetic mental wellness companion conducting a structured
clinical screening interview based on the DAIC-WOZ protocol.

Rules:
- Ask ONE question at a time. Be conversational, never clinical or cold.
- Do not offer diagnoses or medical advice.
- If the user seems distressed, respond with compassion before continuing.
- Cover all the following topics naturally (not as a numbered list):
    1. General mood / how they have been feeling lately
    2. Sleep quality and duration
    3. Feelings of sadness, hopelessness, or emptiness
    4. Loss of interest or pleasure in activities
    5. Energy levels / fatigue
    6. Concentration and focus
    7. Appetite and weight changes
    8. Psychomotor changes (slowing down or restlessness)
    9. Thoughts of self-harm or being better off dead
- After all topics are covered naturally, say a warm closing message and end it with
  exactly: [INTERVIEW_COMPLETE]
"""

sessions: dict = {}

# ─── UTILITY ──────────────────────────────────────────────────────────────────
def tmpfile(sid: str, suffix: str) -> str:
    return os.path.join(TMPDIR, f"ns_{sid}{suffix}")

def to_wav(input_path: str, output_path: str):
    """ffmpeg: any audio/video → 16kHz mono WAV. Blocking — use via asyncio.to_thread."""
    log.info(f"  [ffmpeg] Converting {os.path.basename(input_path)} → WAV")
    result = subprocess.run(
        ["ffmpeg", "-y", "-i", input_path, "-ar", "16000", "-ac", "1", output_path],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        log.error(f"  [ffmpeg] FAILED:\n{result.stderr[-500:]}")
        raise RuntimeError(f"ffmpeg conversion failed: {result.stderr[-200:]}")
    log.info(f"  [ffmpeg] Done → {os.path.basename(output_path)}")

# ─── FEATURE EXTRACTION ───────────────────────────────────────────────────────
def _transcribe_wav(wav_path: str) -> str:
    log.info("  [Whisper] Transcribing audio...")
    t0     = time.time()
    result = whisper_model.transcribe(wav_path)
    text   = result["text"].strip()
    log.info(f"  [Whisper] Done in {time.time()-t0:.1f}s — '{text[:80]}{'...' if len(text)>80 else ''}'")
    return text

def _extract_text_features(text: str) -> torch.Tensor:
    log.info("  [BERT] Extracting text embedding...")
    t0     = time.time()
    inputs = bert_tokenizer(text, return_tensors="pt",
                            truncation=True, max_length=512, padding=True)
    with torch.no_grad():
        out = bert_model(**inputs)
    feat = out.last_hidden_state[:, 0, :]
    log.info(f"  [BERT] Done in {time.time()-t0:.1f}s — shape {feat.shape}")
    return feat

def _extract_wavlm_features(wav_path: str) -> torch.Tensor:
    log.info("  [WavLM] Extracting acoustic embedding...")
    t0       = time.time()
    audio, _ = librosa.load(wav_path, sr=16000)
    inputs   = wavlm_extractor(audio, sampling_rate=16000, return_tensors="pt")
    with torch.no_grad():
        out = wavlm_model(**inputs)
    feat = out.last_hidden_state.mean(dim=1)
    log.info(f"  [WavLM] Done in {time.time()-t0:.1f}s — shape {feat.shape}")
    return feat

def _extract_vision_features(video_path: str) -> torch.Tensor:
    log.info("  [ResNet] Extracting vision features...")
    t0    = time.time()
    cap   = cv2.VideoCapture(video_path)
    feats = []
    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        if count % 60 == 0:
            rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            tensor = vision_transform(rgb).unsqueeze(0)
            with torch.no_grad():
                feats.append(resnet(tensor))
        count += 1
    cap.release()
    feat = torch.stack(feats).mean(dim=0) if feats else torch.zeros(1, 2048)
    log.info(f"  [ResNet] Done in {time.time()-t0:.1f}s — {len(feats)} frames sampled")
    return feat

def _run_neurosense_inference(text_feat: torch.Tensor,
                               wavlm_feat: torch.Tensor) -> dict:
    log.info("  [Inference] Running BiModalGatedAttention + SVM...")
    t0 = time.time()

    if svm_clf is None or svm_scaler is None:
        log.warning("  [Inference] SVM not loaded — returning dummy result")
        return {"prediction": -1, "label": "Model not loaded", "confidence": 0.0}

    text_feat  = text_feat.to(DEVICE)
    wavlm_feat = wavlm_feat.to(DEVICE)

    with torch.no_grad():
        t       = ns_model.text_enc(text_feat)
        w       = ns_model.wav_enc(wavlm_feat)
        stacked = torch.stack([t, w], dim=1)
        attn    = torch.softmax(ns_model.attention_scorer(stacked), dim=1)
        fused   = torch.sum(stacked * attn, dim=1).cpu().numpy()

    fused_scaled = svm_scaler.transform(fused)
    prediction   = int(svm_clf.predict(fused_scaled)[0])
    confidence   = float(svm_clf.decision_function(fused_scaled)[0])

    log.info(f"  [Inference] Done in {time.time()-t0:.1f}s — "
             f"prediction={prediction}, confidence={confidence:.4f}")
    return {
        "prediction": prediction,
        "label":      "Positive" if prediction == 1 else "Negative",
        "confidence": confidence,
    }

# ─── ENDPOINTS ────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    log.info("[health] Ping received")
    return {
        "status":       "NeuroSense backend is live",
        "device":       str(DEVICE),
        "models_ready": svm_clf is not None,
        "groq_key_set": bool(_groq_key),
    }


@app.post("/transcribe")
async def transcribe(audio: UploadFile = File(...)):
    sid      = uuid.uuid4().hex[:8]
    raw_path = tmpfile(sid, "_raw_audio")
    wav_path = tmpfile(sid, ".wav")
    log.info(f"[transcribe] Session={sid} | file={audio.filename} | size≈{audio.size}B")
    try:
        with open(raw_path, "wb") as f:
            shutil.copyfileobj(audio.file, f)
        file_size = os.path.getsize(raw_path)
        log.info(f"[transcribe] Saved raw audio — {file_size} bytes")

        await asyncio.to_thread(to_wav, raw_path, wav_path)
        transcript = await asyncio.to_thread(_transcribe_wav, wav_path)

        log.info(f"[transcribe] ✅ Complete — transcript length: {len(transcript)} chars")
        return {"transcript": transcript}

    except Exception as e:
        log.error(f"[transcribe] ❌ ERROR: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"error": str(e)})
    finally:
        for p in (raw_path, wav_path):
            if os.path.exists(p): os.remove(p)


@app.post("/chat")
async def chat(request: dict):
    session_id = request.get("session_id", "default")
    user_msg   = request.get("message", "")
    log.info(f"[chat] Session={session_id[:8]} | user='{user_msg[:60]}{'...' if len(user_msg)>60 else ''}'")

    if session_id not in sessions:
        sessions[session_id] = []
        log.info(f"[chat] New session created — total active: {len(sessions)}")

    sessions[session_id].append({"role": "user", "content": user_msg})
    turn = len(sessions[session_id])
    log.info(f"[chat] Turn {turn} — calling Groq (llama-3.1-8b-instant)...")

    try:
        t0       = time.time()
        response = await groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            max_tokens=300,
            messages=[{"role": "system", "content": LUMI_SYSTEM}] + sessions[session_id],
        )
        elapsed       = time.time() - t0
        assistant_msg = response.choices[0].message.content
        is_complete   = "[INTERVIEW_COMPLETE]" in assistant_msg

        sessions[session_id].append({"role": "assistant", "content": assistant_msg})

        log.info(f"[chat] ✅ Groq responded in {elapsed:.2f}s — "
                 f"{len(assistant_msg)} chars | complete={is_complete}")

        return {
            "message":            assistant_msg,
            "interview_complete": is_complete,
        }

    except AuthenticationError:
        log.error("[chat] ❌ Groq AuthenticationError — GROQ_API_KEY is invalid or not set")
        return JSONResponse(status_code=401, content={
            "error": "Invalid or missing GROQ_API_KEY. Set it with: export GROQ_API_KEY=gsk_..."
        })
    except RateLimitError:
        log.warning("[chat] ⚠️  Groq RateLimitError — too many requests")
        return JSONResponse(status_code=429, content={
            "error": "Groq rate limit hit. Wait a moment and try again."
        })
    except APIError as e:
        log.error(f"[chat] ❌ Groq APIError: {e}", exc_info=True)
        return JSONResponse(status_code=502, content={"error": f"Groq API error: {str(e)}"})
    except Exception as e:
        log.error(f"[chat] ❌ Unexpected error: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/analyze")
async def analyze(
    audio:      UploadFile = File(...),
    video:      UploadFile = File(None),   # optional — model only uses audio/text
    transcript: UploadFile = File(None),
):
    sid       = uuid.uuid4().hex[:8]
    raw_audio = tmpfile(sid, "_audio_raw")
    wav_path  = tmpfile(sid, ".wav")
    raw_video = tmpfile(sid, "_video.webm")
    tsv_path  = tmpfile(sid, "_transcript.txt")

    log.info(f"[analyze] Session={sid} — starting full pipeline")
    t_total = time.time()

    try:
        # Save uploads
        with open(raw_audio, "wb") as f: shutil.copyfileobj(audio.file, f)
        audio_size = os.path.getsize(raw_audio)
        log.info(f"[analyze] Audio saved — {audio_size/1024:.1f} KB")

        # Video is optional — save only if provided and non-empty
        if video:
            with open(raw_video, "wb") as f: shutil.copyfileobj(video.file, f)
            video_size = os.path.getsize(raw_video)
            log.info(f"[analyze] Video saved — {video_size/1024:.1f} KB"
                     + (" (empty placeholder — skipping)" if video_size < 100 else ""))
        else:
            log.info("[analyze] No video provided — audio-only mode")

        if transcript:
            with open(tsv_path, "wb") as f: shutil.copyfileobj(transcript.file, f)
            log.info(f"[analyze] Transcript saved — {os.path.getsize(tsv_path)} bytes")

        # Step 1: ffmpeg
        await asyncio.to_thread(to_wav, raw_audio, wav_path)

        # Step 2: Whisper transcription
        transcribed_text = await asyncio.to_thread(_transcribe_wav, wav_path)
        log.info(f"[analyze] Whisper transcript ({len(transcribed_text)} chars): "
                 f"'{transcribed_text[:100]}{'...' if len(transcribed_text)>100 else ''}'")

        # Step 3: BERT + WavLM concurrently
        log.info("[analyze] Running BERT + WavLM concurrently...")
        t0 = time.time()
        text_feat, wavlm_feat = await asyncio.gather(
            asyncio.to_thread(_extract_text_features, transcribed_text),
            asyncio.to_thread(_extract_wavlm_features, wav_path),
        )
        log.info(f"[analyze] BERT + WavLM done in {time.time()-t0:.1f}s")

        # Step 4: Inference
        result = await asyncio.to_thread(
            _run_neurosense_inference, text_feat, wavlm_feat
        )

        # Read back transcript TSV
        transcript_text = ""
        if os.path.exists(tsv_path):
            with open(tsv_path, "r", encoding="utf-8") as f:
                transcript_text = f.read()

        total_time = time.time() - t_total
        log.info(f"[analyze] ✅ Pipeline complete in {total_time:.1f}s — "
                 f"label={result['label']}, confidence={result.get('confidence', 0):.4f}")

        return {**result, "transcript_text": transcript_text}

    except Exception as e:
        log.error(f"[analyze] ❌ Pipeline failed: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"error": str(e)})
    finally:
        for p in (raw_audio, wav_path, raw_video, tsv_path):
            if os.path.exists(p):
                os.remove(p)
                log.debug(f"[analyze] Cleaned up {os.path.basename(p)}")
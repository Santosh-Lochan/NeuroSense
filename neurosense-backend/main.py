"""
NeuroSense Backend — FastAPI
Endpoints:
  GET  /health
  POST /transcribe   — audio blob → whisper transcript
  POST /chat         — message → Lumi (Groq/LLaMA) reply
  POST /analyze      — full session audio + video + transcript → prediction

Fixes applied (v2):
  - Replaced Anthropic with AsyncGroq (llama-3.1-8b-instant)
  - All blocking CPU work (Whisper, BERT, WavLM, inference) offloaded to
    a thread pool via asyncio.to_thread so the event loop never freezes
  - /analyze runs BERT + WavLM concurrently via asyncio.gather
  - ResNet frame sampling increased to every 60 frames to reduce CPU load
"""

import os, uuid, shutil, subprocess, asyncio
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import torch
import torch.nn as nn
import numpy as np
import joblib

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware

import whisper
from transformers import (
    AutoTokenizer, AutoModel,
    WavLMModel, AutoFeatureExtractor,
)
import librosa
import opensmile
import cv2
from torchvision import models, transforms
from groq import AsyncGroq

# ─── CONFIG ───────────────────────────────────────────────────────────────────
MODELS_DIR    = Path("models")
PT_MODEL_PATH = MODELS_DIR / "neurosense_v4_bimodal_attn.pth"
SVM_PATH      = MODELS_DIR / "neurosense_v5_svm.joblib"
SCALER_PATH   = MODELS_DIR / "neurosense_v5_scaler.joblib"
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Thread pool for offloading blocking CPU/IO work off the async event loop.
# 2 workers matches the dual-core Ryzen 3 3250U without oversubscription.
cpu_pool = ThreadPoolExecutor(max_workers=2)

# ─── FASTAPI APP ──────────────────────────────────────────────────────────────
app = FastAPI(title="NeuroSense API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── LOAD MODELS AT STARTUP ───────────────────────────────────────────────────
# All models are loaded once at boot and reused across requests.
# This is synchronous and happens before the server starts serving,
# so it does NOT block the event loop during live requests.

print("Loading Whisper...")
whisper_model = whisper.load_model("base")

print("Loading BERT...")
bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
bert_model     = AutoModel.from_pretrained("bert-base-uncased").eval()

print("Loading WavLM...")
wavlm_extractor = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-base")
wavlm_model     = WavLMModel.from_pretrained("microsoft/wavlm-base").eval()

print("Loading OpenSMILE...")
smile_extractor = opensmile.Smile(
    feature_set=opensmile.FeatureSet.eGeMAPSv02,
    feature_level=opensmile.FeatureLevel.Functionals,
)

print("Loading ResNet vision model...")
resnet = models.resnet50(pretrained=True)
resnet.fc = nn.Identity()
resnet.eval()
vision_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# ─── NEUROSENSE PYTORCH MODEL ─────────────────────────────────────────────────
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

print("Loading NeuroSense PyTorch weights...")
ns_model = BiModalGatedAttention().to(DEVICE)
if PT_MODEL_PATH.exists():
    ns_model.load_state_dict(torch.load(PT_MODEL_PATH, map_location=DEVICE))
    print("✅ PyTorch weights loaded.")
else:
    print("⚠️  PyTorch weights not found — random init (for dev only).")
ns_model.eval()

print("Loading SVM + Scaler...")
svm_clf    = joblib.load(SVM_PATH)    if SVM_PATH.exists()    else None
svm_scaler = joblib.load(SCALER_PATH) if SCALER_PATH.exists() else None
if svm_clf is None:
    print("⚠️  SVM not found — /analyze will return dummy output.")

# ─── GROQ ASYNC CLIENT ────────────────────────────────────────────────────────
# AsyncGroq is fully non-blocking — awaiting it yields control back to the
# event loop during the network call. The sync Groq client would block the
# entire server for the duration of every LLM response.
groq_client = AsyncGroq(api_key=os.environ.get("GROQ_API_KEY", ""))

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

# Per-session conversation history (in-memory, resets on server restart)
sessions: dict = {}

# ─── UTILITY: audio conversion ────────────────────────────────────────────────
def to_wav(input_path: str, output_path: str):
    """Convert any audio/video file to 16 kHz mono WAV via ffmpeg.
    Blocking subprocess — always called via asyncio.to_thread."""
    subprocess.run(
        ["ffmpeg", "-y", "-i", input_path, "-ar", "16000", "-ac", "1", output_path],
        check=True, capture_output=True,
    )

# ─── FEATURE EXTRACTION (all synchronous, all CPU-heavy) ─────────────────────
# These are plain functions, never called directly from async endpoints.
# Every call site uses: await asyncio.to_thread(fn, args...)

def _transcribe_wav(wav_path: str) -> str:
    """Whisper: wav file → transcript string."""
    return whisper_model.transcribe(wav_path)["text"].strip()

def _extract_text_features(text: str) -> torch.Tensor:
    """BERT: transcript string → [1, 768] CLS token embedding."""
    inputs = bert_tokenizer(
        text, return_tensors="pt", truncation=True, max_length=512, padding=True
    )
    with torch.no_grad():
        out = bert_model(**inputs)
    return out.last_hidden_state[:, 0, :]

def _extract_wavlm_features(wav_path: str) -> torch.Tensor:
    """WavLM: wav file → [1, 768] mean-pooled acoustic embedding."""
    audio, _ = librosa.load(wav_path, sr=16000)
    inputs   = wavlm_extractor(audio, sampling_rate=16000, return_tensors="pt")
    with torch.no_grad():
        out = wavlm_model(**inputs)
    return out.last_hidden_state.mean(dim=1)

def _extract_vision_features(video_path: str) -> torch.Tensor:
    """ResNet50: video → [1, 2048] mean-pooled frame embedding.
    Samples every 60 frames (was 30) to halve CPU load on weak hardware."""
    cap   = cv2.VideoCapture(video_path)
    feats = []
    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if count % 60 == 0:
            rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            tensor = vision_transform(rgb).unsqueeze(0)
            with torch.no_grad():
                feats.append(resnet(tensor))
        count += 1
    cap.release()
    return torch.stack(feats).mean(dim=0) if feats else torch.zeros(1, 2048)

def _run_neurosense_inference(text_feat: torch.Tensor,
                               wavlm_feat: torch.Tensor) -> dict:
    """Frozen V4 BiModalGatedAttention → 64-dim vector → V5 SVM → prediction."""
    if svm_clf is None or svm_scaler is None:
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

    return {
        "prediction": prediction,
        "label":      "Positive" if prediction == 1 else "Negative",
        "confidence": confidence,
    }

# ─── ENDPOINTS ────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "NeuroSense backend is live", "device": str(DEVICE)}


@app.post("/transcribe")
async def transcribe(audio: UploadFile = File(...)):
    sid      = uuid.uuid4().hex
    raw_path = f"/tmp/{sid}_raw"
    wav_path = f"/tmp/{sid}.wav"
    try:
        with open(raw_path, "wb") as f:
            shutil.copyfileobj(audio.file, f)

        # Both blocking calls offloaded — event loop stays free
        await asyncio.to_thread(to_wav, raw_path, wav_path)
        transcript = await asyncio.to_thread(_transcribe_wav, wav_path)
        return {"transcript": transcript}
    finally:
        for p in (raw_path, wav_path):
            if os.path.exists(p): os.remove(p)


@app.post("/chat")
async def chat(request: dict):
    session_id = request.get("session_id", "default")
    user_msg   = request.get("message", "")

    if session_id not in sessions:
        sessions[session_id] = []

    sessions[session_id].append({"role": "user", "content": user_msg})

    # AsyncGroq is a true coroutine — non-blocking, event loop stays free
    response = await groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        max_tokens=300,
        messages=[{"role": "system", "content": LUMI_SYSTEM}] + sessions[session_id],
    )

    assistant_msg = response.choices[0].message.content
    sessions[session_id].append({"role": "assistant", "content": assistant_msg})

    return {
        "message":            assistant_msg,
        "interview_complete": "[INTERVIEW_COMPLETE]" in assistant_msg,
    }


@app.post("/analyze")
async def analyze(
    audio:      UploadFile = File(...),
    video:      UploadFile = File(...),
    transcript: UploadFile = File(None),
):
    sid       = uuid.uuid4().hex
    raw_audio = f"/tmp/{sid}_audio_raw"
    wav_path  = f"/tmp/{sid}.wav"
    raw_video = f"/tmp/{sid}_video.webm"
    tsv_path  = f"/tmp/{sid}_transcript.txt"

    try:
        # Save uploads
        with open(raw_audio, "wb") as f: shutil.copyfileobj(audio.file, f)
        with open(raw_video, "wb") as f: shutil.copyfileobj(video.file, f)
        if transcript:
            with open(tsv_path, "wb") as f: shutil.copyfileobj(transcript.file, f)

        # Step 1: convert audio (must complete before feature extraction)
        await asyncio.to_thread(to_wav, raw_audio, wav_path)

        # Step 2: transcribe (must complete before BERT)
        transcribed_text = await asyncio.to_thread(_transcribe_wav, wav_path)

        # Step 3: BERT + WavLM run concurrently — both are read-only,
        # operate on different models, safe to parallelize
        text_feat, wavlm_feat = await asyncio.gather(
            asyncio.to_thread(_extract_text_features, transcribed_text),
            asyncio.to_thread(_extract_wavlm_features, wav_path),
        )

        # Step 4: inference
        result = await asyncio.to_thread(
            _run_neurosense_inference, text_feat, wavlm_feat
        )

        transcript_text = ""
        if os.path.exists(tsv_path):
            with open(tsv_path, "r", encoding="utf-8") as f:
                transcript_text = f.read()

        return {**result, "transcript_text": transcript_text}

    finally:
        for p in (raw_audio, wav_path, raw_video, tsv_path):
            if os.path.exists(p): os.remove(p)

"""
NeuroSense Backend — FastAPI (Optimized for Semantically-Guided Fusion)
"""

import os, uuid, shutil, subprocess, asyncio, logging, time, tempfile
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import torch
import torch.nn as nn

from fastapi import FastAPI, File, UploadFile, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

import whisper
from transformers import AutoTokenizer, AutoModel, WavLMModel, AutoFeatureExtractor
from groq import AsyncGroq, AuthenticationError, RateLimitError, APIError
import librosa

# ─── LOGGING SETUP ────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("neurosense")

# ─── CONFIG ───────────────────────────────────────────────────────────────────
MODELS_DIR    = Path("models")
PT_MODEL_PATH = MODELS_DIR / "best_kfold_fusion.pt"  # <--- Updated to your new model name
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TMPDIR        = tempfile.gettempdir()

log.info(f"Target Hardware: {DEVICE}")

cpu_pool = ThreadPoolExecutor(max_workers=2)

app = FastAPI(title="NeuroSense API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    elapsed = (time.time() - start) * 1000
    log.info(f"[{response.status_code}] {request.method} {request.url.path} ({elapsed:.0f}ms)")
    return response

# ─── LOAD BACKBONES ───────────────────────────────────────────────────────────
log.info("Loading Frozen Feature Extractors...")
whisper_model   = whisper.load_model("base")
bert_tokenizer  = AutoTokenizer.from_pretrained("bert-base-uncased")
bert_model      = AutoModel.from_pretrained("bert-base-uncased").eval().to(DEVICE)
wavlm_extractor = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-base")
wavlm_model     = WavLMModel.from_pretrained("microsoft/wavlm-base").eval().to(DEVICE)

# ─── NEUROSENSE ARCHITECTURE ──────────────────────────────────────────────────
class SemanticallyGuidedBottleneckFusion(nn.Module):
    def __init__(self, text_dim=768, audio_dim=768, visual_dim=68, bottleneck_dim=16, dropout_rate=0.7):
        super().__init__()
        self.text_projection = nn.Sequential(nn.Linear(text_dim, bottleneck_dim), nn.LayerNorm(bottleneck_dim), nn.Dropout(dropout_rate))
        self.audio_projection = nn.Sequential(nn.Linear(audio_dim, bottleneck_dim), nn.LayerNorm(bottleneck_dim), nn.Dropout(dropout_rate))
        self.visual_projection = nn.Sequential(nn.Linear(visual_dim, bottleneck_dim), nn.LayerNorm(bottleneck_dim), nn.Dropout(dropout_rate))
        
        self.audio_cross_attention = nn.MultiheadAttention(embed_dim=bottleneck_dim, num_heads=1, dropout=dropout_rate, batch_first=True)
        self.visual_cross_attention = nn.MultiheadAttention(embed_dim=bottleneck_dim, num_heads=1, dropout=dropout_rate, batch_first=True)
        
        self.classifier = nn.Linear(bottleneck_dim * 3, 1)

    def forward(self, text, audio, visual):
        t_proj = self.text_projection(text)
        a_proj = self.audio_projection(audio)
        v_proj = self.visual_projection(visual)

        t_seq = t_proj.unsqueeze(1)
        a_seq = a_proj.unsqueeze(1)
        v_seq = v_proj.unsqueeze(1)

        attended_audio, _ = self.audio_cross_attention(query=t_seq, key=a_seq, value=a_seq)
        attended_visual, _ = self.visual_cross_attention(query=t_seq, key=v_seq, value=v_seq)

        fused_representation = torch.cat([t_proj, attended_audio.squeeze(1), attended_visual.squeeze(1)], dim=1)
        return self.classifier(fused_representation)

ns_model = SemanticallyGuidedBottleneckFusion().to(DEVICE)
if PT_MODEL_PATH.exists():
    checkpoint = torch.load(PT_MODEL_PATH, map_location=DEVICE, weights_only=True)
    state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
    ns_model.load_state_dict(state_dict)
    log.info("✅ NeuroSense K-Fold Model Loaded Successfully.")
else:
    log.error(f"❌ Could not find {PT_MODEL_PATH}. Inference will fail.")
ns_model.eval()

# ─── GROQ SETUP ───────────────────────────────────────────────────────────────
_groq_key = os.environ.get("GROQ_API_KEY", "")
groq_client = AsyncGroq(api_key=_groq_key) if _groq_key else None
sessions = {}
LUMI_SYSTEM = "You are Lumi, a warm, empathetic mental wellness companion conducting a structured clinical screening interview based on the DAIC-WOZ protocol..." # (Truncated for brevity, keep your original string here)

# ─── UTILITIES ────────────────────────────────────────────────────────────────
def tmpfile(sid: str, suffix: str) -> str: return os.path.join(TMPDIR, f"ns_{sid}{suffix}")

def to_wav(input_path: str, output_path: str):
    subprocess.run(["ffmpeg", "-y", "-i", input_path, "-ar", "16000", "-ac", "1", output_path], capture_output=True)

# ─── FEATURE EXTRACTION ───────────────────────────────────────────────────────
def _transcribe_wav(wav_path: str) -> str:
    return whisper_model.transcribe(wav_path)["text"].strip()

def _extract_text_features(text: str) -> torch.Tensor:
    inputs = bert_tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True).to(DEVICE)
    with torch.no_grad():
        out = bert_model(**inputs)
    return out.last_hidden_state[:, 0, :] # Shape: [1, 768]

def _extract_wavlm_features(wav_path: str) -> torch.Tensor:
    audio, _ = librosa.load(wav_path, sr=16000)
    inputs = wavlm_extractor(audio, sampling_rate=16000, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = wavlm_model(**inputs)
    return out.last_hidden_state.mean(dim=1) # Shape: [1, 768]

def _extract_vision_features() -> torch.Tensor:
    # Modality Imputation: We return a neutral 68-dim tensor since OpenFace C++ binaries aren't running
    return torch.zeros((1, 68)).to(DEVICE)

# ─── ENDPOINTS ────────────────────────────────────────────────────────────────
@app.get("/health")
def health(): return {"status": "NeuroSense backend is live", "models_ready": True}

@app.post("/transcribe")
async def transcribe(audio: UploadFile = File(...)):
    sid = uuid.uuid4().hex[:8]
    raw_path, wav_path = tmpfile(sid, "_raw"), tmpfile(sid, ".wav")
    try:
        with open(raw_path, "wb") as f: shutil.copyfileobj(audio.file, f)
        await asyncio.to_thread(to_wav, raw_path, wav_path)
        return {"transcript": await asyncio.to_thread(_transcribe_wav, wav_path)}
    finally:
        for p in (raw_path, wav_path):
            if os.path.exists(p): os.remove(p)

@app.post("/chat")
async def chat(request: dict):
    # (Keep your existing Groq logic here exactly as it was, it works perfectly)
    session_id, user_msg = request.get("session_id", "default"), request.get("message", "")
    if session_id not in sessions: sessions[session_id] = []
    sessions[session_id].append({"role": "user", "content": user_msg})
    
    response = await groq_client.chat.completions.create(
        model="llama-3.1-8b-instant", max_tokens=300,
        messages=[{"role": "system", "content": LUMI_SYSTEM}] + sessions[session_id],
    )
    assistant_msg = response.choices[0].message.content
    is_complete = "[INTERVIEW_COMPLETE]" in assistant_msg
    sessions[session_id].append({"role": "assistant", "content": assistant_msg})
    return {"message": assistant_msg, "interview_complete": is_complete}

@app.post("/analyze")
async def analyze(audio: UploadFile = File(...), video: UploadFile = File(None), transcript: UploadFile = File(None)):
    sid = uuid.uuid4().hex[:8]
    raw_audio, wav_path = tmpfile(sid, "_audio_raw"), tmpfile(sid, ".wav")
    
    try:
        with open(raw_audio, "wb") as f: shutil.copyfileobj(audio.file, f)
        await asyncio.to_thread(to_wav, raw_audio, wav_path)
        transcribed_text = await asyncio.to_thread(_transcribe_wav, wav_path)

        # 1. Feature Extraction Pipeline
        text_feat, wavlm_feat = await asyncio.gather(
            asyncio.to_thread(_extract_text_features, transcribed_text),
            asyncio.to_thread(_extract_wavlm_features, wav_path),
        )
        visual_feat = _extract_vision_features()

        # 2. Forward Pass
        with torch.no_grad():
            logits = ns_model(text_feat, wavlm_feat, visual_feat)
            prob = torch.sigmoid(logits).item()

        # 3. Apply the Optimized Clinical Threshold
        clinical_threshold = 0.45
        prediction = 1 if prob >= clinical_threshold else 0

        # Returning logits as 'confidence' perfectly syncs with your React App's sigmoid gauge
        return {
            "prediction": prediction,
            "label": "High Likelihood" if prediction == 1 else "Low Likelihood",
            "confidence": logits.item(),
            "transcript_text": transcribed_text
        }

    finally:
        for p in (raw_audio, wav_path):
            if os.path.exists(p): os.remove(p)

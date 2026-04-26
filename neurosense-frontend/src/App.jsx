import { useState, useRef, useEffect, useCallback } from 'react';
import Orb from './Orb';
import UploadPanel from './UploadPanel';
import './App.css';

const BACKEND = 'http://localhost:8000';

// ── helpers ───────────────────────────────────────────────────────────────────
function formatTime(seconds) {
  const m = Math.floor(seconds / 60).toString().padStart(2, '0');
  const s = Math.floor(seconds % 60).toString().padStart(2, '0');
  return `${m}:${s}`;
}

function daicRow(start, stop, speaker, value) {
  return `${start.toFixed(3)}\t${stop.toFixed(3)}\t${speaker}\t${value}`;
}

function confidenceToPercent(rawScore) {
  const sigmoid = 1 / (1 + Math.exp(-rawScore));
  return Math.round(sigmoid * 100);
}

function scoreColor(pct) {
  if (pct >= 65) return '#ff4d6d';
  if (pct >= 40) return '#ffb347';
  return '#39d98a';
}

function scoreLabel(pct) {
  if (pct >= 65) return 'High Likelihood';
  if (pct >= 40) return 'Moderate';
  return 'Low Likelihood';
}

function getSupportedMimeType() {
  const types = [
    'audio/webm;codecs=opus', 'audio/webm',
    'audio/ogg;codecs=opus',  'audio/ogg', 'audio/mp4',
  ];
  for (const type of types) {
    if (MediaRecorder.isTypeSupported(type)) return type;
  }
  return '';
}

// ── ScoreGauge ────────────────────────────────────────────────────────────────
function ScoreGauge({ pct }) {
  const color         = scoreColor(pct);
  const circumference = 2 * Math.PI * 52;
  const offset        = circumference * (1 - pct / 100);
  return (
    <div className="gauge">
      <svg className="gauge__svg" viewBox="0 0 120 120" width="160" height="160">
        <circle cx="60" cy="60" r="52" fill="none"
          stroke="rgba(255,255,255,0.05)" strokeWidth="8" />
        <circle cx="60" cy="60" r="52" fill="none"
          stroke={color} strokeWidth="8" strokeLinecap="round"
          strokeDasharray={circumference} strokeDashoffset={offset}
          transform="rotate(-90 60 60)"
          style={{ transition: 'stroke-dashoffset 1.2s cubic-bezier(0.22,1,0.36,1), stroke 0.6s ease' }}
        />
        <text x="60" y="55" textAnchor="middle"
          fill={color} fontSize="22" fontFamily="IBM Plex Mono" fontWeight="500">
          {pct}%
        </text>
        <text x="60" y="73" textAnchor="middle"
          fill="rgba(255,255,255,0.3)" fontSize="9" fontFamily="IBM Plex Mono" letterSpacing="2">
          LIKELIHOOD
        </text>
      </svg>
      <p className="gauge__label" style={{ color }}>{scoreLabel(pct)}</p>
    </div>
  );
}

// ── App ───────────────────────────────────────────────────────────────────────
export default function App() {
  const [phase, setPhase]           = useState('welcome');
  const [orbState, setOrbState]     = useState('idle');
  const [messages, setMessages]     = useState([]);
  const [result, setResult]         = useState(null);
  const [isHolding, setIsHolding]   = useState(false);
  const [micReady, setMicReady]     = useState(false);
  const [elapsed, setElapsed]       = useState(0);
  const [statusMsg, setStatusMsg]   = useState('');
  const [permError, setPermError]   = useState('');
  const [hasVideo, setHasVideo]     = useState(false);
  const [sessionId]                 = useState(() => crypto.randomUUID());

  // Upload panel visibility
  const [uploadOpen, setUploadOpen] = useState(false);

  const transcriptRows  = useRef([]);
  const sessionStart    = useRef(null);
  const turnStart       = useRef(0);
  const fullAudioChunks = useRef([]);
  const fullVideoChunks = useRef([]);
  const fullAudioRec    = useRef(null);
  const fullVideoRec    = useRef(null);
  const answerChunks    = useRef([]);
  const answerRec       = useRef(null);
  const chatEndRef      = useRef(null);
  const timerRef        = useRef(null);
  const finishRef       = useRef(null);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  useEffect(() => {
    if (phase === 'interview') {
      timerRef.current = setInterval(() => {
        setElapsed(Math.floor((Date.now() - sessionStart.current) / 1000));
      }, 1000);
    }
    return () => clearInterval(timerRef.current);
  }, [phase]);

  const relTime = () => (Date.now() - sessionStart.current) / 1000;

  // ── TTS ───────────────────────────────────────────────────────────────────
  const speak = useCallback((text) => new Promise((resolve) => {
    window.speechSynthesis.cancel();
    const utt = new SpeechSynthesisUtterance(text);
    utt.rate  = 0.90;
    utt.pitch = 1.05;
    const trySetVoice = () => {
      const voices = window.speechSynthesis.getVoices();
      const pref   = voices.find(v => /samantha|karen|moira|zoe|female/i.test(v.name));
      if (pref) utt.voice = pref;
    };
    trySetVoice();
    if (!utt.voice) {
      window.speechSynthesis.addEventListener('voiceschanged', trySetVoice, { once: true });
    }
    const wordCount    = text.split(/\s+/).length;
    const estimatedMs  = Math.max((wordCount / 2.5) * 1000 + 3000, 4000);
    const fallback     = setTimeout(resolve, estimatedMs);
    utt.onend  = () => { clearTimeout(fallback); resolve(); };
    utt.onerror = ()  => { clearTimeout(fallback); resolve(); };
    window.speechSynthesis.speak(utt);
  }), []);

  // ── sendMessage ───────────────────────────────────────────────────────────
  const sendMessage = useCallback(async (userText) => {
    setOrbState('thinking');
    setMicReady(false);

    if (userText !== 'Hello, I am ready to begin.') {
      setMessages(prev => [...prev, { role: 'user', text: userText }]);
    }

    try {
      const res = await fetch(`${BACKEND}/chat`, {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify({ session_id: sessionId, message: userText }),
      });
      if (!res.ok) throw new Error(`Backend returned ${res.status}`);

      const data  = await res.json();
      const clean = data.message.replace('[INTERVIEW_COMPLETE]', '').trim();
      setMessages(prev => [...prev, { role: 'lumi', text: clean }]);

      const lumiStart = relTime();
      setOrbState('speaking');
      await speak(clean);
      const lumiEnd = relTime();
      transcriptRows.current.push(daicRow(lumiStart, lumiEnd, 'Ellie', clean.toLowerCase()));

      if (data.interview_complete) {
        await finishRef.current?.();
      } else {
        setOrbState('idle');
        setMicReady(true);
        setStatusMsg('Hold the button and speak your answer');
      }
    } catch (err) {
      setOrbState('idle');
      setMicReady(false);
      setStatusMsg(err.message.toLowerCase().includes('fetch')
        ? '⚠ Cannot reach backend — make sure it is running on port 8000'
        : `⚠ ${err.message}`);
    }
  }, [sessionId, speak]);

  // ── finishAndAnalyze (live session) ───────────────────────────────────────
  const finishAndAnalyze = useCallback(async () => {
    clearInterval(timerRef.current);
    setPhase('analyzing');
    setOrbState('thinking');

    fullAudioRec.current?.stop();
    fullVideoRec.current?.stop();
    await new Promise(r => setTimeout(r, 800));

    const audioBlob = new Blob(fullAudioChunks.current, { type: 'audio/webm' });
    const videoBlob = fullVideoChunks.current.length > 0
      ? new Blob(fullVideoChunks.current, { type: 'video/webm' })
      : null;

    const tsvText = ['start_time\tstop_time\tspeaker\tvalue', ...transcriptRows.current].join('\n');

    try {
      const form = new FormData();
      form.append('audio', audioBlob, 'session.webm');
      form.append('video', videoBlob ?? new Blob([], { type: 'video/webm' }), 'session.webm');
      form.append('transcript', new Blob([tsvText], { type: 'text/plain' }), 'transcript.txt');

      const res  = await fetch(`${BACKEND}/analyze`, { method: 'POST', body: form });
      if (!res.ok) throw new Error(`Analysis failed: ${res.status}`);

      const data = await res.json();
      setResult({ ...data, tsvText, pct: confidenceToPercent(data.confidence ?? 0) });
    } catch (err) {
      setResult({ prediction: null, label: 'Analysis failed', tsvText, pct: null, error: true, errorMsg: err.message });
    }

    setPhase('result');
    setOrbState('idle');
  }, []);

  useEffect(() => { finishRef.current = finishAndAnalyze; }, [finishAndAnalyze]);

  // ── analyzeUploaded — called from UploadPanel ─────────────────────────────
  const analyzeUploaded = useCallback(async (audioFile, videoFile, transcriptFile) => {
    setUploadOpen(false);
    setPhase('analyzing');
    setOrbState('thinking');

    // Read transcript text so we can display it on result screen
    let tsvText = '';
    if (transcriptFile) {
      tsvText = await transcriptFile.text();
    }

    try {
      const form = new FormData();
      form.append('audio', audioFile, audioFile.name);
      form.append('video',
        videoFile ?? new Blob([], { type: 'video/webm' }),
        videoFile ? videoFile.name : 'empty.webm'
      );
      if (transcriptFile) {
        form.append('transcript', transcriptFile, transcriptFile.name);
      }

      const res  = await fetch(`${BACKEND}/analyze`, { method: 'POST', body: form });
      if (!res.ok) throw new Error(`Analysis failed: ${res.status}`);

      const data = await res.json();
      setResult({ ...data, tsvText, pct: confidenceToPercent(data.confidence ?? 0) });
    } catch (err) {
      setResult({ prediction: null, label: 'Analysis failed', tsvText, pct: null, error: true, errorMsg: err.message });
    }

    setPhase('result');
    setOrbState('idle');
  }, []);

  // ── startSession ──────────────────────────────────────────────────────────
  const startSession = async () => {
    setPermError('');
    setStatusMsg('Requesting microphone…');

    let audioStream;
    try {
      audioStream = await navigator.mediaDevices.getUserMedia({ audio: true });
    } catch (err) {
      setStatusMsg('');
      if (err.name === 'NotAllowedError' || err.name === 'PermissionDeniedError') {
        setPermError('Microphone access was denied. Click the lock icon in your browser address bar, allow microphone, then try again.');
      } else if (err.name === 'NotFoundError') {
        setPermError('No microphone detected on this device.');
      } else {
        setPermError(`Microphone error: ${err.message}`);
      }
      return;
    }

    let videoStream = null;
    try {
      videoStream = await navigator.mediaDevices.getUserMedia({ video: true });
      setHasVideo(true);
    } catch {
      setHasVideo(false);
    }

    const mimeType = getSupportedMimeType();

    const aRec = mimeType
      ? new MediaRecorder(audioStream, { mimeType })
      : new MediaRecorder(audioStream);
    aRec.ondataavailable = e => { if (e.data.size > 0) fullAudioChunks.current.push(e.data); };
    aRec.start(1000);
    fullAudioRec.current = aRec;

    if (videoStream) {
      const vRec = new MediaRecorder(videoStream);
      vRec.ondataavailable = e => { if (e.data.size > 0) fullVideoChunks.current.push(e.data); };
      vRec.start(1000);
      fullVideoRec.current = vRec;
    }

    sessionStart.current = Date.now();
    setStatusMsg('');
    setPhase('interview');
    setTimeout(async () => {
      await sendMessage('Hello, I am ready to begin.');
    }, 0);
  };

  // ── hold-to-speak ─────────────────────────────────────────────────────────
  const handleMicDown = async () => {
    if (!micReady || isHolding) return;
    setIsHolding(true);
    setOrbState('listening');
    setStatusMsg('Listening… release to send');
    answerChunks.current = [];
    turnStart.current    = relTime();

    try {
      const stream   = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mimeType = getSupportedMimeType();
      const recorder = mimeType
        ? new MediaRecorder(stream, { mimeType })
        : new MediaRecorder(stream);
      recorder.ondataavailable = e => { if (e.data.size > 0) answerChunks.current.push(e.data); };
      recorder.start();
      answerRec.current = { recorder, stream };
    } catch (err) {
      setIsHolding(false);
      setOrbState('idle');
      setStatusMsg(err.name === 'NotAllowedError'
        ? '⚠ Microphone permission revoked — please refresh'
        : '⚠ Could not access microphone');
    }
  };

  const handleMicUp = async () => {
    if (!isHolding || !answerRec.current) return;
    setIsHolding(false);
    setOrbState('thinking');
    setStatusMsg('Transcribing…');

    const { recorder, stream } = answerRec.current;
    recorder.stop();
    stream.getTracks().forEach(t => t.stop());
    await new Promise(r => recorder.onstop = r);

    const blob      = new Blob(answerChunks.current, { type: 'audio/webm' });
    const answerEnd = relTime();

    if (blob.size < 1000) {
      setOrbState('idle');
      setMicReady(true);
      setStatusMsg('Nothing was recorded — try holding longer');
      return;
    }

    let transcript = '';
    try {
      const form = new FormData();
      form.append('audio', blob, 'answer.webm');
      const res  = await fetch(`${BACKEND}/transcribe`, { method: 'POST', body: form });
      if (!res.ok) throw new Error(`Transcribe failed: ${res.status}`);
      const data = await res.json();
      transcript = data.transcript?.trim() || '';
    } catch {
      transcript = '[inaudible]';
    }

    if (!transcript || transcript === '[inaudible]') {
      setOrbState('idle');
      setMicReady(true);
      setStatusMsg('Could not hear you clearly — please try again');
      return;
    }

    transcriptRows.current.push(
      daicRow(turnStart.current, answerEnd, 'Participant', transcript.toLowerCase())
    );
    setStatusMsg('');
    await sendMessage(transcript);
  };

  // ── download transcript ───────────────────────────────────────────────────
  const downloadTranscript = () => {
    if (!result?.tsvText) return;
    const blob = new Blob([result.tsvText], { type: 'text/plain' });
    const url  = URL.createObjectURL(blob);
    const a    = document.createElement('a');
    a.href     = url;
    a.download = `neurosense_${sessionId.slice(0, 8)}.txt`;
    a.click();
    URL.revokeObjectURL(url);
  };

  // ── render ────────────────────────────────────────────────────────────────
  return (
    <div className="app">
      <div className="app__bg" aria-hidden>
        <div className="app__bg-glow" />
        <div className="app__bg-grid" />
      </div>

      <div className="app__inner">

        {/* ── WELCOME ── */}
        {phase === 'welcome' && (
          <div className="screen screen--welcome">
            <div className="welcome__orb">
              <Orb state="idle" size={180} />
            </div>
            <div className="welcome__copy">
              <p className="welcome__eyebrow">NEUROSENSE / v5.0</p>
              <h1 className="welcome__title">Meet Lumi.</h1>
              <p className="welcome__sub">
                Lumi will guide you through a short mental wellness
                check-in — nine questions, roughly five minutes.
                Find a <em>quiet space</em> before starting.
              </p>

              <div className="welcome__pills">
                <span className="pill pill--req">🎙 Microphone required</span>
                <span className="pill pill--opt">📷 Camera optional</span>
              </div>

              <div className="welcome__notice">
                <span className="notice__icon">⬡</span>
                <span>
                  Audio is processed locally and discarded immediately
                  after analysis. Nothing is stored or transmitted.
                </span>
              </div>

              {permError && (
                <div className="perm-error">
                  <span className="perm-error__icon">⚠</span>
                  <span>{permError}</span>
                </div>
              )}

              {statusMsg && !permError && (
                <p className="status-checking">{statusMsg}</p>
              )}

              <button className="btn btn--start" onClick={startSession}>
                Begin Session
              </button>
            </div>
          </div>
        )}

        {/* ── INTERVIEW ── */}
        {phase === 'interview' && (
          <div className="screen screen--interview">
            <div className="interview__bar">
              <span className="interview__bar-label">SESSION</span>
              <span className="interview__bar-timer">{formatTime(elapsed)}</span>
              <div className="interview__bar-right">
                {hasVideo
                  ? <span className="bar-badge bar-badge--on">📷 VIDEO ON</span>
                  : <span className="bar-badge bar-badge--off">📷 AUDIO ONLY</span>}
              </div>
            </div>

            <div className="interview__body">
              <div className="interview__orb-col">
                <Orb state={orbState} size={130} />
              </div>
              <div className="interview__chat-col">
                <div className="chat">
                  {messages.length === 0 && (
                    <p className="chat__empty">Lumi is connecting…</p>
                  )}
                  {messages.map((m, i) => (
                    <div key={i} className={`bubble bubble--${m.role}`}>
                      <span className="bubble__who">{m.role === 'lumi' ? 'LUMI' : 'YOU'}</span>
                      <p className="bubble__text">{m.text}</p>
                    </div>
                  ))}
                  <div ref={chatEndRef} />
                </div>
              </div>
            </div>

            <div className="interview__controls">
              {statusMsg && (
                <p className={`status-hint ${statusMsg.startsWith('⚠') ? 'status-hint--warn' : ''}`}>
                  {statusMsg}
                </p>
              )}
              <button
                className={`mic-btn ${isHolding ? 'mic-btn--active' : ''} ${!micReady ? 'mic-btn--disabled' : ''}`}
                onPointerDown={handleMicDown}
                onPointerUp={handleMicUp}
                onPointerLeave={isHolding ? handleMicUp : undefined}
                disabled={!micReady && !isHolding}
              >
                <span className="mic-btn__icon">{isHolding ? '▪' : '●'}</span>
                <span className="mic-btn__label">
                  {isHolding ? 'Release to send' : micReady ? 'Hold to speak' : 'Lumi is speaking…'}
                </span>
              </button>
            </div>
          </div>
        )}

        {/* ── ANALYZING ── */}
        {phase === 'analyzing' && (
          <div className="screen screen--analyzing">
            <Orb state="thinking" size={150} />
            <div className="analyzing__copy">
              <h2 className="analyzing__title">Analyzing session</h2>
              <p className="analyzing__sub">Running Whisper · BERT · WavLM · NeuroSense V4 · SVM</p>
              <p className="analyzing__note">This may take a few minutes on CPU hardware.</p>
            </div>
          </div>
        )}

        {/* ── RESULT ── */}
        {phase === 'result' && result && (
          <div className="screen screen--result">
            <Orb state="idle" size={90} />
            <div className="result">
              <p className="result__eyebrow">ANALYSIS COMPLETE</p>
              <h2 className="result__title">Session Result</h2>

              {!result.error ? (
                <>
                  <ScoreGauge pct={result.pct} />
                  <div className={`result__badge result__badge--${result.prediction === 1 ? 'pos' : 'neg'}`}>
                    <span className="result__badge-icon">{result.prediction === 1 ? '◆' : '◇'}</span>
                    <span>{result.label}</span>
                  </div>
                  <p className="result__score-note">
                    Score reflects estimated likelihood of clinically significant
                    depressive symptoms based on audio and speech patterns (PHQ-10 threshold ≥ 10).
                  </p>
                </>
              ) : (
                <div className="result__error">
                  <span className="result__error-icon">⚠</span>
                  <p>{result.label}</p>
                  {result.errorMsg && <p className="result__error-detail">{result.errorMsg}</p>}
                  <p className="result__error-tip">The transcript was still saved — download it below.</p>
                </div>
              )}

              <p className="result__disclaimer">
                Research prototype only. Not a medical diagnosis.
                If you have concerns, please speak with a qualified professional.
              </p>

              <div className="result__actions">
                <button className="btn btn--dl" onClick={downloadTranscript}>↓ Download Transcript</button>
                <button className="btn btn--ghost" onClick={() => window.location.reload()}>New Session</button>
              </div>

              {result.tsvText && (
                <details className="result__preview">
                  <summary>Preview transcript (DAIC-WOZ format)</summary>
                  <pre>{result.tsvText}</pre>
                </details>
              )}
            </div>
          </div>
        )}

      </div>

      {/* ── FLOATING UPLOAD BUTTON — visible on all phases except analyzing ── */}
      {phase !== 'analyzing' && (
        <UploadPanel
          open={uploadOpen}
          onToggle={() => setUploadOpen(o => !o)}
          onAnalyze={analyzeUploaded}
        />
      )}
    </div>
  );
}
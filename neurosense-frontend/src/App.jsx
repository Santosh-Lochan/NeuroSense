import { useState, useRef, useEffect, useCallback } from 'react';
import Lumi from './components/Lumi';
import './App.css';

const BACKEND = 'http://localhost:8000';

// ─── DAIC-WOZ transcript helpers ─────────────────────────────────────────────
function fmtRow(start, stop, speaker, value) {
  return `${start.toFixed(3)}\t${stop.toFixed(3)}\t${speaker}\t${value}`;
}

// ─── APP ─────────────────────────────────────────────────────────────────────
export default function App() {
  const [phase, setPhase]           = useState('welcome'); // welcome | interview | analyzing | result
  const [lumiState, setLumiState]   = useState('idle');
  const [messages, setMessages]     = useState([]);
  const [hint, setHint]             = useState('');
  const [isRecording, setIsRecording] = useState(false);
  const [result, setResult]         = useState(null);
  const [sessionId]                 = useState(() => crypto.randomUUID());

  // Transcript
  const transcriptRows = useRef([]);
  const sessionStart   = useRef(null);
  const turnStart      = useRef(0);

  // Full-session recording
  const fullAudioChunks  = useRef([]);
  const fullVideoChunks  = useRef([]);
  const fullAudioRec     = useRef(null);
  const fullVideoRec     = useRef(null);
  const mainStream       = useRef(null);

  // Per-answer recording
  const answerRec        = useRef(null);
  const answerChunks     = useRef([]);

  const chatEndRef = useRef(null);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const relT = () => (Date.now() - sessionStart.current) / 1000;

  const pushRow = (speaker, text, start, end) => {
    transcriptRows.current.push(fmtRow(start, end, speaker, text));
  };

  // ── TTS ────────────────────────────────────────────────────────────────────
  const speak = useCallback((text) => new Promise((res) => {
    window.speechSynthesis.cancel();
    const u = new SpeechSynthesisUtterance(text);
    u.rate = 0.9; u.pitch = 1.05;
    u.onend = res; u.onerror = res;
    window.speechSynthesis.speak(u);
  }), []);

  // ── Start interview ────────────────────────────────────────────────────────
  const startInterview = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true, video: true });
      mainStream.current = stream;

      const aRec = new MediaRecorder(stream, { mimeType: 'audio/webm' });
      aRec.ondataavailable = e => fullAudioChunks.current.push(e.data);
      aRec.start(1000);
      fullAudioRec.current = aRec;

      const vRec = new MediaRecorder(stream);
      vRec.ondataavailable = e => fullVideoChunks.current.push(e.data);
      vRec.start(1000);
      fullVideoRec.current = vRec;

      sessionStart.current = Date.now();
      setPhase('interview');
      await lumiSay('Hello, I am ready to begin.');
    } catch {
      setHint('⚠ Camera or microphone access was denied. Please allow both and try again.');
    }
  };

  // ── Lumi speaks a turn ─────────────────────────────────────────────────────
  const lumiSay = useCallback(async (userText) => {
    setLumiState('thinking');

    if (userText !== 'Hello, I am ready to begin.') {
      setMessages(p => [...p, { role: 'user', text: userText }]);
    }

    let lumiText = '';
    let complete = false;

    try {
      const res  = await fetch(`${BACKEND}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: sessionId, message: userText }),
      });
      const data = await res.json();
      complete  = data.interview_complete;
      lumiText  = data.message.replace('[INTERVIEW_COMPLETE]', '').trim();
    } catch {
      lumiText = 'Sorry, I could not reach the server. Please check that the backend is running.';
    }

    setMessages(p => [...p, { role: 'lumi', text: lumiText }]);

    const ts = relT();
    setLumiState('speaking');
    await speak(lumiText);
    const te = relT();
    pushRow('Ellie', lumiText.toLowerCase(), ts, te);

    if (complete) {
      await finishAndAnalyze();
    } else {
      setLumiState('idle');
      setHint('Hold the button and speak — release when done');
    }
  }, [sessionId, speak]);

  // ── Answer recording ───────────────────────────────────────────────────────
  const startAnswer = async () => {
    if (isRecording) return;
    setIsRecording(true);
    setLumiState('listening');
    setHint('Listening…');
    answerChunks.current = [];
    turnStart.current    = relT();

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const rec    = new MediaRecorder(stream, { mimeType: 'audio/webm' });
      rec.ondataavailable = e => answerChunks.current.push(e.data);
      rec.start();
      answerRec.current = { rec, stream };
    } catch {
      setIsRecording(false);
      setHint('⚠ Microphone access failed.');
    }
  };

  const stopAnswer = async () => {
    if (!isRecording || !answerRec.current) return;
    setIsRecording(false);
    setLumiState('thinking');
    setHint('Transcribing…');

    const { rec, stream } = answerRec.current;
    rec.stop();
    stream.getTracks().forEach(t => t.stop());
    await new Promise(r => rec.onstop = r);

    const blob = new Blob(answerChunks.current, { type: 'audio/webm' });
    const end  = relT();

    let text = '[inaudible]';
    try {
      const form = new FormData();
      form.append('audio', blob, 'answer.webm');
      const res  = await fetch(`${BACKEND}/transcribe`, { method: 'POST', body: form });
      const data = await res.json();
      text = data.transcript || '[inaudible]';
    } catch { /* keep fallback */ }

    pushRow('Participant', text.toLowerCase(), turnStart.current, end);
    setHint('');
    await lumiSay(text);
  };

  // ── Finish & analyze ───────────────────────────────────────────────────────
  const finishAndAnalyze = async () => {
    setPhase('analyzing');
    setLumiState('thinking');

    fullAudioRec.current?.stop();
    fullVideoRec.current?.stop();
    mainStream.current?.getTracks().forEach(t => t.stop());
    await new Promise(r => setTimeout(r, 800));

    const audioBlob = new Blob(fullAudioChunks.current, { type: 'audio/webm' });
    const videoBlob = new Blob(fullVideoChunks.current, { type: 'video/webm' });
    const header    = 'start_time\tstop_time\tspeaker\tvalue';
    const tsText    = [header, ...transcriptRows.current].join('\n');

    let res = null;
    try {
      const form = new FormData();
      form.append('audio',      audioBlob, 'session.webm');
      form.append('video',      videoBlob, 'session.webm');
      form.append('transcript', new Blob([tsText], { type: 'text/plain' }), 'transcript.txt');
      const r    = await fetch(`${BACKEND}/analyze`, { method: 'POST', body: form });
      res        = await r.json();
    } catch {
      res = { prediction: null, label: 'Backend unavailable', error: true };
    }

    setResult({ ...res, transcriptText: tsText });
    setPhase('result');
    setLumiState('idle');
  };

  // ── Download transcript ────────────────────────────────────────────────────
  const downloadTranscript = () => {
    if (!result?.transcriptText) return;
    const blob = new Blob([result.transcriptText], { type: 'text/plain' });
    const url  = URL.createObjectURL(blob);
    const a    = document.createElement('a');
    a.href = url;
    a.download = `neurosense_${sessionId.slice(0, 8)}.txt`;
    a.click();
    URL.revokeObjectURL(url);
  };

  // ── Render ─────────────────────────────────────────────────────────────────
  return (
    <div className="app">
      <div className="app__noise" aria-hidden />

      {/* ── WELCOME ── */}
      {phase === 'welcome' && (
        <div className="screen screen--welcome">
          <div className="welcome__top">
            <Lumi state="idle" size={110} />
            <div className="welcome__brand">
              <h1 className="brand__name">NeuroSense</h1>
              <span className="brand__version">v5.0 — Research Prototype</span>
            </div>
          </div>

          <div className="welcome__body">
            <p className="welcome__intro">
              Lumi will guide you through a structured mental wellness
              check-in based on the DAIC-WOZ clinical protocol.
            </p>

            <div className="welcome__specs">
              <div className="spec">
                <span className="spec__key">DURATION</span>
                <span className="spec__val">~5 minutes</span>
              </div>
              <div className="spec">
                <span className="spec__key">MODALITIES</span>
                <span className="spec__val">Audio · Video · Text</span>
              </div>
              <div className="spec">
                <span className="spec__key">PRIVACY</span>
                <span className="spec__val">Local processing only</span>
              </div>
            </div>

            <p className="welcome__disclaimer">
              This tool is a research prototype and does not constitute
              a medical diagnosis. Please ensure you are in a quiet,
              well-lit environment before beginning.
            </p>

            <button className="btn btn--start" onClick={startInterview}>
              <span className="btn__text">Begin Session</span>
              <span className="btn__icon">→</span>
            </button>
          </div>
        </div>
      )}

      {/* ── INTERVIEW ── */}
      {phase === 'interview' && (
        <div className="screen screen--interview">
          <div className="interview__header">
            <Lumi state={lumiState} size={80} />
            <div className="interview__meta">
              <span className="interview__title">Session in Progress</span>
              <span className="interview__sub">DAIC-WOZ Protocol</span>
            </div>
          </div>

          <div className="chat">
            {messages.length === 0 && (
              <p className="chat__empty">Initialising interview protocol…</p>
            )}
            {messages.map((m, i) => (
              <div key={i} className={`message message--${m.role}`}>
                <span className="message__speaker">
                  {m.role === 'lumi' ? 'LUMI' : 'YOU'}
                </span>
                <p className="message__text">{m.text}</p>
              </div>
            ))}
            <div ref={chatEndRef} />
          </div>

          {hint && <p className="hint">{hint}</p>}

          <button
            className={`btn btn--mic ${isRecording ? 'btn--mic-active' : ''}`}
            onPointerDown={startAnswer}
            onPointerUp={stopAnswer}
            onPointerLeave={isRecording ? stopAnswer : undefined}
            disabled={lumiState === 'thinking' || lumiState === 'speaking'}
          >
            <span className="btn__icon">{isRecording ? '■' : '●'}</span>
            <span className="btn__text">
              {isRecording ? 'Release to send' : 'Hold to speak'}
            </span>
          </button>
        </div>
      )}

      {/* ── ANALYZING ── */}
      {phase === 'analyzing' && (
        <div className="screen screen--analyzing">
          <Lumi state="thinking" size={110} />
          <div className="analyzing__copy">
            <p className="analyzing__title">Analysing Session</p>
            <p className="analyzing__sub">
              Running Whisper · BERT · WavLM · NeuroSense V4 · SVM
            </p>
            <div className="analyzing__bar">
              <div className="analyzing__fill" />
            </div>
            <p className="analyzing__note">
              This may take 2–4 minutes on CPU hardware.
            </p>
          </div>
        </div>
      )}

      {/* ── RESULT ── */}
      {phase === 'result' && result && (
        <div className="screen screen--result">
          <Lumi state="idle" size={80} />

          <div className="result">
            <div className="result__header">
              <span className="result__label">Session Complete</span>
              <span className={`result__badge result__badge--${result.prediction === 1 ? 'pos' : result.prediction === 0 ? 'neg' : 'warn'}`}>
                {result.label}
              </span>
            </div>

            {typeof result.confidence === 'number' && (
              <div className="result__row">
                <span className="result__key">SVM DECISION SCORE</span>
                <span className="result__val">{result.confidence.toFixed(4)}</span>
              </div>
            )}

            <p className="result__disclaimer">
              This result is generated by a research prototype (F1 ≈ 52.8%, n=142).
              It is <em>not</em> a clinical diagnosis. If you have concerns about
              your mental health, please speak with a qualified professional.
            </p>

            <div className="result__actions">
              <button className="btn btn--download" onClick={downloadTranscript}>
                <span className="btn__icon">↓</span>
                <span className="btn__text">Download Transcript (.txt)</span>
              </button>
              <button className="btn btn--ghost" onClick={() => window.location.reload()}>
                New Session
              </button>
            </div>

            {result.transcriptText && (
              <details className="transcript">
                <summary className="transcript__summary">
                  Preview · DAIC-WOZ transcript
                </summary>
                <pre className="transcript__body">{result.transcriptText}</pre>
              </details>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

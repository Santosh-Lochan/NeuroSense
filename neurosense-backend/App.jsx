import { useState, useRef, useEffect, useCallback } from "react";

// ─── CONSTANTS ────────────────────────────────────────────────────────────────
const BACKEND = "http://localhost:8000";

const MASCOT_STATES = {
  idle:      { primary: "#38bdf8", secondary: "#0c1445", glow: "#0ea5e9", label: "LUMI · READY" },
  listening: { primary: "#f87171", secondary: "#2d0a0a", glow: "#ef4444", label: "LUMI · LISTENING" },
  thinking:  { primary: "#c084fc", secondary: "#1e0a2e", glow: "#a855f7", label: "LUMI · THINKING" },
  speaking:  { primary: "#f0f9ff", secondary: "#0c2a4a", glow: "#7dd3fc", label: "LUMI · SPEAKING" },
};

// ─── MASCOT COMPONENT ─────────────────────────────────────────────────────────
function Mascot({ state = "idle", size = 140 }) {
  const cfg = MASCOT_STATES[state];
  return (
    <div className={`mascot mascot--${state}`} style={{ "--size": `${size}px` }}>
      <div className="mascot__orb" style={{
        "--primary": cfg.primary,
        "--secondary": cfg.secondary,
        "--glow": cfg.glow,
      }}>
        <div className="mascot__highlight" />
        <div className="mascot__highlight mascot__highlight--2" />
        {state === "listening" && <div className="mascot__ring" />}
        {state === "thinking" && (
          <>
            <div className="mascot__particle" style={{ "--delay": "0s" }} />
            <div className="mascot__particle" style={{ "--delay": "0.4s" }} />
            <div className="mascot__particle" style={{ "--delay": "0.8s" }} />
          </>
        )}
      </div>
      <span className="mascot__label">{cfg.label}</span>
    </div>
  );
}

// ─── TRANSCRIPT ROW HELPER ────────────────────────────────────────────────────
// Mirrors DAIC-WOZ format: start_time stop_time speaker value
// e.g. "36.588\t39.868\tEllie\ti'm ellie thanks for coming in today"
function formatTranscriptRow(startTime, stopTime, speaker, value) {
  return `${startTime.toFixed(3)}\t${stopTime.toFixed(3)}\t${speaker}\t${value}`;
}

// ─── MAIN APP ─────────────────────────────────────────────────────────────────
export default function App() {
  const [phase, setPhase]               = useState("welcome");   // welcome | interview | analyzing | result
  const [mascotState, setMascotState]   = useState("idle");
  const [messages, setMessages]         = useState([]);
  const [result, setResult]             = useState(null);
  const [isRecordingAnswer, setIsRecordingAnswer] = useState(false);
  const [sessionId]                     = useState(() => crypto.randomUUID());
  const [statusText, setStatusText]     = useState("");

  // Transcript state (DAIC-WOZ rows)
  const transcriptRowsRef  = useRef([]);   // array of formatted strings
  const sessionStartRef    = useRef(null); // Date.now() when interview began
  const turnStartRef       = useRef(null); // when current speaker started

  // Recording refs
  const fullAudioChunks    = useRef([]);
  const fullVideoChunks    = useRef([]);
  const fullAudioRecorder  = useRef(null);
  const fullVideoRecorder  = useRef(null);
  const mainStreamRef      = useRef(null);

  const answerRecorderRef  = useRef(null);
  const answerChunksRef    = useRef([]);

  const messagesEndRef     = useRef(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // ── helpers ─────────────────────────────────────────────────────────────────
  const relativeTime = () => (Date.now() - sessionStartRef.current) / 1000;

  const pushTranscriptRow = (speaker, text, start, end) => {
    transcriptRowsRef.current.push(formatTranscriptRow(start, end, speaker, text));
  };

  const speakText = useCallback((text) => new Promise((resolve) => {
    window.speechSynthesis.cancel();
    const utt = new SpeechSynthesisUtterance(text);
    utt.rate = 0.92; utt.pitch = 1.1;
    utt.onend = resolve;
    utt.onerror = resolve;
    window.speechSynthesis.speak(utt);
  }), []);

  // ── start interview ──────────────────────────────────────────────────────────
  const startInterview = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true, video: true });
      mainStreamRef.current = stream;

      // Full-session audio recorder
      const aRec = new MediaRecorder(stream, { mimeType: "audio/webm" });
      aRec.ondataavailable = e => fullAudioChunks.current.push(e.data);
      aRec.start(1000);
      fullAudioRecorder.current = aRec;

      // Full-session video recorder
      const vRec = new MediaRecorder(stream);
      vRec.ondataavailable = e => fullVideoChunks.current.push(e.data);
      vRec.start(1000);
      fullVideoRecorder.current = vRec;

      sessionStartRef.current = Date.now();
      setPhase("interview");
      await lumiSay("Hello, I'm ready to begin.");
    } catch (err) {
      setStatusText("⚠️ Camera/mic permission denied. Please allow access and try again.");
    }
  };

  // ── Lumi says something (API → TTS → transcript) ─────────────────────────────
  const lumiSay = useCallback(async (userText) => {
    setMascotState("thinking");

    // Add user message to chat (unless it's the bootstrap message)
    if (userText !== "Hello, I'm ready to begin.") {
      setMessages(prev => [...prev, { role: "user", text: userText }]);
    }

    try {
      const res  = await fetch(`${BACKEND}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: sessionId, message: userText }),
      });
      const data = await res.json();
      const rawMessage = data.message;
      const lumiText   = rawMessage.replace("[INTERVIEW_COMPLETE]", "").trim();

      setMessages(prev => [...prev, { role: "lumi", text: lumiText }]);

      // Record Lumi's turn in transcript
      const lumiStart = relativeTime();
      setMascotState("speaking");
      await speakText(lumiText);
      const lumiEnd = relativeTime();
      pushTranscriptRow("Ellie", lumiText.toLowerCase(), lumiStart, lumiEnd);

      if (data.interview_complete) {
        await finishAndAnalyze();
      } else {
        setMascotState("listening");
        setStatusText("Hold the button and speak your answer");
      }
    } catch {
      setMascotState("idle");
      setStatusText("⚠️ Could not reach backend. Is it running on :8000?");
    }
  }, [sessionId, speakText]);

  // ── answer recording ─────────────────────────────────────────────────────────
  const startAnswer = async () => {
    if (isRecordingAnswer) return;
    setIsRecordingAnswer(true);
    setMascotState("listening");
    setStatusText("Recording… release when done");
    answerChunksRef.current = [];
    turnStartRef.current = relativeTime();

    try {
      const stream  = await navigator.mediaDevices.getUserMedia({ audio: true });
      const recorder = new MediaRecorder(stream, { mimeType: "audio/webm" });
      recorder.ondataavailable = e => answerChunksRef.current.push(e.data);
      recorder.start();
      answerRecorderRef.current = { recorder, stream };
    } catch {
      setIsRecordingAnswer(false);
      setStatusText("⚠️ Mic access failed.");
    }
  };

  const stopAnswer = async () => {
    if (!isRecordingAnswer) return;
    setIsRecordingAnswer(false);
    setMascotState("thinking");
    setStatusText("Transcribing…");

    const { recorder, stream } = answerRecorderRef.current;
    recorder.stop();
    stream.getTracks().forEach(t => t.stop());
    await new Promise(r => recorder.onstop = r);

    const blob = new Blob(answerChunksRef.current, { type: "audio/webm" });
    const answerEnd = relativeTime();

    // Transcribe
    let transcript = "";
    try {
      transcript = await transcribeAudio(blob);
    } catch {
      transcript = "[inaudible]";
    }

    // Push participant row to transcript
    pushTranscriptRow("Participant", transcript.toLowerCase(), turnStartRef.current, answerEnd);
    setStatusText("");
    await lumiSay(transcript);
  };

  // ── transcription ────────────────────────────────────────────────────────────
  const transcribeAudio = async (audioBlob) => {
    const form = new FormData();
    form.append("audio", audioBlob, "answer.webm");
    const res  = await fetch(`${BACKEND}/transcribe`, { method: "POST", body: form });
    const data = await res.json();
    return data.transcript || "";
  };

  // ── finish & analyze ─────────────────────────────────────────────────────────
  const finishAndAnalyze = async () => {
    setPhase("analyzing");
    setMascotState("thinking");
    setStatusText("Processing your session…");

    // Stop full-session recorders
    fullAudioRecorder.current?.stop();
    fullVideoRecorder.current?.stop();
    mainStreamRef.current?.getTracks().forEach(t => t.stop());

    await new Promise(r => setTimeout(r, 1000)); // let recorders flush

    const audioBlob = new Blob(fullAudioChunks.current, { type: "audio/webm" });
    const videoBlob = new Blob(fullVideoChunks.current, { type: "video/webm" });

    // Build DAIC-WOZ transcript text
    const transcriptHeader = "start_time\tstop_time\tspeaker\tvalue";
    const transcriptText   = [transcriptHeader, ...transcriptRowsRef.current].join("\n");

    try {
      const form = new FormData();
      form.append("audio",      audioBlob,    "session.webm");
      form.append("video",      videoBlob,    "session.webm");
      form.append("transcript", new Blob([transcriptText], { type: "text/plain" }), "transcript.txt");

      const res  = await fetch(`${BACKEND}/analyze`, { method: "POST", body: form });
      const data = await res.json();

      setResult({ ...data, transcriptText });
    } catch {
      // Even if backend fails, we still have the transcript
      setResult({ prediction: null, label: "Backend unavailable", transcriptText, error: true });
    }

    setPhase("result");
    setMascotState("idle");
  };

  // ── download transcript ───────────────────────────────────────────────────────
  const downloadTranscript = () => {
    if (!result?.transcriptText) return;
    const blob = new Blob([result.transcriptText], { type: "text/plain" });
    const url  = URL.createObjectURL(blob);
    const a    = document.createElement("a");
    a.href = url; a.download = `neurosense_transcript_${sessionId.slice(0,8)}.txt`;
    a.click();
    URL.revokeObjectURL(url);
  };

  // ── render ────────────────────────────────────────────────────────────────────
  return (
    <>
      <div className="app">
        {/* Background grid */}
        <div className="bg-grid" aria-hidden />

        <div className="shell">

          {/* ── WELCOME ── */}
          {phase === "welcome" && (
            <div className="screen screen--welcome">
              <Mascot state="idle" size={160} />
              <div className="welcome-copy">
                <h1 className="logo">NeuroSense</h1>
                <p className="tagline">
                  Lumi will guide you through a short mental wellness check-in.<br />
                  Please be in a <strong>quiet, well-lit space</strong> — this takes about 5 minutes.
                </p>
                <p className="disclaimer">
                  🔒 All data stays on your device and is never stored after analysis.
                </p>
                <button className="btn btn--primary" onClick={startInterview}>
                  Begin Session
                </button>
              </div>
            </div>
          )}

          {/* ── INTERVIEW ── */}
          {phase === "interview" && (
            <div className="screen screen--interview">
              <Mascot state={mascotState} size={100} />

              <div className="chat-window">
                {messages.length === 0 && (
                  <div className="chat-empty">Lumi is warming up…</div>
                )}
                {messages.map((m, i) => (
                  <div key={i} className={`bubble bubble--${m.role}`}>
                    <span className="bubble__speaker">{m.role === "lumi" ? "Lumi" : "You"}</span>
                    <p className="bubble__text">{m.text}</p>
                  </div>
                ))}
                <div ref={messagesEndRef} />
              </div>

              {statusText && <p className="status-text">{statusText}</p>}

              <button
                className={`btn btn--mic ${isRecordingAnswer ? "btn--mic-active" : ""}`}
                onPointerDown={startAnswer}
                onPointerUp={stopAnswer}
                onPointerLeave={isRecordingAnswer ? stopAnswer : undefined}
                disabled={mascotState === "thinking" || mascotState === "speaking"}
              >
                <span className="mic-icon">{isRecordingAnswer ? "⏹" : "🎙"}</span>
                {isRecordingAnswer ? "Release to send" : "Hold to speak"}
              </button>
            </div>
          )}

          {/* ── ANALYZING ── */}
          {phase === "analyzing" && (
            <div className="screen screen--analyzing">
              <Mascot state="thinking" size={120} />
              <p className="analyzing-text">Analyzing your session…</p>
              <p className="analyzing-sub">{statusText}</p>
            </div>
          )}

          {/* ── RESULT ── */}
          {phase === "result" && result && (
            <div className="screen screen--result">
              <Mascot state="idle" size={100} />
              <div className="result-card">
                <h2 className="result-title">Session Complete</h2>

                {!result.error ? (
                  <div className={`result-badge result-badge--${result.prediction === 1 ? "positive" : "negative"}`}>
                    {result.label}
                  </div>
                ) : (
                  <div className="result-badge result-badge--warn">Backend unavailable</div>
                )}

                <p className="result-note">
                  This is a research prototype and does not constitute a medical diagnosis.
                  If you have concerns, please speak with a qualified mental health professional.
                </p>

                <div className="result-actions">
                  <button className="btn btn--secondary" onClick={downloadTranscript}>
                    ⬇ Download Transcript (.txt)
                  </button>
                  <button className="btn btn--ghost" onClick={() => window.location.reload()}>
                    Start New Session
                  </button>
                </div>

                {result.transcriptText && (
                  <details className="transcript-preview">
                    <summary>Preview Transcript (DAIC-WOZ format)</summary>
                    <pre>{result.transcriptText}</pre>
                  </details>
                )}
              </div>
            </div>
          )}

        </div>
      </div>
    </>
  );
}

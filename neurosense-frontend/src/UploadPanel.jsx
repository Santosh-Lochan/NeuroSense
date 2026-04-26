import { useState, useRef } from 'react';
import './UploadPanel.css';

/*
  Slot status: 'idle' | 'uploading' | 'success' | 'error'

  Props:
    open       — boolean, panel visible
    onToggle   — () => void
    onAnalyze  — (audioFile, videoFile, transcriptFile) => void
*/

const SLOTS = [
  {
    id:       'audio',
    label:    'Voice Recording',
    hint:     '.wav · .mp3 · .webm · .ogg · .m4a',
    accept:   'audio/*,.webm',
    required: true,
    icon:     '🎙',
  },
  {
    id:       'video',
    label:    'Camera Recording',
    hint:     '.mp4 · .webm · .mov  (optional)',
    accept:   'video/*,.webm',
    required: false,
    icon:     '📷',
  },
  {
    id:       'transcript',
    label:    'Interview Transcript',
    hint:     '.txt  DAIC-WOZ tab-separated format',
    accept:   '.txt,text/plain',
    required: false,
    icon:     '📄',
  },
];

function StatusDot({ status }) {
  return (
    <span
      className={`udot udot--${status}`}
      aria-label={status}
      title={status}
    />
  );
}

export default function UploadPanel({ open, onToggle, onAnalyze }) {
  const [files, setFiles]   = useState({ audio: null, video: null, transcript: null });
  const [status, setStatus] = useState({ audio: 'idle', video: 'idle', transcript: 'idle' });
  const [analyzing, setAnalyzing] = useState(false);
  const inputRefs = {
    audio:      useRef(null),
    video:      useRef(null),
    transcript: useRef(null),
  };

  const canAnalyze = files.audio !== null && !analyzing;

  const handleFileChange = async (slotId, e) => {
    const file = e.target.files?.[0];
    if (!file) return;

    // Flash uploading state
    setStatus(prev => ({ ...prev, [slotId]: 'uploading' }));

    // Simulate a brief validation delay — gives the yellow flash visibility
    await new Promise(r => setTimeout(r, 700));

    // Basic client-side validation
    let valid = true;
    if (slotId === 'audio' && !file.type.startsWith('audio/') && !file.name.endsWith('.webm')) {
      valid = false;
    }
    if (slotId === 'video' && !file.type.startsWith('video/') && !file.name.endsWith('.webm')) {
      valid = false;
    }
    if (slotId === 'transcript' && !file.type.startsWith('text/') && !file.name.endsWith('.txt')) {
      valid = false;
    }

    if (valid) {
      setFiles(prev  => ({ ...prev,  [slotId]: file }));
      setStatus(prev => ({ ...prev, [slotId]: 'success' }));
    } else {
      setFiles(prev  => ({ ...prev,  [slotId]: null }));
      setStatus(prev => ({ ...prev, [slotId]: 'error' }));
      // Reset input so the same file can be re-selected
      if (inputRefs[slotId].current) inputRefs[slotId].current.value = '';
    }
  };

  const handleRemove = (slotId) => {
    setFiles(prev  => ({ ...prev,  [slotId]: null }));
    setStatus(prev => ({ ...prev, [slotId]: 'idle' }));
    if (inputRefs[slotId].current) inputRefs[slotId].current.value = '';
  };

  const handleAnalyze = async () => {
    if (!canAnalyze) return;
    setAnalyzing(true);
    await onAnalyze(files.audio, files.video, files.transcript);
    // Reset panel after submit
    setFiles({ audio: null, video: null, transcript: null });
    setStatus({ audio: 'idle', video: 'idle', transcript: 'idle' });
    setAnalyzing(false);
  };

  return (
    <>
      {/* ── Floating trigger button ── */}
      <button
        className={`upload-fab ${open ? 'upload-fab--open' : ''}`}
        onClick={onToggle}
        aria-label="Upload existing session files"
      >
        <span className="upload-fab__icon">{open ? '✕' : '⬆'}</span>
        <span className="upload-fab__label">
          {open ? 'Close' : 'Upload Files'}
        </span>
      </button>

      {/* ── Slide-in panel ── */}
      <div className={`upload-panel ${open ? 'upload-panel--open' : ''}`}>
        <div className="upload-panel__header">
          <p className="upload-panel__title">Upload Existing Session</p>
          <p className="upload-panel__sub">
            Already have a recorded interview? Upload the files directly
            to skip the live session and run analysis immediately.
          </p>
        </div>

        <div className="upload-panel__slots">
          {SLOTS.map(slot => (
            <div key={slot.id} className="uslot">
              {/* Status indicator dot on the left */}
              <StatusDot status={status[slot.id]} />

              <div className="uslot__body">
                <div className="uslot__top">
                  <span className="uslot__icon">{slot.icon}</span>
                  <div className="uslot__info">
                    <span className="uslot__label">
                      {slot.label}
                      {slot.required && <span className="uslot__req"> *</span>}
                    </span>
                    <span className="uslot__hint">{slot.hint}</span>
                  </div>
                </div>

                {/* File name display or pick button */}
                {files[slot.id] ? (
                  <div className="uslot__file">
                    <span className="uslot__filename">{files[slot.id].name}</span>
                    <button
                      className="uslot__remove"
                      onClick={() => handleRemove(slot.id)}
                      aria-label={`Remove ${slot.label}`}
                    >
                      ✕
                    </button>
                  </div>
                ) : (
                  <button
                    className={`uslot__pick ${status[slot.id] === 'error' ? 'uslot__pick--error' : ''}`}
                    onClick={() => inputRefs[slot.id].current?.click()}
                    disabled={analyzing}
                  >
                    {status[slot.id] === 'uploading' ? 'Validating…'
                     : status[slot.id] === 'error'   ? 'Invalid file — try again'
                     : 'Choose file'}
                  </button>
                )}

                {/* Hidden real file input */}
                <input
                  ref={inputRefs[slot.id]}
                  type="file"
                  accept={slot.accept}
                  style={{ display: 'none' }}
                  onChange={e => handleFileChange(slot.id, e)}
                />
              </div>
            </div>
          ))}
        </div>

        {/* Analyze button */}
        <button
          className={`upload-panel__submit ${canAnalyze ? 'upload-panel__submit--ready' : ''}`}
          onClick={handleAnalyze}
          disabled={!canAnalyze}
        >
          {analyzing ? 'Sending to analysis…' : 'Analyze Uploaded Session →'}
        </button>

        <p className="upload-panel__note">
          * Audio recording is required. Video and transcript are optional
          but improve analysis accuracy.
        </p>
      </div>

      {/* Backdrop — closes panel on click */}
      {open && (
        <div className="upload-backdrop" onClick={onToggle} />
      )}
    </>
  );
}

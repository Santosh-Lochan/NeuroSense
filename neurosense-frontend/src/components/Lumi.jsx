import { useEffect, useRef } from 'react';
import './Lumi.css';

/*
  Lumi — the animated orb mascot
  States: idle | listening | thinking | speaking
*/

const STATE_META = {
  idle:      { label: 'LUMI · STANDBY',   color: '#2dd4bf', glow: 'rgba(45,212,191,0.3)'  },
  listening: { label: 'LUMI · LISTENING', color: '#f87171', glow: 'rgba(248,113,113,0.35)' },
  thinking:  { label: 'LUMI · PROCESSING',color: '#a78bfa', glow: 'rgba(167,139,250,0.3)'  },
  speaking:  { label: 'LUMI · SPEAKING',  color: '#60a5fa', glow: 'rgba(96,165,250,0.3)'   },
};

export default function Lumi({ state = 'idle', size = 120 }) {
  const canvasRef = useRef(null);
  const animRef   = useRef(null);
  const phaseRef  = useRef(0);
  const meta      = STATE_META[state] || STATE_META.idle;

  /* Animated canvas ring for "listening" state */
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const W = canvas.width = size * 2;
    const H = canvas.height = size * 2;
    const cx = W / 2, cy = H / 2;

    const draw = () => {
      ctx.clearRect(0, 0, W, H);
      if (state === 'listening') {
        phaseRef.current += 0.04;
        const bars = 32;
        for (let i = 0; i < bars; i++) {
          const angle  = (i / bars) * Math.PI * 2;
          const amp    = 0.5 + 0.5 * Math.sin(phaseRef.current * 3 + i * 0.5);
          const r1     = size * 0.72;
          const r2     = r1 + amp * size * 0.22;
          const alpha  = 0.3 + amp * 0.5;
          ctx.beginPath();
          ctx.moveTo(cx + Math.cos(angle) * r1, cy + Math.sin(angle) * r1);
          ctx.lineTo(cx + Math.cos(angle) * r2, cy + Math.sin(angle) * r2);
          ctx.strokeStyle = `rgba(248,113,113,${alpha})`;
          ctx.lineWidth = 1.5;
          ctx.stroke();
        }
      } else if (state === 'speaking') {
        phaseRef.current += 0.06;
        const rings = 3;
        for (let r = 0; r < rings; r++) {
          const progress = ((phaseRef.current * 0.5 + r / rings) % 1);
          const radius   = size * 0.6 + progress * size * 0.55;
          const alpha    = (1 - progress) * 0.4;
          ctx.beginPath();
          ctx.arc(cx, cy, radius, 0, Math.PI * 2);
          ctx.strokeStyle = `rgba(96,165,250,${alpha})`;
          ctx.lineWidth = 1;
          ctx.stroke();
        }
      }
      animRef.current = requestAnimationFrame(draw);
    };

    draw();
    return () => cancelAnimationFrame(animRef.current);
  }, [state, size]);

  return (
    <div className={`lumi lumi--${state}`} style={{ '--size': `${size}px` }}>
      {/* Canvas layer for listening bars / speaking rings */}
      <canvas
        ref={canvasRef}
        className="lumi__canvas"
        style={{ width: size * 2, height: size * 2 }}
      />

      {/* Core orb */}
      <div
        className="lumi__orb"
        style={{
          '--orb-color': meta.color,
          '--orb-glow':  meta.glow,
        }}
      >
        <div className="lumi__core" />
        <div className="lumi__highlight" />
        <div className="lumi__highlight lumi__highlight--2" />

        {/* Thinking: orbiting dot */}
        {state === 'thinking' && (
          <div className="lumi__orbit">
            <div className="lumi__dot" />
          </div>
        )}
      </div>

      {/* State label */}
      <span className="lumi__label">{meta.label}</span>
    </div>
  );
}

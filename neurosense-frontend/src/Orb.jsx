import React from 'react';
import './Orb.css';

/*
  state: 'idle' | 'listening' | 'thinking' | 'speaking'
*/
export default function Orb({ state = 'idle', size = 160 }) {
  return (
    <div className={`orb-wrap orb-wrap--${state}`} style={{ '--size': `${size}px` }}>
      {/* Outer glow ring */}
      <div className="orb-halo" />

      {/* Second pulse ring — only on listening */}
      {state === 'listening' && <div className="orb-halo orb-halo--2" />}

      {/* The orb itself */}
      <div className="orb">
        {/* Glass highlight */}
        <div className="orb__shine" />
        <div className="orb__shine orb__shine--2" />

        {/* Thinking dots inside orb */}
        {state === 'thinking' && (
          <div className="orb__dots">
            <span /><span /><span />
          </div>
        )}

        {/* Speaking waveform bars */}
        {state === 'speaking' && (
          <div className="orb__wave">
            {[...Array(5)].map((_, i) => (
              <span key={i} style={{ '--i': i }} />
            ))}
          </div>
        )}
      </div>

      {/* State label */}
      <div className="orb-label">
        <span className="orb-label__dot" />
        <span className="orb-label__text">
          {state === 'idle'      && 'LUMI · STANDBY'}
          {state === 'listening' && 'LUMI · LISTENING'}
          {state === 'thinking'  && 'LUMI · PROCESSING'}
          {state === 'speaking'  && 'LUMI · SPEAKING'}
        </span>
      </div>
    </div>
  );
}

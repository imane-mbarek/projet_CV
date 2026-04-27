import { useState } from 'react'
import { useWebSocket } from './hooks/useWebSocket'
import VideoFeed    from './components/VideoFeed'
import SwimmerPanel from './components/SwimmerPanel'
import AlertLog     from './components/AlertLog'
import YChart       from './components/YChart'

const SPEEDS = [0.3, 0.5, 1.0, 2.0]

export default function App() {
  const {
    connected, frameData, persons, alerts, status,
    fps, frameIdx, sessionTime, activeAlerts, redLineY,
    startPipeline, stopPipeline, setPaused, setSpeed,
  } = useWebSocket()

  const [videoPath, setVideoPath] = useState('')

  const fmt = (s) => {
    const m = Math.floor(s / 60)
    const sec = Math.floor(s % 60)
    return `${String(m).padStart(2,'0')}:${String(sec).padStart(2,'0')}`
  }

  return (
    <div style={{ minHeight: '100vh', background: '#060d1a', color: '#c9d1d9',
                  fontFamily: 'monospace' }}>
      <style>{`
        @keyframes alertBlink { 0%,100%{opacity:1} 50%{opacity:0.5} }
        @keyframes pulse      { 0%,100%{opacity:1} 50%{opacity:0.3} }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        ::-webkit-scrollbar { width: 4px; }
        ::-webkit-scrollbar-track { background: #0a1220; }
        ::-webkit-scrollbar-thumb { background: #1a2535; }
      `}</style>

      {/* Navbar */}
      <div style={{ background: '#0d1826', borderBottom: '1px solid #1a2535',
                    padding: '10px 20px', display: 'flex', alignItems: 'center', gap: 16 }}>
        <span style={{ color: '#1D9E75', fontWeight: 700, fontSize: 16, letterSpacing: 2 }}>
          SAFESWIM
        </span>
        <div style={{ flex: 1 }} />
        <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
          <div style={{ width: 8, height: 8, borderRadius: '50%',
                        background: connected ? '#1D9E75' : '#E24B4A',
                        animation: 'pulse 1.4s infinite' }} />
          <span style={{ fontSize: 11, color: '#8b949e' }}>
            {connected ? 'CONNECTED' : 'DISCONNECTED'}
          </span>
        </div>
        {activeAlerts.length > 0 && (
          <div style={{ background: '#E24B4A', color: '#fff', padding: '3px 10px',
                        borderRadius: 4, fontSize: 11, fontWeight: 700,
                        animation: 'alertBlink 0.7s infinite' }}>
            ALERTE x{activeAlerts.length}
          </div>
        )}
      </div>

      {/* Metric row */}
      <div style={{ display: 'flex', gap: 1, borderBottom: '1px solid #1a2535' }}>
        {[
          ['FPS',      fps.toFixed(1)],
          ['SWIMMERS', persons.length],
          ['ALERTS',   activeAlerts.length],
          ['FRAME',    frameIdx],
          ['TIME',     fmt(sessionTime)],
        ].map(([label, val]) => (
          <div key={label} style={{ flex: 1, padding: '8px 16px',
                                    background: '#0a1220', borderRight: '1px solid #1a2535' }}>
            <div style={{ fontSize: 10, color: '#4a5568', textTransform: 'uppercase',
                          letterSpacing: '0.8px' }}>{label}</div>
            <div style={{ fontSize: 22, fontWeight: 700, color: '#c9d1d9' }}>{val}</div>
          </div>
        ))}
      </div>

      {/* Main layout */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 300px',
                    gap: 0, height: 'calc(100vh - 105px)' }}>

        {/* Left — video + chart */}
        <div style={{ padding: 8, display: 'flex', flexDirection: 'column', gap: 10,
                      overflow: 'auto', borderRight: '1px solid #1a2535' }}>
          <div style={{ width: '100%', display: 'flex', justifyContent: 'center' }}>
            <div style={{ width: '100%', maxWidth: 920 }}>
              <VideoFeed frameData={frameData} persons={persons}
                         activeAlerts={activeAlerts} redLineY={redLineY} />
            </div>
          </div>

          {/* Controls */}
          <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
            <input value={videoPath} onChange={e => setVideoPath(e.target.value)}
                   placeholder="Path to video file..."
                   style={{ flex: 1, background: '#0a1220', border: '1px solid #1a2535',
                            color: '#c9d1d9', padding: '6px 10px', borderRadius: 4,
                            fontFamily: 'monospace', fontSize: 12 }} />
            {!status.running
              ? <button onClick={() => startPipeline(videoPath)}
                        style={btnStyle('#1D9E75')}>START</button>
              : <button onClick={stopPipeline}
                        style={btnStyle('#E24B4A')}>STOP</button>
            }
            {status.running && (
              <button onClick={() => setPaused(!status.paused)}
                      style={btnStyle('#378ADD')}>
                {status.paused ? 'RESUME' : 'PAUSE'}
              </button>
            )}
            {SPEEDS.map(s => (
              <button key={s} onClick={() => setSpeed(s)}
                      style={btnStyle(status.speed === s ? '#EF9F27' : '#1a2535', 36)}>
                {s}x
              </button>
            ))}
          </div>

          <div style={{ background: '#0a1220', border: '1px solid #1a2535',
                        borderRadius: 6, padding: 8 }}>
            <YChart persons={persons} redLineY={redLineY} />
          </div>
        </div>

        {/* Right — swimmers + alerts */}
        <div style={{ display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
          <div style={{ flex: 1, padding: 12, overflowY: 'auto',
                        borderBottom: '1px solid #1a2535' }}>
            <SwimmerPanel persons={persons} />
          </div>
          <div style={{ padding: 12, overflowY: 'auto', maxHeight: '40%' }}>
            <AlertLog alerts={alerts} />
          </div>
        </div>
      </div>
    </div>
  )
}

function btnStyle(bg, minW = 70) {
  return {
    background: bg, color: bg === '#1a2535' ? '#8b949e' : '#fff',
    border: 'none', borderRadius: 4, padding: '6px 12px',
    fontFamily: 'monospace', fontSize: 11, fontWeight: 700,
    cursor: 'pointer', minWidth: minW,
  }
}
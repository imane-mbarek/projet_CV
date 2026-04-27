import React, { useContext } from 'react';
import { SocketContext } from '../App';

const MetricsBar = () => {
  const { frameData } = useContext(SocketContext);
  
  const formatTime = (seconds) => {
    if (!seconds) return "00:00";
    const m = Math.floor(seconds / 60).toString().padStart(2, '0');
    const s = Math.floor(seconds % 60).toString().padStart(2, '0');
    return `${m}:${s}`;
  };

  return (
    <div className="metrics-bar">
      <div className="card">
        <div className="section-label">FPS</div>
        <div className="metric-value">{frameData?.fps?.toFixed(1) || '0.0'}</div>
      </div>
      <div className="card">
        <div className="section-label">SWIMMERS</div>
        <div className="metric-value">{frameData?.persons?.length || 0}</div>
      </div>
      <div className="card">
        <div className="section-label">ALERTS</div>
        <div className="metric-value" style={{ color: frameData?.active_alerts?.length > 0 ? 'var(--alert-red)' : 'inherit' }}>
          {frameData?.active_alerts?.length || 0}
        </div>
      </div>
      <div className="card">
        <div className="section-label">FRAME</div>
        <div className="metric-value">{frameData?.frame_idx || 0}</div>
      </div>
      <div className="card">
        <div className="section-label">SESSION TIME</div>
        <div className="metric-value">{formatTime(frameData?.session_time)}</div>
      </div>
    </div>
  );
};

export default MetricsBar;

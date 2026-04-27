import React, { useContext, useState } from 'react';
import { SocketContext } from '../App';

const Controls = () => {
  const { emitCommand, backendStatus } = useContext(SocketContext);
  const [videoPath, setVideoPath] = useState('C:/path/to/video.mp4');

  const speeds = [0.3, 0.5, 1.0, 2.0];

  return (
    <div className="card controls">
      <input 
        type="text" 
        value={videoPath} 
        onChange={(e) => setVideoPath(e.target.value)}
        placeholder="Enter video path..."
      />
      <button 
        className={backendStatus.running ? 'active' : ''} 
        onClick={() => {
          if (backendStatus.running) {
            emitCommand('stop_pipeline');
          } else {
            emitCommand('start_pipeline', { video: videoPath });
          }
        }}
      >
        {backendStatus.running ? 'STOP' : 'START'}
      </button>
      <button 
        onClick={() => emitCommand('set_paused', { paused: !backendStatus.paused })}
        disabled={!backendStatus.running}
      >
        {backendStatus.paused ? 'RESUME' : 'PAUSE'}
      </button>
      
      <div style={{ display: 'flex', gap: '4px', marginLeft: 'auto' }}>
        {speeds.map(s => (
          <button 
            key={s}
            className={backendStatus.speed === s ? 'active' : ''}
            onClick={() => emitCommand('set_speed', { speed: s })}
            disabled={!backendStatus.running}
          >
            {s}x
          </button>
        ))}
      </div>
    </div>
  );
};

export default Controls;

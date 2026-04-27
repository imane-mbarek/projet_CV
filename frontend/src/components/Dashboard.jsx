import React, { useContext } from 'react';
import { SocketContext } from '../App';
import VideoFeed from './VideoFeed';
import Controls from './Controls';
import MetricsBar from './MetricsBar';
import ChartArea from './ChartArea';
import SwimmerCards from './SwimmerCards';
import AlertLog from './AlertLog';

const Dashboard = () => {
  const { connected, frameData } = useContext(SocketContext);
  const activeAlerts = frameData?.active_alerts?.length > 0;

  return (
    <div className="dashboard-layout">
      {/* Navbar */}
      <div className="navbar">
        <div className="logo">
          SAFESWIM
          <div className={`connection-dot ${connected ? 'connected' : 'disconnected'}`} />
        </div>
        {activeAlerts && <div className="alerte-pill">ALERTE</div>}
      </div>

      <div className="left-panel">
        <div className="card" style={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
          <div className="section-label">SWIMMERS</div>
          <div style={{ overflowY: 'auto', flex: 1, paddingRight: '4px' }}>
            <SwimmerCards />
          </div>
        </div>
      </div>

      <div className="center-panel">
        <MetricsBar />
        <VideoFeed />
        <Controls />
      </div>

      <div className="right-panel">
        <div className="card">
          <div className="section-label">DEPTH / Y-POSITION</div>
          <ChartArea />
        </div>
        
        <div className="card" style={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
          <div className="section-label">ALERT LOG</div>
          <div style={{ overflowY: 'auto', flex: 1, paddingRight: '4px' }}>
            <AlertLog />
          </div>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;

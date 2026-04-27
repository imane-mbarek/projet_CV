import { useEffect, useRef, useState, useCallback } from 'react'
import { io } from 'socket.io-client'

const SOCKET_URL = 'http://localhost:5001'

export function useWebSocket() {
  const socketRef = useRef(null)

  const [connected,    setConnected]    = useState(false)
  const [frameData,    setFrameData]    = useState(null)
  const [persons,      setPersons]      = useState([])
  const [alerts,       setAlerts]       = useState([])
  const [status,       setStatus]       = useState({ running: false, paused: false, speed: 1.0 })
  const [fps,          setFps]          = useState(0)
  const [frameIdx,     setFrameIdx]     = useState(0)
  const [sessionTime,  setSessionTime]  = useState(0)
  const [activeAlerts, setActiveAlerts] = useState([])
  const [redLineY,     setRedLineY]     = useState(300)

  useEffect(() => {
    const socket = io(SOCKET_URL, {
      transports: ['websocket', 'polling'],
      reconnection: true,
      reconnectionAttempts: Infinity,
      reconnectionDelay: 1000,
      reconnectionDelayMax: 5000,
      timeout: 20000,
    })
    socketRef.current = socket

    socket.on('connect',    () => setConnected(true))
    socket.on('disconnect', () => setConnected(false))
    socket.on('connect_error', () => setConnected(false))

    socket.on('frame', (data) => {
      setFrameData(data.frame)
      setPersons(data.persons          ?? [])
      setFps(data.fps                  ?? 0)
      setFrameIdx(data.frame_idx       ?? 0)
      setSessionTime(data.session_time ?? 0)
      setActiveAlerts(data.active_alerts ?? [])
      setRedLineY(data.red_line_y      ?? 300)
    })

    socket.on('new_alert', (alert) => {
      setAlerts(prev => [alert, ...prev].slice(0, 50))
    })

    socket.on('status', (s) => setStatus(s))

    return () => socket.disconnect()
  }, [])

  const startPipeline = useCallback((videoPath) => {
    socketRef.current?.emit('start_pipeline', { video: videoPath })
  }, [])

  const stopPipeline = useCallback(() => {
    socketRef.current?.emit('stop_pipeline')
  }, [])

  const setPaused = useCallback((paused) => {
    socketRef.current?.emit('set_paused', { paused })
  }, [])

  const setSpeed = useCallback((speed) => {
    socketRef.current?.emit('set_speed', { speed })
  }, [])

  return {
    connected, frameData, persons, alerts, status,
    fps, frameIdx, sessionTime, activeAlerts, redLineY,
    startPipeline, stopPipeline, setPaused, setSpeed,
  }
}
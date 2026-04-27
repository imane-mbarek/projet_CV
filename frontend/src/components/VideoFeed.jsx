import { useEffect, useRef } from 'react'

const SWIMMER_COLORS = ['#1D9E75','#378ADD','#EF9F27','#D4537E','#5DCAA5','#AFA9EC']
const getColor = (id) => SWIMMER_COLORS[id % SWIMMER_COLORS.length]
const FEED_ZOOM = 1.1

export default function VideoFeed({ frameData, persons, activeAlerts, redLineY }) {
  const canvasRef = useRef(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    ctx.clearRect(0, 0, canvas.width, canvas.height)

    const sx = canvas.width  / 640
    const sy = canvas.height / 480

    // Danger line
    const ry = redLineY * sy
    ctx.save()
    ctx.setLineDash([8, 6])
    ctx.strokeStyle = '#E24B4A'
    ctx.lineWidth = 1.5
    ctx.beginPath()
    ctx.moveTo(0, ry)
    ctx.lineTo(canvas.width, ry)
    ctx.stroke()
    ctx.restore()

    persons.forEach((p) => {
      const [x, y, w, h] = p.bbox
      const color      = getColor(p.person_id)
      const isDrowning = p.drowning_alert
      const isKalman   = p.is_predicted

      // Bounding box
      ctx.save()
      ctx.strokeStyle = isDrowning ? '#E24B4A' : color
      ctx.lineWidth   = isDrowning ? 3 : 2
      if (isKalman) ctx.setLineDash([6, 4])
      ctx.strokeRect(x * sx, y * sy, w * sx, h * sy)
      ctx.restore()

      // Trail
      if (p.centroid_history.length > 1) {
        ctx.save()
        ctx.strokeStyle  = color
        ctx.lineWidth    = 1.5
        ctx.globalAlpha  = 0.5
        ctx.beginPath()
        p.centroid_history.forEach(([cx, cy], i) => {
          i === 0 ? ctx.moveTo(cx * sx, cy * sy) : ctx.lineTo(cx * sx, cy * sy)
        })
        ctx.stroke()
        ctx.restore()
      }

      // Label
      const label = `ID ${p.person_id}  ${p.speed_px} px/f`
      ctx.save()
      ctx.font = 'bold 12px monospace'
      ctx.fillStyle = isDrowning ? '#E24B4A' : color
      ctx.fillRect(x * sx, y * sy - 20, ctx.measureText(label).width + 10, 20)
      ctx.fillStyle = '#071018'
      ctx.fillText(label, x * sx + 5, y * sy - 6)
      ctx.restore()

      // Time below line
      if (p.time_below_line > 0) {
        ctx.save()
        ctx.fillStyle = '#E24B4A'
        ctx.font      = 'bold 10px monospace'
        ctx.fillText(`${p.time_below_line}s`, (x + w) * sx - 30, y * sy + 14)
        ctx.restore()
      }

      // Legs suspicious
      if (p.legs_suspicious) {
        ctx.save()
        ctx.fillStyle = '#EF9F27'
        ctx.font      = '10px monospace'
        ctx.fillText('jambes!', x * sx, (y + h) * sy + 14)
        ctx.restore()
      }
    })
  }, [persons, redLineY])

  return (
    <div style={{
      position: 'relative',
      background: '#000',
      borderRadius: 4,
      overflow: 'hidden',
      border: '1px solid #141f2f',
      boxShadow: '0 10px 28px rgba(0,0,0,0.5)',
      marginInline: 'auto',
    }}>
      <div style={{
        position: 'relative',
        width: '100%',
        aspectRatio: '4/3',
        overflow: 'hidden',
      }}>
      {frameData
        ? <img
            src={`data:image/jpeg;base64,${frameData}`}
            style={{
              width: '100%',
              height: '100%',
              display: 'block',
              objectFit: 'cover',
              transform: `scale(${FEED_ZOOM})`,
              transformOrigin: 'center center',
              filter: 'contrast(1.12) saturate(1.06) brightness(1.03)',
            }}
            alt="feed"
          />
        : <div style={{
            width: '100%', aspectRatio: '4/3', background: '#0a1220',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            color: '#4a5568', fontFamily: 'monospace', fontSize: 13,
          }}>
            NO SIGNAL
          </div>
      }
      <canvas
        ref={canvasRef}
        width={640} height={480}
        style={{
          position: 'absolute',
          top: 0,
          left: 0,
          width: '100%',
          height: '100%',
          transform: `scale(${FEED_ZOOM})`,
          transformOrigin: 'center center',
        }}
      />
      <div style={{
        position: 'absolute',
        inset: 0,
        border: '1px solid rgba(94, 118, 145, 0.28)',
        pointerEvents: 'none',
      }} />
      </div>
      {activeAlerts.length > 0 && (
        <div style={{
          position: 'absolute', bottom: 0, left: 0, right: 0,
          background: '#A32D2D', color: '#fff', textAlign: 'center',
          padding: '6px 0', fontFamily: 'monospace', fontWeight: 700,
          fontSize: 13, animation: 'alertBlink 0.7s infinite',
        }}>
          ALERTE NOYADE — ID {activeAlerts.join(', ')}
        </div>
      )}
    </div>
  )
}
const SWIMMER_COLORS = ['#1D9E75','#378ADD','#EF9F27','#D4537E','#5DCAA5','#AFA9EC']
const getColor = (id) => SWIMMER_COLORS[id % SWIMMER_COLORS.length]

function badge(person) {
  if (person.drowning_alert)   return { label: 'DANGER',    color: '#E24B4A' }
  if (person.legs_suspicious)  return { label: 'ATTENTION', color: '#EF9F27' }
  if (person.frames_lost > 10) return { label: 'PERDU',     color: '#6e7681' }
  return                              { label: 'OK',         color: '#1D9E75' }
}

function SwimmerCard({ person }) {
  const color  = getColor(person.person_id)
  const b      = badge(person)
  const barW   = Math.min(100, (person.time_below_line / 6) * 100)
  const barColor = barW < 33 ? '#1D9E75' : barW < 66 ? '#EF9F27' : '#E24B4A'

  return (
    <div style={{
      background: '#0a1220', border: '1px solid #1a2535',
      borderRadius: 8, padding: '10px 14px', marginBottom: 8,
    }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 6 }}>
        <div style={{
          width: 28, height: 28, borderRadius: '50%', background: color,
          display: 'flex', alignItems: 'center', justifyContent: 'center',
          fontWeight: 700, fontSize: 12, color: '#000', flexShrink: 0,
        }}>
          {person.person_id}
        </div>
        <span style={{ fontFamily: 'monospace', fontSize: 12, color: '#c9d1d9', flex: 1 }}>
          {person.speed_px} px/f
          {person.is_predicted &&
            <span style={{ color: '#378ADD', marginLeft: 6 }}>[Kalman]</span>}
        </span>
        <span style={{
          background: b.color + '22', color: b.color,
          border: `1px solid ${b.color}`, borderRadius: 4,
          padding: '1px 6px', fontSize: 10, fontFamily: 'monospace', fontWeight: 700,
        }}>
          {b.label}
        </span>
      </div>
      <div style={{ fontFamily: 'monospace', fontSize: 11, color: '#8b949e', marginBottom: 6 }}>
        below line: {person.time_below_line}s
      </div>
      <div style={{ background: '#1a2535', borderRadius: 4, height: 4 }}>
        <div style={{
          width: `${barW}%`, height: '100%', background: barColor,
          borderRadius: 4, transition: 'width 0.3s',
        }} />
      </div>
    </div>
  )
}

export default function SwimmerPanel({ persons }) {
  return (
    <div>
      <div style={{
        fontFamily: 'monospace', fontSize: 10, color: '#4a5568',
        textTransform: 'uppercase', letterSpacing: '0.8px', marginBottom: 10,
      }}>
        Swimmers ({persons.length})
      </div>
      {persons.length === 0
        ? <div style={{ color: '#4a5568', fontFamily: 'monospace', fontSize: 12 }}>
            No swimmers detected
          </div>
        : persons.map(p => <SwimmerCard key={p.person_id} person={p} />)
      }
    </div>
  )
}
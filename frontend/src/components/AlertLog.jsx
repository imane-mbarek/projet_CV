const TYPE_COLORS = {
  drowning: '#E24B4A',
  warning:  '#EF9F27',
  info:     '#378ADD',
  error:    '#D4537E',
}
const TYPE_LABELS = {
  drowning: 'ALERTE',
  warning:  'WARN',
  info:     'INFO',
  error:    'ERR',
}

export default function AlertLog({ alerts }) {
  return (
    <div>
      <div style={{
        fontFamily: 'monospace', fontSize: 10, color: '#4a5568',
        textTransform: 'uppercase', letterSpacing: '0.8px', marginBottom: 10,
      }}>
        Alert Log ({alerts.length})
      </div>
      <div style={{ maxHeight: 220, overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: 4 }}>
        {alerts.length === 0
          ? <div style={{ color: '#4a5568', fontFamily: 'monospace', fontSize: 12 }}>No alerts</div>
          : alerts.map((a, i) => {
              const color = TYPE_COLORS[a.type] ?? '#8b949e'
              const label = TYPE_LABELS[a.type] ?? a.type.toUpperCase()
              return (
                <div key={i} style={{
                  display: 'flex', gap: 8, alignItems: 'flex-start',
                  opacity: Math.max(0.4, 1 - i * 0.04),
                }}>
                  <span style={{
                    color, fontFamily: 'monospace', fontSize: 10,
                    fontWeight: 700, flexShrink: 0, paddingTop: 1,
                  }}>
                    [{label}]
                  </span>
                  <span style={{ color: '#c9d1d9', fontFamily: 'monospace', fontSize: 11, flex: 1 }}>
                    {a.message}
                  </span>
                  <span style={{
                    color: '#4a5568', fontFamily: 'monospace', fontSize: 10,
                    flexShrink: 0, paddingTop: 1,
                  }}>
                    {a.time}
                  </span>
                </div>
              )
            })
        }
      </div>
    </div>
  )
}
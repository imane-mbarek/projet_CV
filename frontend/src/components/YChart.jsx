import { LineChart, Line, YAxis, ReferenceLine, ResponsiveContainer, Legend } from 'recharts'

const SWIMMER_COLORS = ['#1D9E75','#378ADD','#EF9F27','#D4537E','#5DCAA5','#AFA9EC']
const getColor = (id) => SWIMMER_COLORS[id % SWIMMER_COLORS.length]

export default function YChart({ persons, redLineY }) {
  // Build unified data array — one entry per frame index
  const maxLen = Math.max(0, ...persons.map(p => p.y_positions.length))
  const data = Array.from({ length: maxLen }, (_, i) => {
    const point = { i }
    persons.forEach(p => {
      point[`id${p.person_id}`] = p.y_positions[i] ?? null
    })
    return point
  })

  return (
    <div>
      <div style={{ fontFamily: 'monospace', fontSize: 10, color: '#4a5568',
                    textTransform: 'uppercase', letterSpacing: '0.8px', marginBottom: 8 }}>
        Y Position (depth)
      </div>
      <ResponsiveContainer width="100%" height={160}>
        <LineChart data={data}>
          <YAxis domain={[0, 480]} reversed tick={{ fill: '#4a5568', fontSize: 10 }}
                 width={30} />
          <ReferenceLine y={redLineY} stroke="#E24B4A" strokeDasharray="4 3" />
          {persons.map(p => (
            <Line key={p.person_id} type="monotone" dataKey={`id${p.person_id}`}
                  stroke={getColor(p.person_id)} dot={false} isAnimationActive={false}
                  strokeWidth={1.5} connectNulls />
          ))}
          {persons.length > 0 && <Legend wrapperStyle={{ fontSize: 10, fontFamily: 'monospace' }} />}
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}
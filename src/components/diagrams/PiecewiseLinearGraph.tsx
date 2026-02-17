/**
 * Fig. 3.2 — Piecewise linear function with 4 regions & 3 joints
 * Demonstrates how a shallow network creates a piecewise linear function
 */
export default function PiecewiseLinearGraph() {
  const W = 580, H = 340
  const pad = { top: 30, right: 30, bottom: 55, left: 50 }
  const gw = W - pad.left - pad.right
  const gh = H - pad.top - pad.bottom

  const xMin = -1, xMax = 7, yMin = -0.5, yMax = 4.5
  const sx = (v: number) => pad.left + ((v - xMin) / (xMax - xMin)) * gw
  const sy = (v: number) => pad.top + ((yMax - v) / (yMax - yMin)) * gh

  // The piecewise linear function with 3 joints
  const joints = [
    { x: 0, y: 0.5 },   // joint 1
    { x: 2, y: 3.5 },   // joint 2 (peak)
    { x: 4, y: 1.0 },   // joint 3 (valley)
  ]

  // Segments: extend before joint1 and after joint3
  const points = [
    { x: -1, y: 0.5 },   // R1: flat (all off)
    ...joints,
    { x: 6.5, y: 3.5 },  // R4: rising again
  ]

  const pathD = points.map((p, i) =>
    `${i === 0 ? 'M' : 'L'}${sx(p.x).toFixed(1)},${sy(p.y).toFixed(1)}`
  ).join(' ')

  // Region colors
  const regions = [
    { x1: -1, x2: 0, label: 'R1', color: '#d946ef33', textY: 1.5 },
    { x1: 0, x2: 2, label: 'R2', color: '#00e5ff22', textY: 3.0 },
    { x1: 2, x2: 4, label: 'R3', color: '#ff910022', textY: 2.8 },
    { x1: 4, x2: 6.5, label: 'R4', color: '#39ffb022', textY: 3.2 },
  ]

  // Region info for bottom legend
  const regionInfo = [
    { label: 'R1', desc: 'all off → pente = 0', color: '#d946ef' },
    { label: 'R2', desc: 'h₁ on → ϕ₁·θ₁₁', color: '#00e5ff' },
    { label: 'R3', desc: 'h₁,h₂ on', color: '#ff9100' },
    { label: 'R4', desc: 'all on', color: '#39ffb0' },
  ]

  return (
    <svg viewBox={`0 0 ${W} ${H}`} width="100%" style={{ maxWidth: 580 }}>
      <rect width={W} height={H} rx={8} fill="#0d1117" />

      {/* Region shading */}
      {regions.map((r, i) => (
        <rect key={i} x={sx(r.x1)} y={pad.top} width={sx(r.x2) - sx(r.x1)} height={gh}
          fill={r.color} />
      ))}

      {/* Grid */}
      {[0, 2, 4, 6].map(t => (
        <line key={`gx${t}`} x1={sx(t)} x2={sx(t)} y1={pad.top} y2={H - pad.bottom}
          stroke="rgba(255,255,255,0.06)" strokeWidth={1} />
      ))}
      {[0, 1, 2, 3, 4].map(t => (
        <line key={`gy${t}`} x1={pad.left} x2={W - pad.right} y1={sy(t)} y2={sy(t)}
          stroke="rgba(255,255,255,0.06)" strokeWidth={1} />
      ))}

      {/* Axes */}
      <line x1={pad.left} x2={W - pad.right} y1={sy(0)} y2={sy(0)}
        stroke="rgba(255,255,255,0.35)" strokeWidth={1.5} />
      <line x1={sx(0)} x2={sx(0)} y1={pad.top} y2={H - pad.bottom}
        stroke="rgba(255,255,255,0.35)" strokeWidth={1.5} />

      {/* Axis labels */}
      <text x={W - pad.right + 8} y={sy(0) + 4} fill="rgba(255,255,255,0.6)"
        fontSize={13} fontFamily="monospace" fontWeight={700}>x</text>
      <text x={sx(0) - 8} y={pad.top - 8} fill="rgba(255,255,255,0.6)"
        fontSize={13} fontFamily="monospace" fontWeight={700} textAnchor="end">y</text>

      {/* Function line (glow) */}
      <path d={pathD} fill="none" stroke="#00e5ff" strokeWidth={6} opacity={0.15}
        strokeLinejoin="round" />
      {/* Function line */}
      <path d={pathD} fill="none" stroke="#00e5ff" strokeWidth={2.5}
        strokeLinejoin="round" strokeLinecap="round" />

      {/* Joint dots */}
      {joints.map((j, i) => (
        <g key={i}>
          <circle cx={sx(j.x)} cy={sy(j.y)} r={5} fill="#0d1117" stroke="#ff9100"
            strokeWidth={2} />
          <text x={sx(j.x)} y={sy(j.y) - 12} fill="#ff9100" fontSize={10}
            textAnchor="middle" fontFamily="monospace" fontWeight={700}>
            joint{i + 1}
          </text>
        </g>
      ))}

      {/* Region labels */}
      {regions.map((r, i) => (
        <text key={i} x={sx((r.x1 + r.x2) / 2)} y={sy(r.textY)}
          fill={regionInfo[i].color} fontSize={14} fontWeight={700}
          textAnchor="middle" fontFamily="monospace" opacity={0.8}>
          {r.label}
        </text>
      ))}

      {/* Bottom legend */}
      {regionInfo.map((r, i) => (
        <g key={i} transform={`translate(${pad.left + i * 135}, ${H - 18})`}>
          <rect width={10} height={10} rx={2} fill={r.color} opacity={0.8} />
          <text x={14} y={9} fill="rgba(255,255,255,0.55)" fontSize={9.5}
            fontFamily="monospace">
            <tspan fontWeight={700} fill={r.color}>{r.label}</tspan> {r.desc}
          </text>
        </g>
      ))}
    </svg>
  )
}

/**
 * Fig. 3.1 — ReLU function graph: y = max(0, z)
 * Pure SVG, cyberpunk-styled
 */
export default function ReluGraph() {
  const W = 520, H = 300
  const pad = { top: 30, right: 30, bottom: 40, left: 50 }
  const gw = W - pad.left - pad.right
  const gh = H - pad.top - pad.bottom

  // Domain: z ∈ [-4, 6], Range: y ∈ [-1, 6]
  const xMin = -4, xMax = 6, yMin = -1, yMax = 6
  const sx = (v: number) => pad.left + ((v - xMin) / (xMax - xMin)) * gw
  const sy = (v: number) => pad.top + ((yMax - v) / (yMax - yMin)) * gh

  // Grid lines
  const xTicks = [-4, -2, 0, 2, 4, 6]
  const yTicks = [0, 2, 4, 6]

  return (
    <svg viewBox={`0 0 ${W} ${H}`} width="100%" style={{ maxWidth: 520 }}>
      {/* Background */}
      <rect width={W} height={H} rx={8} fill="#0d1117" />

      {/* Grid */}
      {xTicks.map(t => (
        <line key={`gx${t}`} x1={sx(t)} x2={sx(t)} y1={pad.top} y2={H - pad.bottom}
          stroke="rgba(255,255,255,0.06)" strokeWidth={1} />
      ))}
      {yTicks.map(t => (
        <line key={`gy${t}`} x1={pad.left} x2={W - pad.right} y1={sy(t)} y2={sy(t)}
          stroke="rgba(255,255,255,0.06)" strokeWidth={1} />
      ))}

      {/* Axes */}
      <line x1={pad.left} x2={W - pad.right} y1={sy(0)} y2={sy(0)}
        stroke="rgba(255,255,255,0.35)" strokeWidth={1.5} />
      <line x1={sx(0)} x2={sx(0)} y1={pad.top} y2={H - pad.bottom}
        stroke="rgba(255,255,255,0.35)" strokeWidth={1.5} />

      {/* Axis labels */}
      {xTicks.map(t => (
        <text key={`lx${t}`} x={sx(t)} y={sy(0) + 18} fill="rgba(255,255,255,0.5)"
          fontSize={11} textAnchor="middle" fontFamily="monospace">{t}</text>
      ))}
      {yTicks.filter(t => t !== 0).map(t => (
        <text key={`ly${t}`} x={sx(0) - 10} y={sy(t) + 4} fill="rgba(255,255,255,0.5)"
          fontSize={11} textAnchor="end" fontFamily="monospace">{t}</text>
      ))}

      {/* Axis names */}
      <text x={W - pad.right + 8} y={sy(0) + 4} fill="rgba(255,255,255,0.6)"
        fontSize={13} fontFamily="monospace" fontWeight={700}>z</text>
      <text x={sx(0) - 6} y={pad.top - 8} fill="rgba(255,255,255,0.6)"
        fontSize={13} fontFamily="monospace" fontWeight={700} textAnchor="end">y</text>

      {/* ReLU: flat part z < 0 */}
      <line x1={sx(xMin)} x2={sx(0)} y1={sy(0)} y2={sy(0)}
        stroke="#d946ef" strokeWidth={2.5} strokeLinecap="round" />

      {/* ReLU: linear part z >= 0 */}
      <line x1={sx(0)} x2={sx(yMax)} y1={sy(0)} y2={sy(yMax)}
        stroke="#00e5ff" strokeWidth={2.5} strokeLinecap="round" />

      {/* Origin dot */}
      <circle cx={sx(0)} cy={sy(0)} r={4} fill="#00e5ff" />

      {/* Glow effect on the active part */}
      <line x1={sx(0)} x2={sx(yMax)} y1={sy(0)} y2={sy(yMax)}
        stroke="#00e5ff" strokeWidth={6} strokeLinecap="round" opacity={0.15} />

      {/* Annotations */}
      <text x={sx(-2.5)} y={sy(0) - 12} fill="#d946ef" fontSize={12}
        fontFamily="monospace" textAnchor="middle" fontWeight={700}>
        clipped à 0
      </text>
      <text x={sx(4)} y={sy(4) - 12} fill="#00e5ff" fontSize={12}
        fontFamily="monospace" textAnchor="middle" fontWeight={700}>
        pente = 1
      </text>

      {/* Title */}
      <text x={W / 2} y={H - 8} fill="rgba(255,255,255,0.4)" fontSize={11}
        textAnchor="middle" fontFamily="monospace">
        ReLU(z) = max(0, z)
      </text>
    </svg>
  )
}

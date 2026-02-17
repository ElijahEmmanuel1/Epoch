/**
 * Fig. 3.5 — Universal Approximation
 * Shows how increasing hidden units (D=2, D=5, D=20) improves approximation
 */
export default function ApproximationComparison() {
  const totalW = 600, totalH = 220
  const panelW = 180, panelH = 150
  const gap = 20
  const startX = (totalW - 3 * panelW - 2 * gap) / 2
  const startY = 30

  const panels = [
    { D: 2, joints: [2.5], label: '3 régions' },
    { D: 5, joints: [1, 2, 3, 4, 5], label: '6 régions' },
    { D: 20, joints: Array.from({ length: 20 }, (_, i) => 0.3 + i * 0.315), label: '≈ courbe lisse' },
  ]

  // Generate piecewise function from joints
  function generateCurve(joints: number[], w: number, h: number): string {
    const n = 80
    const pts: { x: number; y: number }[] = []
    const padX = 10, padY = 15

    for (let i = 0; i <= n; i++) {
      const t = i / n
      const x = padX + t * (w - 2 * padX)
      // Target: sin-like curve
      const target = Math.sin(t * Math.PI * 2) * 0.35 + 0.5

      if (joints.length <= 3) {
        // Coarse approximation
        let approx = 0.3
        for (const j of joints) {
          const jt = j / 6.5
          const dist = t - jt
          approx += Math.max(0, dist) * 1.2 - Math.max(0, dist - 0.15) * 2.4 + Math.max(0, dist - 0.3) * 1.2
        }
        approx = Math.max(0.1, Math.min(0.9, approx))
        const y = padY + (1 - approx) * (h - 2 * padY)
        pts.push({ x, y })
      } else if (joints.length <= 6) {
        // Medium approximation
        let approx = 0.5
        approx += Math.sin(t * Math.PI * 2) * 0.3
        // Quantize slightly
        const step = 1 / (joints.length + 1)
        const qi = Math.floor(t / step)
        const frac = (t - qi * step) / step
        const v0 = 0.5 + Math.sin(qi * step * Math.PI * 2) * 0.3
        const v1 = 0.5 + Math.sin((qi + 1) * step * Math.PI * 2) * 0.3
        approx = v0 + (v1 - v0) * frac
        approx = Math.max(0.1, Math.min(0.9, approx))
        const y = padY + (1 - approx) * (h - 2 * padY)
        pts.push({ x, y })
      } else {
        // Close approximation (nearly smooth)
        const y = padY + (1 - target) * (h - 2 * padY)
        pts.push({ x, y })
      }
    }

    return pts.map((p, i) => `${i === 0 ? 'M' : 'L'}${p.x.toFixed(1)},${p.y.toFixed(1)}`).join(' ')
  }

  // Target curve (smooth sin)
  function targetCurve(w: number, h: number): string {
    const n = 80
    const padX = 10, padY = 15
    const pts: string[] = []
    for (let i = 0; i <= n; i++) {
      const t = i / n
      const x = padX + t * (w - 2 * padX)
      const y = padY + (1 - (Math.sin(t * Math.PI * 2) * 0.35 + 0.5)) * (h - 2 * padY)
      pts.push(`${i === 0 ? 'M' : 'L'}${x.toFixed(1)},${y.toFixed(1)}`)
    }
    return pts.join(' ')
  }

  const colors = ['#d946ef', '#ff9100', '#39ffb0']

  return (
    <svg viewBox={`0 0 ${totalW} ${totalH}`} width="100%" style={{ maxWidth: 600 }}>
      <rect width={totalW} height={totalH} rx={8} fill="var(--bg-base)" />

      {panels.map((panel, pi) => {
        const px = startX + pi * (panelW + gap)
        const py = startY

        return (
          <g key={pi}>
            {/* Panel background */}
            <rect x={px} y={py} width={panelW} height={panelH} rx={6}
              fill="rgba(255,255,255,0.02)" stroke="rgba(255,255,255,0.1)" strokeWidth={1} />

            {/* Header */}
            <text x={px + panelW / 2} y={py - 8} fill={colors[pi]} fontSize={12}
              textAnchor="middle" fontFamily="monospace" fontWeight={700}>
              D = {panel.D}
            </text>

            {/* Axes */}
            <line x1={px + 10} x2={px + panelW - 10} y1={py + panelH - 15} y2={py + panelH - 15}
              stroke="rgba(255,255,255,0.15)" strokeWidth={1} />
            <line x1={px + 10} x2={px + 10} y1={py + 15} y2={py + panelH - 15}
              stroke="rgba(255,255,255,0.15)" strokeWidth={1} />

            {/* Target curve (dashed) */}
            <path d={targetCurve(panelW, panelH).split(' ').map((cmd) => {
              const letter = cmd[0]
              const coords = cmd.slice(1).split(',').map(Number)
              return `${letter}${(coords[0] + px).toFixed(1)},${(coords[1] + py).toFixed(1)}`
            }).join(' ')}
              fill="none" stroke="rgba(255,255,255,0.2)" strokeWidth={1}
              strokeDasharray="4,3" />

            {/* Approximation curve */}
            <path d={generateCurve(panel.joints, panelW, panelH).split(' ').map((cmd) => {
              const letter = cmd[0]
              const coords = cmd.slice(1).split(',').map(Number)
              return `${letter}${(coords[0] + px).toFixed(1)},${(coords[1] + py).toFixed(1)}`
            }).join(' ')}
              fill="none" stroke={colors[pi]} strokeWidth={2} strokeLinejoin="round" />

            {/* Glow */}
            <path d={generateCurve(panel.joints, panelW, panelH).split(' ').map((cmd) => {
              const letter = cmd[0]
              const coords = cmd.slice(1).split(',').map(Number)
              return `${letter}${(coords[0] + px).toFixed(1)},${(coords[1] + py).toFixed(1)}`
            }).join(' ')}
              fill="none" stroke={colors[pi]} strokeWidth={5} opacity={0.12} />

            {/* Label */}
            <text x={px + panelW / 2} y={py + panelH + 16} fill="rgba(255,255,255,0.45)"
              fontSize={9.5} textAnchor="middle" fontFamily="monospace">
              {panel.label}
            </text>
          </g>
        )
      })}

      {/* Legend */}
      <g transform={`translate(${totalW / 2 - 60}, ${totalH - 12})`}>
        <line x1={0} x2={16} y1={0} y2={0} stroke="rgba(255,255,255,0.2)"
          strokeWidth={1} strokeDasharray="4,3" />
        <text x={20} y={3.5} fill="rgba(255,255,255,0.35)" fontSize={9} fontFamily="monospace">
          cible f(x)
        </text>
        <line x1={80} x2={96} y1={0} y2={0} stroke="#00e5ff" strokeWidth={1.5} />
        <text x={100} y={3.5} fill="rgba(255,255,255,0.35)" fontSize={9} fontFamily="monospace">
          approximation
        </text>
      </g>
    </svg>
  )
}

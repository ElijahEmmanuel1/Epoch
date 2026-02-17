/**
 * Fig. 3.3 — Shallow Network Pipeline
 * Flowchart: 3 steps (linear → ReLU → linear combination)
 */
export default function ShallowNetPipeline() {
  const W = 680, H = 280

  // Colors
  const cyan = '#00e5ff'
  const magenta = '#d946ef'
  const orange = '#ff9100'
  const green = '#39ffb0'
  const textDim = 'rgba(255,255,255,0.55)'

  // Arrow helper
  const Arrow = ({ x1, x2, y }: { x1: number; x2: number; y: number }) => (
    <g>
      <line x1={x1} x2={x2 - 6} y1={y} y2={y} stroke="rgba(255,255,255,0.3)" strokeWidth={1.5} />
      <polygon points={`${x2},${y} ${x2 - 8},${y - 4} ${x2 - 8},${y + 4}`}
        fill="rgba(255,255,255,0.3)" />
    </g>
  )

  // Layout
  const step1X = 20, step2X = 220, step3X = 420, outX = 620
  const boxW = 160
  const midY = H / 2

  // Neuron rows
  const neuronY = [midY - 70, midY, midY + 70]
  const neuronLabels = [
    { pre: 'θ₁₀ + θ₁₁·x', post: 'h₁' },
    { pre: 'θ₂₀ + θ₂₁·x', post: 'h₂' },
    { pre: 'θ₃₀ + θ₃₁·x', post: 'h₃' },
  ]

  return (
    <svg viewBox={`0 0 ${W} ${H}`} width="100%" style={{ maxWidth: 680 }}>
      <rect width={W} height={H} rx={8} fill="var(--bg-base)" />

      {/* Step labels */}
      <text x={step1X + boxW / 2} y={22} fill={textDim} fontSize={10}
        textAnchor="middle" fontFamily="monospace" fontWeight={700} letterSpacing={1}>
        ÉTAPE 1
      </text>
      <text x={step2X + boxW / 2} y={22} fill={textDim} fontSize={10}
        textAnchor="middle" fontFamily="monospace" fontWeight={700} letterSpacing={1}>
        ÉTAPE 2
      </text>
      <text x={step3X + boxW / 2} y={22} fill={textDim} fontSize={10}
        textAnchor="middle" fontFamily="monospace" fontWeight={700} letterSpacing={1}>
        ÉTAPE 3
      </text>

      {/* Input x */}
      <text x={8} y={midY + 4} fill={green} fontSize={16} fontFamily="monospace" fontWeight={700}>x</text>

      {/* Step 1: Linear functions */}
      {neuronY.map((ny, i) => (
        <g key={`s1-${i}`}>
          <rect x={step1X + 20} y={ny - 18} width={boxW - 20} height={36} rx={5}
            fill="rgba(0,229,255,0.06)" stroke={cyan} strokeWidth={1} />
          <text x={step1X + 20 + (boxW - 20) / 2} y={ny + 4} fill={cyan} fontSize={11}
            textAnchor="middle" fontFamily="monospace">{neuronLabels[i].pre}</text>
        </g>
      ))}

      {/* Arrows step1 → step2 */}
      {neuronY.map((ny, i) => (
        <Arrow key={`a1-${i}`} x1={step1X + boxW} x2={step2X + 20} y={ny} />
      ))}

      {/* Step 2: ReLU */}
      {neuronY.map((ny, i) => (
        <g key={`s2-${i}`}>
          <rect x={step2X + 20} y={ny - 18} width={boxW - 20} height={36} rx={5}
            fill="rgba(217,70,239,0.06)" stroke={magenta} strokeWidth={1} />
          {/* Mini ReLU icon */}
          <polyline points={`${step2X + 35},${ny + 5} ${step2X + 45},${ny + 5} ${step2X + 55},${ny - 8}`}
            fill="none" stroke={magenta} strokeWidth={1.5} />
          <text x={step2X + 70} y={ny + 4} fill={magenta} fontSize={11}
            fontFamily="monospace">
            {neuronLabels[i].post} = ReLU[·]
          </text>
        </g>
      ))}

      {/* Arrows step2 → step3 */}
      {neuronY.map((ny, i) => (
        <Arrow key={`a2-${i}`} x1={step2X + boxW} x2={step3X + 20} y={ny} />
      ))}

      {/* Step 3: Linear combination */}
      <rect x={step3X + 20} y={midY - 45} width={boxW - 20} height={90} rx={6}
        fill="rgba(255,145,0,0.06)" stroke={orange} strokeWidth={1.5} />
      <text x={step3X + 20 + (boxW - 20) / 2} y={midY - 15} fill={orange} fontSize={11}
        textAnchor="middle" fontFamily="monospace" fontWeight={700}>
        y = ϕ₀
      </text>
      <text x={step3X + 20 + (boxW - 20) / 2} y={midY + 2} fill={orange} fontSize={11}
        textAnchor="middle" fontFamily="monospace">
        + ϕ₁·h₁ + ϕ₂·h₂
      </text>
      <text x={step3X + 20 + (boxW - 20) / 2} y={midY + 19} fill={orange} fontSize={11}
        textAnchor="middle" fontFamily="monospace">
        + ϕ₃·h₃
      </text>

      {/* Converging lines into step 3 */}
      {neuronY.map((ny, i) => (
        ny !== midY ? (
          <line key={`conv-${i}`} x1={step3X + 20} x2={step3X + 20}
            y1={ny} y2={ny < midY ? midY - 45 : midY + 45}
            stroke="rgba(255,255,255,0)" strokeWidth={0} />
        ) : null
      ))}

      {/* Arrow step3 → output */}
      <Arrow x1={step3X + boxW} x2={outX} y={midY} />

      {/* Output y */}
      <text x={outX + 8} y={midY + 5} fill={green} fontSize={16}
        fontFamily="monospace" fontWeight={700}>y</text>

      {/* Bottom: parameter count */}
      <text x={W / 2} y={H - 10} fill={textDim} fontSize={10}
        textAnchor="middle" fontFamily="monospace">
        10 paramètres : ϕ₀,ϕ₁,ϕ₂,ϕ₃ + θ₁₀,θ₁₁,θ₂₀,θ₂₁,θ₃₀,θ₃₁
      </text>
    </svg>
  )
}

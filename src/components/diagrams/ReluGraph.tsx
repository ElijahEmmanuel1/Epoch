import { motion } from 'framer-motion'
import { useState } from 'react'

export default function ReluGraph() {
  const [x, setX] = useState(0.5)

  // Map visualization coordinates
  // Range x: [-2, 2] -> SVG x: [20, 280]
  // Range y: [-1, 2] -> SVG y: [130, 20]
  const scaleX = (val: number) => 20 + (val + 2) * (260 / 4)
  const scaleY = (val: number) => 130 - (val + 1) * (110 / 3)

  const relu = (v: number) => Math.max(0, v)

  const handleMouseMove = (e: React.MouseEvent<SVGSVGElement>) => {
    const rect = e.currentTarget.getBoundingClientRect()
    const mouseX = e.clientX - rect.left
    // Inverse map X
    const rawX = ((mouseX - 20) / (260 / 4)) - 2
    const clampedX = Math.max(-2, Math.min(2, rawX))
    setX(clampedX)
  }

  return (
    <div style={{
      width: '100%',
      background: 'var(--bg-deep)',
      borderRadius: 'var(--radius-md)',
      border: '1px solid var(--border)',
      padding: 16,
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      gap: 12
    }}>
      <div style={{ fontSize: 14, fontFamily: 'var(--font-mono)', color: 'var(--text-primary)' }}>
        ReLU({x.toFixed(2)}) = <span style={{ color: 'var(--cyan)' }}>{relu(x).toFixed(2)}</span>
      </div>

      <svg
        viewBox="0 0 300 150"
        style={{ width: '100%', maxWidth: 300, overflow: 'visible', cursor: 'crosshair' }}
        onMouseMove={handleMouseMove}
        onMouseLeave={() => setX(0.5)}
      >
        {/* Axes */}
        <line x1="20" y1={scaleY(0)} x2="280" y2={scaleY(0)} stroke="var(--text-muted)" strokeWidth="1" />
        <line x1={scaleX(0)} y1="140" x2={scaleX(0)} y2="20" stroke="var(--text-muted)" strokeWidth="1" />

        {/* Labels */}
        <text x="280" y={scaleY(0) + 15} fill="var(--text-muted)" fontSize="10">x</text>
        <text x={scaleX(0) - 15} y="15" fill="var(--text-muted)" fontSize="10">y</text>

        {/* ReLU Function */}
        <motion.polyline
          points={`${scaleX(-2)},${scaleY(0)} ${scaleX(0)},${scaleY(0)} ${scaleX(2)},${scaleY(2)}`}
          fill="none"
          stroke="var(--cyan)"
          strokeWidth="3"
          strokeLinecap="round"
          filter="drop-shadow(0 0 5px var(--cyan-glow))"
          initial={{ pathLength: 0 }}
          animate={{ pathLength: 1 }}
          transition={{ duration: 1 }}
        />

        {/* Interactive Point */}
        <circle
          cx={scaleX(x)}
          cy={scaleY(relu(x))}
          r="6"
          fill="var(--bg-deep)"
          stroke="var(--text-primary)"
          strokeWidth="2"
        />

        {/* Dotted lines to axes */}
        <line
          x1={scaleX(x)} y1={scaleY(relu(x))}
          x2={scaleX(x)} y2={scaleY(0)}
          stroke="var(--cyan)"
          strokeDasharray="2 2"
          opacity="0.5"
        />
        {x > 0 && (
          <line
            x1={scaleX(x)} y1={scaleY(relu(x))}
            x2={scaleX(0)} y2={scaleY(relu(x))}
            stroke="var(--cyan)"
            strokeDasharray="2 2"
            opacity="0.5"
          />
        )}

      </svg>
      <div style={{ fontSize: 12, color: 'var(--text-secondary)', textAlign: 'center' }}>
        Survolez le graphe pour tester la fonction ! z &lt; 0 → 0
      </div>
    </div>
  )
}

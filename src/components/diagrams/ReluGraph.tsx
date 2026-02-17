import { motion } from 'framer-motion'
import { useState } from 'react'

export default function ReluGraph() {
  const [x, setX] = useState(0.5)

  const scaleX = (val: number) => 20 + (val + 2) * (260 / 4)
  const scaleY = (val: number) => 130 - (val + 1) * (110 / 3)

  const relu = (v: number) => Math.max(0, v)

  const handleMouseMove = (e: React.MouseEvent<SVGSVGElement>) => {
    const rect = e.currentTarget.getBoundingClientRect()
    const mouseX = e.clientX - rect.left
    const rawX = ((mouseX - 20) / (260 / 4)) - 2
    const clampedX = Math.max(-2, Math.min(2, rawX))
    setX(clampedX)
  }

  return (
    <div className="w-full bg-[var(--bg-base)] rounded-[var(--radius-md)] border border-[var(--border-default)] p-4 flex flex-col items-center gap-3">
      <div className="text-sm font-mono text-[var(--text-primary)]">
        ReLU({x.toFixed(2)}) = <span className="text-[var(--color-accent)]">{relu(x).toFixed(2)}</span>
      </div>

      <svg
        viewBox="0 0 300 150"
        className="w-full max-w-[300px] overflow-visible cursor-crosshair"
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
          stroke="var(--color-accent)"
          strokeWidth="3"
          strokeLinecap="round"
          filter="drop-shadow(0 0 5px var(--color-accent))"
          initial={{ pathLength: 0 }}
          animate={{ pathLength: 1 }}
          transition={{ duration: 1 }}
        />

        {/* Interactive Point */}
        <circle
          cx={scaleX(x)}
          cy={scaleY(relu(x))}
          r="6"
          fill="var(--bg-base)"
          stroke="var(--text-primary)"
          strokeWidth="2"
        />

        {/* Dotted lines to axes */}
        <line
          x1={scaleX(x)} y1={scaleY(relu(x))}
          x2={scaleX(x)} y2={scaleY(0)}
          stroke="var(--color-accent)"
          strokeDasharray="2 2"
          opacity="0.5"
        />
        {x > 0 && (
          <line
            x1={scaleX(x)} y1={scaleY(relu(x))}
            x2={scaleX(0)} y2={scaleY(relu(x))}
            stroke="var(--color-accent)"
            strokeDasharray="2 2"
            opacity="0.5"
          />
        )}
      </svg>
      <div className="text-xs text-[var(--text-secondary)] text-center">
        Survolez le graphe pour tester la fonction ! z &lt; 0 → 0
      </div>
    </div>
  )
}

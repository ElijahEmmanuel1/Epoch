import { motion } from 'framer-motion'
import { useState } from 'react'

export default function SlopeConstructionGraph() {
  const [stage, setStage] = useState(0)

  const stageButtons = [
    { label: '1. ReLU(x)', color: 'var(--color-accent)' },
    { label: '2. -ReLU(x-1)', color: 'var(--color-primary)' },
    { label: '3. Slope', color: 'var(--color-success)' },
  ]

  return (
    <div className="w-full bg-[var(--bg-base)] rounded-[var(--radius-md)] border border-[var(--border-default)] p-4 flex flex-col items-center gap-3">
      <div className="flex gap-2.5 text-xs">
        {stageButtons.map((btn, i) => (
          <button
            key={i}
            onClick={() => setStage(i)}
            className="transition-opacity font-mono"
            style={{ opacity: stage === i ? 1 : 0.5, color: btn.color }}
          >
            {btn.label}
          </button>
        ))}
      </div>

      <svg viewBox="0 0 300 150" className="w-full max-w-[300px] overflow-visible">
        {/* Axes */}
        <line x1="20" y1="130" x2="280" y2="130" stroke="var(--text-muted)" strokeWidth="1" />
        <line x1="40" y1="140" x2="40" y2="20" stroke="var(--text-muted)" strokeWidth="1" />

        {/* Labels */}
        <text x="280" y="145" fill="var(--text-muted)" fontSize="10">x</text>
        <text x="40" y="15" fill="var(--text-muted)" fontSize="10">y</text>
        <text x="40" y="145" fill="var(--text-muted)" fontSize="10">0</text>
        <text x="90" y="145" fill="var(--text-muted)" fontSize="10">1</text>
        <text x="140" y="145" fill="var(--text-muted)" fontSize="10">2</text>

        {/* Stage 0: ReLU(x) */}
        {(stage === 0 || stage === 2) && (
          <motion.polyline
            points="20,130 40,130 280,10"
            fill="none"
            stroke="var(--color-accent)"
            strokeWidth={stage === 2 ? 1 : 2}
            strokeDasharray={stage === 2 ? "4 4" : "none"}
            opacity={stage === 2 ? 0.3 : 1}
            initial={{ pathLength: 0 }}
            animate={{ pathLength: 1 }}
          />
        )}

        {/* Stage 1: -ReLU(x-1) */}
        {(stage === 1 || stage === 2) && (
          <motion.polyline
            points="20,130 90,130 280,225"
            fill="none"
            stroke="var(--color-primary)"
            strokeWidth={stage === 2 ? 1 : 2}
            strokeDasharray={stage === 2 ? "4 4" : "none"}
            opacity={stage === 2 ? 0.3 : 1}
            initial={{ pathLength: 0 }}
            animate={{ pathLength: 1 }}
          />
        )}

        {/* Stage 2: Sum */}
        {stage === 2 && (
          <motion.polyline
            points="20,130 40,130 90,105 280,105"
            fill="none"
            stroke="var(--color-success)"
            strokeWidth="3"
            strokeLinecap="round"
            filter="drop-shadow(0 0 4px var(--color-success))"
            initial={{ pathLength: 0 }}
            animate={{ pathLength: 1 }}
          />
        )}

        {/* Dashed line for slope change */}
        <line x1="90" y1="130" x2="90" y2="105" stroke="var(--border-subtle)" strokeDasharray="2 2" />
      </svg>

      <div className="text-xs text-[var(--text-secondary)] text-center">
        {stage === 0 && "ReLU(x) : Une rampe qui commence à 0"}
        {stage === 1 && "-ReLU(x-1) : Une rampe négative décalée de 1"}
        {stage === 2 && "Somme : La pente monte puis s'annule -> Plateau !"}
      </div>
    </div>
  )
}

import { motion } from 'framer-motion'
import { useState } from 'react'

export default function SlopeConstructionGraph() {
    const [stage, setStage] = useState(0) // 0=ReLU1, 1=ReLU2, 2=Combined

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
            <div style={{ display: 'flex', gap: 10, fontSize: 12 }}>
                <button onClick={() => setStage(0)} style={{ opacity: stage === 0 ? 1 : 0.5, color: 'var(--cyan)' }}>1. ReLU(x)</button>
                <button onClick={() => setStage(1)} style={{ opacity: stage === 1 ? 1 : 0.5, color: 'var(--magenta)' }}>2. -ReLU(x-1)</button>
                <button onClick={() => setStage(2)} style={{ opacity: stage === 2 ? 1 : 0.5, color: 'var(--green)' }}>3. Slope</button>
            </div>

            <svg viewBox="0 0 300 150" style={{ width: '100%', maxWidth: 300, overflow: 'visible' }}>
                {/* Axes */}
                <line x1="20" y1="130" x2="280" y2="130" stroke="var(--text-muted)" strokeWidth="1" />
                <line x1="40" y1="140" x2="40" y2="20" stroke="var(--text-muted)" strokeWidth="1" />

                {/* Labels */}
                <text x="280" y="145" fill="var(--text-muted)" fontSize="10">x</text>
                <text x="40" y="15" fill="var(--text-muted)" fontSize="10">y</text>
                <text x="40" y="145" fill="var(--text-muted)" fontSize="10">0</text>
                <text x="90" y="145" fill="var(--text-muted)" fontSize="10">1</text>
                <text x="140" y="145" fill="var(--text-muted)" fontSize="10">2</text>

                {/* Graphs */}
                {/* Stage 0: ReLU(x) */}
                {(stage === 0 || stage === 2) && (
                    <motion.polyline
                        points="20,130 40,130 280,10"
                        fill="none"
                        stroke="var(--cyan)"
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
                        points="20,130 90,130 280,225" // Descends
                        fill="none"
                        stroke="var(--magenta)"
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
                        stroke="var(--green)"
                        strokeWidth="3"
                        strokeLinecap="round"
                        filter="drop-shadow(0 0 4px var(--green-glow))"
                        initial={{ pathLength: 0 }}
                        animate={{ pathLength: 1 }}
                    />
                )}

                {/* Dashed lines for slope change */}
                <line x1="90" y1="130" x2="90" y2="105" stroke="var(--divider)" strokeDasharray="2 2" />

            </svg>
            <div style={{ fontSize: 12, color: 'var(--text-secondary)', textAlign: 'center' }}>
                {stage === 0 && "ReLU(x) : Une rampe qui commence à 0"}
                {stage === 1 && "-ReLU(x-1) : Une rampe négative décalée de 1"}
                {stage === 2 && "Somme : La pente monte puis s'annule -> Plateau !"}
            </div>
        </div>
    )
}

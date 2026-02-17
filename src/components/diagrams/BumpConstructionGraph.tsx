import { motion } from 'framer-motion'
import { useState } from 'react'

export default function BumpConstructionGraph() {
    const [stage, setStage] = useState(0) // 0=Slope1, 1=Slope2, 2=Combined

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
                <button onClick={() => setStage(0)} style={{ opacity: stage === 0 ? 1 : 0.5, color: 'var(--cyan)' }}>1. Pente Montante</button>
                <button onClick={() => setStage(1)} style={{ opacity: stage === 1 ? 1 : 0.5, color: 'var(--magenta)' }}>2. Pente Descendante</button>
                <button onClick={() => setStage(2)} style={{ opacity: stage === 2 ? 1 : 0.5, color: 'var(--green)' }}>3. Bosse (Bump)</button>
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
                <text x="190" y="145" fill="var(--text-muted)" fontSize="10">3</text>

                {/* Graphs */}
                {/* Stage 0: Slope Up (0 to 1) */}
                {(stage === 0 || stage === 2) && (
                    <motion.polyline
                        points="20,130 40,130 90,80 280,80"
                        fill="none"
                        stroke="var(--cyan)"
                        strokeWidth={stage === 2 ? 1 : 2}
                        strokeDasharray={stage === 2 ? "4 4" : "none"}
                        opacity={stage === 2 ? 0.3 : 1}
                        initial={{ pathLength: 0 }}
                        animate={{ pathLength: 1 }}
                    />
                )}

                {/* Stage 1: Slope Down (starts at 2) */}
                {(stage === 1 || stage === 2) && (
                    <motion.polyline
                        points="20,130 140,130 190,80 280,80"
                        fill="none"
                        stroke="var(--magenta)"
                        strokeWidth={stage === 2 ? 1 : 2}
                        strokeDasharray={stage === 2 ? "4 4" : "none"}
                        opacity={stage === 2 ? 0.3 : 1}
                        initial={{ pathLength: 0 }}
                        animate={{ pathLength: 1 }}
                    />
                )}

                {/* Stage 2: Sum (Subtraction actually: Up - Down) */}
                {/* Up: stays at 1 after x=1. Down: stays at 0 until x=2, then rises to 1 at x=3.
            So Up - Down:
            x < 1: 0 - 0 = 0
            1 < x < 2: rises to 1 - 0 = 1
            2 < x < 3: 1 - (rise to 1) = falls to 0
            x > 3: 1 - 1 = 0
        */}
                {stage === 2 && (
                    <motion.path
                        d="M 20 130 L 40 130 L 90 80 L 140 80 L 190 130 L 280 130"
                        fill="var(--green-glow)"
                        stroke="var(--green)"
                        strokeWidth="3"
                        strokeLinecap="round"
                        filter="drop-shadow(0 0 6px var(--green))"
                        initial={{ pathLength: 0 }}
                        animate={{ pathLength: 1 }}
                    />
                )}

                {/* Dashed lines */}
                <line x1="90" y1="130" x2="90" y2="80" stroke="var(--divider)" strokeDasharray="2 2" />
                <line x1="140" y1="130" x2="140" y2="80" stroke="var(--divider)" strokeDasharray="2 2" />
                <line x1="190" y1="130" x2="190" y2="80" stroke="var(--divider)" strokeDasharray="2 2" />

            </svg>
            <div style={{ fontSize: 12, color: 'var(--text-secondary)', textAlign: 'center' }}>
                {stage === 0 && "Unités 1 & 2 : Créent un plateau positif"}
                {stage === 1 && "Unités 3 & 4 : Créent un plateau décalé"}
                {stage === 2 && "Soustraction : On garde seulement la partie centrale -> Une BOSSE !"}
            </div>
        </div>
    )
}

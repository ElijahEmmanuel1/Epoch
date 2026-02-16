import { motion } from 'framer-motion'
import styles from './NeuralDiagram.module.css'

interface Props {
  courseId: string
  highlightedVar: string | null
}

interface NeuronLayer {
  label: string
  count: number
  color: string
  varName?: string
}

const diagrams: Record<string, NeuronLayer[]> = {
  'supervised-learning': [
    { label: 'Input x', count: 2, color: 'var(--cyan)', varName: 'x' },
    { label: 'f[x,ϕ]', count: 1, color: 'var(--magenta)', varName: 'y' },
    { label: 'Loss L', count: 1, color: 'var(--orange)', varName: 'loss' },
  ],
  tensors: [
    { label: 'Scalaire', count: 1, color: 'var(--cyan)' },
    { label: 'Vecteur', count: 3, color: 'var(--magenta)' },
    { label: 'Matrice', count: 4, color: 'var(--orange)', varName: 'result' },
  ],
  'shallow-networks': [
    { label: 'Input x', count: 1, color: 'var(--cyan)', varName: 'x' },
    { label: 'ReLU h', count: 3, color: 'var(--magenta)', varName: 'relu' },
    { label: 'Output y', count: 1, color: 'var(--green)', varName: 'output' },
  ],
  'deep-networks': [
    { label: 'Input', count: 3, color: 'var(--cyan)' },
    { label: 'h₁', count: 4, color: 'var(--magenta)', varName: 'hidden' },
    { label: 'h₂', count: 4, color: 'var(--magenta)', varName: 'hidden' },
    { label: 'h₃', count: 4, color: 'var(--magenta)', varName: 'hidden' },
    { label: 'Output', count: 2, color: 'var(--green)' },
  ],
  'loss-functions': [
    { label: 'Prediction ŷ', count: 3, color: 'var(--cyan)' },
    { label: 'Loss L', count: 1, color: 'var(--orange)', varName: 'loss' },
  ],
  'gradient-descent': [
    { label: 'ϕ_t', count: 2, color: 'var(--magenta)' },
    { label: 'α·∇L', count: 2, color: 'var(--orange)', varName: 'grad' },
    { label: 'ϕ_{t+1}', count: 2, color: 'var(--green)' },
  ],
  backprop: [
    { label: 'Input', count: 2, color: 'var(--cyan)' },
    { label: 'Hidden', count: 3, color: 'var(--magenta)' },
    { label: 'Output', count: 1, color: 'var(--green)' },
    { label: '∇ Loss', count: 1, color: 'var(--orange)', varName: 'grad' },
  ],
  regularization: [
    { label: 'Input', count: 3, color: 'var(--cyan)' },
    { label: 'Dropout', count: 4, color: 'var(--magenta)' },
    { label: 'Output', count: 2, color: 'var(--green)' },
  ],
  cnn: [
    { label: 'Input', count: 4, color: 'var(--cyan)' },
    { label: 'Conv', count: 3, color: 'var(--magenta)' },
    { label: 'Pool', count: 2, color: 'var(--orange)' },
    { label: 'FC', count: 2, color: 'var(--green)' },
  ],
  resnet: [
    { label: 'Input', count: 3, color: 'var(--cyan)' },
    { label: 'Conv', count: 3, color: 'var(--magenta)', varName: 'hidden' },
    { label: '+Skip', count: 3, color: 'var(--orange)', varName: 'hidden' },
    { label: 'Output', count: 2, color: 'var(--green)' },
  ],
  rnn: [
    { label: 'x_t', count: 2, color: 'var(--cyan)' },
    { label: 'h_t', count: 3, color: 'var(--magenta)' },
    { label: 'y_t', count: 2, color: 'var(--green)' },
  ],
  attention: [
    { label: 'Q', count: 3, color: 'var(--cyan)' },
    { label: 'K', count: 3, color: 'var(--magenta)' },
    { label: 'V', count: 3, color: 'var(--orange)' },
    { label: 'Attn', count: 3, color: 'var(--green)' },
  ],
  gan: [
    { label: 'Noise z', count: 2, color: 'var(--cyan)' },
    { label: 'G(z)', count: 3, color: 'var(--magenta)' },
    { label: 'D(x)', count: 1, color: 'var(--orange)' },
  ],
  diffusion: [
    { label: 'x₀', count: 3, color: 'var(--cyan)' },
    { label: '→ noise', count: 3, color: 'var(--magenta)' },
    { label: 'denoise', count: 3, color: 'var(--orange)' },
    { label: 'x̂₀', count: 3, color: 'var(--green)' },
  ],
}

export default function NeuralDiagram({ courseId, highlightedVar }: Props) {
  const layers = diagrams[courseId] || diagrams['supervised-learning']
  const svgWidth = 420
  const svgHeight = 180
  const layerGap = svgWidth / (layers.length + 1)

  return (
    <div className={styles.diagram}>
      <div className={styles.label}>Architecture Visualisation</div>
      <svg
        viewBox={`0 0 ${svgWidth} ${svgHeight}`}
        className={styles.svg}
      >
        {/* Connections */}
        {layers.map((layer, li) => {
          if (li === 0) return null
          const prevLayer = layers[li - 1]
          const x1 = layerGap * li
          const x2 = layerGap * (li + 1)
          const connections = []
          for (let pi = 0; pi < prevLayer.count; pi++) {
            for (let ci = 0; ci < layer.count; ci++) {
              const y1 = getNodeY(pi, prevLayer.count, svgHeight)
              const y2 = getNodeY(ci, layer.count, svgHeight)
              const isHighlighted = highlightedVar && (
                layer.varName === highlightedVar || prevLayer.varName === highlightedVar
              )
              connections.push(
                <motion.line
                  key={`${li}-${pi}-${ci}`}
                  x1={x1} y1={y1}
                  x2={x2} y2={y2}
                  stroke={isHighlighted ? layer.color : 'var(--border-active)'}
                  strokeWidth={isHighlighted ? 1.5 : 0.5}
                  strokeOpacity={isHighlighted ? 0.8 : 0.3}
                  initial={{ pathLength: 0 }}
                  animate={{ pathLength: 1 }}
                  transition={{ duration: 0.8, delay: li * 0.15 }}
                />
              )
            }
          }
          return connections
        })}

        {/* Nodes */}
        {layers.map((layer, li) => {
          const x = layerGap * (li + 1)
          return Array.from({ length: layer.count }, (_, ni) => {
            const y = getNodeY(ni, layer.count, svgHeight)
            const isHighlighted = highlightedVar && layer.varName === highlightedVar
            return (
              <g key={`node-${li}-${ni}`}>
                {/* Glow */}
                {isHighlighted && (
                  <motion.circle
                    cx={x} cy={y} r={14}
                    fill={layer.color}
                    opacity={0.15}
                    animate={{ r: [14, 18, 14], opacity: [0.15, 0.3, 0.15] }}
                    transition={{ duration: 2, repeat: Infinity }}
                  />
                )}
                {/* Node */}
                <motion.circle
                  cx={x} cy={y}
                  r={isHighlighted ? 8 : 6}
                  fill={isHighlighted ? layer.color : 'var(--bg-deep)'}
                  stroke={layer.color}
                  strokeWidth={isHighlighted ? 2.5 : 1.5}
                  initial={{ scale: 0 }}
                  animate={{ scale: 1 }}
                  transition={{ duration: 0.4, delay: li * 0.1 + ni * 0.03 }}
                />
              </g>
            )
          })
        })}

        {/* Layer Labels */}
        {layers.map((layer, li) => {
          const x = layerGap * (li + 1)
          return (
            <text
              key={`label-${li}`}
              x={x}
              y={svgHeight - 8}
              textAnchor="middle"
              fill="var(--text-muted)"
              fontSize="9"
              fontFamily="var(--font-mono)"
            >
              {layer.label}
            </text>
          )
        })}
      </svg>
    </div>
  )
}

function getNodeY(index: number, total: number, height: number): number {
  const usableHeight = height - 50
  if (total === 1) return usableHeight / 2 + 10
  const gap = usableHeight / (total + 1)
  return gap * (index + 1) + 10
}

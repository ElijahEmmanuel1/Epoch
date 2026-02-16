import { motion } from 'framer-motion'
import type { ConsoleEntry } from './CodeReactor'
import styles from './TensorMonitor.module.css'

interface Props {
  entries: ConsoleEntry[]
  isRunning: boolean
}

export default function TensorMonitor({ entries, isRunning }: Props) {
  if (isRunning) {
    return (
      <div className={styles.monitor}>
        <div className={styles.running}>
          <div className={styles.dots}>
            <span /><span /><span />
          </div>
          <span>Executing neural computation…</span>
        </div>
      </div>
    )
  }

  if (entries.length === 0) {
    return (
      <div className={styles.monitor}>
        <div className={styles.empty}>
          <span className={styles.prompt}>{'>'}</span>
          <span className={styles.emptyText}>
            Exécutez votre code pour voir les résultats ici
          </span>
        </div>
      </div>
    )
  }

  return (
    <div className={styles.monitor}>
      {entries.map((entry, i) => (
        <motion.div
          key={i}
          className={`${styles.entry} ${styles[entry.type]}`}
          initial={{ opacity: 0, x: -8 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: i * 0.05 }}
        >
          <div className={styles.entryContent}>
            <span className={styles.prompt}>
              {entry.type === 'error' ? '✗' : entry.type === 'info' ? 'ℹ' : '>'}
            </span>
            <span className={styles.entryText}>{entry.text}</span>
          </div>

          {/* Tensor Heatmap visualization */}
          {entry.type === 'tensor' && entry.values && (
            <div className={styles.heatmap}>
              <div className={styles.heatmapLabel}>
                Tensor Heatmap — Shape: {entry.shape}
              </div>
              <div className={styles.heatmapGrid}>
                {entry.values.map((row, ri) => (
                  <div key={ri} className={styles.heatmapRow}>
                    {row.map((val, ci) => (
                      <div
                        key={ci}
                        className={styles.heatmapCell}
                        style={{
                          background: getHeatmapColor(val),
                          opacity: Math.abs(val) * 0.5 + 0.5,
                        }}
                        title={val.toFixed(4)}
                      />
                    ))}
                  </div>
                ))}
              </div>
              <div className={styles.heatmapScale}>
                <span>-1.0</span>
                <div className={styles.scaleBar} />
                <span>+1.0</span>
              </div>
            </div>
          )}
        </motion.div>
      ))}
    </div>
  )
}

function getHeatmapColor(value: number): string {
  // Negative → cyan, Zero → dark, Positive → magenta
  if (value < 0) {
    const intensity = Math.min(Math.abs(value), 1)
    return `rgba(0, 229, 255, ${intensity * 0.8})`
  } else {
    const intensity = Math.min(value, 1)
    return `rgba(217, 70, 239, ${intensity * 0.8})`
  }
}

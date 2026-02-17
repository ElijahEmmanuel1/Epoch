import { motion } from 'framer-motion'
import { clsx } from 'clsx'
import type { ConsoleEntry } from './CodeReactor'

interface Props {
  entries: ConsoleEntry[]
  isRunning: boolean
}

export default function TensorMonitor({ entries, isRunning }: Props) {
  if (isRunning) {
    return (
      <div className="px-4 py-3 font-mono text-xs">
        <div className="flex items-center gap-2.5 text-[var(--text-muted)]">
          <div className="flex gap-1">
            <span className="w-1 h-1 rounded-full bg-primary animate-bounce" style={{ animationDelay: '0ms' }} />
            <span className="w-1 h-1 rounded-full bg-primary animate-bounce" style={{ animationDelay: '150ms' }} />
            <span className="w-1 h-1 rounded-full bg-primary animate-bounce" style={{ animationDelay: '300ms' }} />
          </div>
          <span>Exécution en cours…</span>
        </div>
      </div>
    )
  }

  if (entries.length === 0) {
    return (
      <div className="px-4 py-3 font-mono text-xs">
        <div className="flex items-center gap-2 text-[var(--text-muted)]">
          <span className="text-[var(--text-muted)]">{'>'}</span>
          <span className="italic">Exécutez votre code pour voir les résultats ici</span>
        </div>
      </div>
    )
  }

  return (
    <div className="px-4 py-3 font-mono text-xs space-y-2">
      {entries.map((entry, i) => (
        <motion.div
          key={i}
          initial={{ opacity: 0, x: -6 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: i * 0.05 }}
        >
          <div className="flex items-start gap-2 leading-relaxed">
            <span className={clsx(
              'shrink-0 select-none',
              entry.type === 'error' ? 'text-[var(--color-error)]' :
              entry.type === 'info' ? 'text-[var(--color-success)]' :
              entry.type === 'tensor' ? 'text-[var(--color-accent)]' :
              'text-[var(--color-success)]'
            )}>
              {entry.type === 'error' ? '✗' : entry.type === 'info' ? 'ℹ' : '>'}
            </span>
            <span className={clsx(
              'break-words',
              entry.type === 'error' ? 'text-[var(--color-error)]' :
              entry.type === 'info' ? 'text-[var(--color-success)]/70' :
              entry.type === 'tensor' ? 'text-[var(--color-accent)]' :
              'text-[var(--text-primary)]'
            )}>{entry.text}</span>
          </div>

          {entry.type === 'tensor' && entry.values && (
            <div className="mt-2 ml-4 p-2.5 bg-[var(--bg-elevated)] border border-[var(--border-default)] rounded-lg inline-block transition-colors">
              <div className="text-[9px] font-semibold tracking-wider uppercase text-[var(--text-muted)] mb-2">
                Heatmap — Shape: {entry.shape}
              </div>
              <div className="flex flex-col gap-0.5">
                {entry.values.map((row, ri) => (
                  <div key={ri} className="flex gap-0.5">
                    {row.map((val, ci) => (
                      <div
                        key={ci}
                        className="w-5 h-5 rounded-sm cursor-crosshair transition-transform hover:scale-125 hover:z-10"
                        style={{
                          background: val < 0
                            ? `rgba(99, 102, 241, ${Math.min(Math.abs(val), 1) * 0.8})`
                            : `rgba(6, 182, 212, ${Math.min(val, 1) * 0.8})`,
                        }}
                        title={val.toFixed(4)}
                      />
                    ))}
                  </div>
                ))}
              </div>
              <div className="flex items-center gap-1.5 mt-2 text-[9px] text-[var(--text-muted)]">
                <span>-1.0</span>
                <div className="flex-1 h-1 rounded-full bg-gradient-to-r from-primary via-transparent to-accent" />
                <span>+1.0</span>
              </div>
            </div>
          )}
        </motion.div>
      ))}
    </div>
  )
}
